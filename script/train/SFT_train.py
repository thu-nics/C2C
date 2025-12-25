"""
Use SFT trainer to train rosetta model
"""

import gc
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys
import json
import yaml
import argparse
import shutil
import wandb
import torch.distributed as dist  # Added for Distributed Data Parallel support
from torch.nn.parallel import DistributedDataParallel  # For type checking
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import math
import contextlib

from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import create_projector, save_projector
from rosetta.train.dataset_adapters import ChatDataset, AlignedChatDataset, RosettaDataCollator, create_dataset, BaselineDataCollator, BaselineChatDataset
from rosetta.model.aligner import TokenAligner, AlignmentStrategy
from rosetta.train.model_utils import k_nearest_sources, last_aligned_sources
from rosetta.model.projector import AllInOneProjector
from rosetta.utils.evaluate import set_default_chat_template

# PEFT imports for LoRA (baseline mode)
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

torch.autograd.set_detect_anomaly(True)

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

def enable_full_determinism():
    """Enable stricter determinism settings for reproducibility."""
    # Must be set before CUDA context creation for cuBLAS determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # PyTorch deterministic algorithms (may raise if non-deterministic ops are used)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    # Disable TF32 to reduce numeric variability
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

def broadcast_decision_from_rank0(decision: bool, distributed: bool, device: str, rank: int) -> bool:
    """Broadcast a boolean decision from rank 0 to all ranks so control flow matches."""
    if not distributed:
        return decision
    if rank == 0:
        tensor_flag = torch.tensor([1 if decision else 0], device=device, dtype=torch.int)
    else:
        tensor_flag = torch.empty(1, device=device, dtype=torch.int)
    dist.broadcast(tensor_flag, src=0)
    return bool(tensor_flag.item())

def freeze_model(model: nn.Module):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model: nn.Module):
    """Unfreeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_projectors(rosetta_model: RosettaModel):
    """Unfreeze only the projector parameters"""
    for projector in rosetta_model.projector_list:
        for param in projector.parameters():
            param.requires_grad = True


def detect_training_mode(model_config: Dict[str, Any]) -> str:
    """Detect whether to use baseline or Rosetta training based on config"""
    if "baseline_model" in model_config and "base_model" not in model_config:
        return "baseline"
    elif "base_model" in model_config and "teacher_model" in model_config:
        return "rosetta"
    else:
        raise ValueError("Invalid model configuration. Provide either 'baseline_model' for baseline training "
                       "or both 'base_model' and 'teacher_model' for Rosetta training.")


def setup_lora_model(model: nn.Module, lora_config: Dict[str, Any]) -> nn.Module:
    """Setup LoRA for the model"""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required for LoRA training. Install with: pip install peft")
    
    # Default LoRA configuration
    default_config = {
        "r": 16,  # LoRA rank
        "lora_alpha": 32,  # LoRA scaling parameter
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    # Update with user config
    config = {**default_config, **lora_config}
    
    # Create LoRA config
    peft_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        task_type=getattr(TaskType, config["task_type"])
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    return model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file based on file extension"""
    file_ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, "r") as f:
        if file_ext == ".json":
            config = json.load(f)
        elif file_ext in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}. Supported formats: .json, .yaml, .yml")

    return config


def setup_partial_training(model: nn.Module, partial_config: Dict[str, Any]) -> nn.Module:
    """Setup partial parameter training (alternative to LoRA)"""
    method = partial_config.get("method", "layer_wise")
    ratio = partial_config.get("ratio", 0.6)  # 60% of parameters
    
    if method == "layer_wise":
        # Freeze/unfreeze entire layers
        total_layers = len(model.model.layers)
        layers_to_train = int(total_layers * ratio)
        
        # Freeze all parameters first
        freeze_model(model)
        
        # Unfreeze the last N layers
        for i in range(total_layers - layers_to_train, total_layers):
            unfreeze_model(model.model.layers[i])
            
        # Also unfreeze the final layer norm and lm_head
        if hasattr(model.model, 'norm'):
            unfreeze_model(model.model.norm)
        if hasattr(model, 'lm_head'):
            unfreeze_model(model.lm_head)
            
        print(f"Training last {layers_to_train} layers out of {total_layers} total layers")
        
    elif method == "parameter_wise":
        # Freeze/unfreeze based on parameter importance or random selection
        all_params = list(model.named_parameters())
        total_params = len(all_params)
        params_to_train = int(total_params * ratio)
        
        # Freeze all first
        freeze_model(model)
        
        # Unfreeze last N parameters (you can implement more sophisticated selection)
        for name, param in all_params[-params_to_train:]:
            param.requires_grad = True
            
        print(f"Training {params_to_train} parameters out of {total_params} total parameters")
    
    return model


def build_layer_mapping(n_target=28, n_source=36):

    source_positions = [i / (n_source - 1) for i in range(n_source)]
    target_positions = [j / (n_target - 1) for j in range(n_target)]

    mapping = []
    for i, sp in enumerate(target_positions):
        closest_j = min(range(n_source), key=lambda j: abs(source_positions[j] - sp))
        mapping.append((i, closest_j))

    return mapping

def build_shared_mlp(source_dim: int, hidden_dim: int, target_dim: int, num_layers: int, 
                use_layer_norm: bool, dropout: float, dtype: torch.dtype) -> nn.Sequential:
    """Build a single MLP projection module"""
    layers = []
        
    # Input projection
    layers.append(nn.Linear(source_dim, hidden_dim, dtype=dtype))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim, dtype=dtype))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout))
        
    # Hidden layers
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, dtype=dtype))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
    # Output projection
    if num_layers > 1:
        layers.append(nn.Linear(hidden_dim, target_dim, dtype=dtype))
    else:
        # Single layer case
        layers = [nn.Linear(source_dim, target_dim, dtype=dtype)]
        
    return nn.Sequential(*layers)
    
def setup_models(model_config: Dict[str, Any], training_mode: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Setup models based on training mode (baseline or rosetta)"""
    
    if training_mode == "baseline":
        # Baseline mode: single model training
        model_name = model_config["baseline_model"]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        set_default_chat_template(tokenizer, model_name)
        
        # Load baseline model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=model_config.get("attn_implementation", None)
        )
        
        return model, tokenizer, None, None
    
    else:  # rosetta mode
        # Load tokenizer (use base model tokenizer)
        slm_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])

        if slm_tokenizer.pad_token is None:
            slm_tokenizer.pad_token = slm_tokenizer.eos_token
            slm_tokenizer.pad_token_id = slm_tokenizer.eos_token_id
        set_default_chat_template(slm_tokenizer, model_config["base_model"])
        
        # Load LLM tokenizer if alignment is enabled
        llm_tokenizer = None
        if model_config.get("is_do_alignment", False):
            llm_tokenizer = AutoTokenizer.from_pretrained(model_config["teacher_model"])
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
                llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
            set_default_chat_template(llm_tokenizer, model_config["teacher_model"])

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config["base_model"],
            torch_dtype=dtype,
            attn_implementation=model_config.get("attn_implementation", None)
        )
        
        # Load teacher model  
        if model_config["teacher_model"] == "google/gemma-3-1b-it":
            teacher_model = AutoModelForCausalLM.from_pretrained(
                model_config["teacher_model"],
                torch_dtype=dtype,
                attn_implementation=model_config.get("attn_implementation", None),
                sliding_window=4096
            )
        else:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                model_config["teacher_model"],
                torch_dtype=dtype,
                attn_implementation=model_config.get("attn_implementation", None)
            )
        
        # Get model dimensions and layer counts
        base_dim = int(base_model.model.layers[0].self_attn.k_proj.out_features / base_model.config.num_key_value_heads)
        teacher_dim = int(teacher_model.model.layers[0].self_attn.k_proj.out_features / teacher_model.config.num_key_value_heads)
        base_num_heads = base_model.config.num_key_value_heads
        teacher_num_heads = teacher_model.config.num_key_value_heads
        slm_num_layers = base_model.config.num_hidden_layers
        llm_num_layers = teacher_model.config.num_hidden_layers
        
        # Create projector from config
        projector_config = model_config["projector"]
        projector_params = projector_config["params"].copy()
        projector_params["dtype"] = dtype
        projector_list = []
        # Only M projectors (share projector across sources): one per target layer
        num_projectors = slm_num_layers

        # shared_key_projection=build_shared_mlp(
        #     source_dim=teacher_dim,
        #     hidden_dim=projector_params["hidden_dim"],
        #     target_dim=base_dim,
        #     num_layers=projector_params["num_layers"],
        #     use_layer_norm=projector_params["use_layer_norm"],
        #     dropout=projector_params["dropout"],
        #     dtype=dtype
        # )
        # shared_value_projection=build_shared_mlp(
        #     source_dim=teacher_dim,
        #     hidden_dim=projector_params["hidden_dim"],
        #     target_dim=base_dim,
        #     num_layers=projector_params["num_layers"],
        #     use_layer_norm=projector_params["use_layer_norm"],
        #     dropout=projector_params["dropout"],
        #     dtype=dtype
        # )
        for _ in range(num_projectors):
            projector = create_projector(
                projector_config["type"],
                source_dim=teacher_dim,
                target_dim=base_dim,
                source_num_heads=teacher_num_heads,
                target_num_heads=base_num_heads,
                # shared_key_projection=shared_key_projection,
                # shared_value_projection=shared_value_projection,
                **projector_params
            )
            projector_list.append(projector.to(device))
        
        # Init RosettaModel
        K = 1

        rosetta_model = RosettaModel(
            model_list=[base_model, teacher_model],
            base_model_idx=0,
            projector_list=projector_list,
            include_response=model_config.get("include_response", False),
            multi_source_fusion_mode=model_config.get("multi_source_fusion_mode", "sequential")
        ).to(device).eval()
        
        
        # mapping stretegy
        if model_config["mapping"] == "last_aligned":
            source_target_mapping = last_aligned_sources(slm_num_layers, llm_num_layers, K)
        elif model_config["mapping"] == "k_nearest":
            source_target_mapping = k_nearest_sources(slm_num_layers, llm_num_layers, K)
        else:
            raise ValueError(f"Invalid mapping strategy: {model_config['mapping']}")
        print(f"Using {model_config['mapping']} mapping strategy (target: [sources])")

        # set projector
        for target_layer_idx, src_list in source_target_mapping.items():
            for source_layer_idx in src_list:
                rosetta_model.set_projector_config(
                    source_model_idx=1,  # Teacher model
                    source_model_layer_idx=source_layer_idx,
                    target_model_idx=0,  # Base model
                    target_model_layer_idx=target_layer_idx,
                    projector_idx=target_layer_idx,  # share projector per target layer
                )

        # Optional aligner construction (used by collator)
        aligner = None
        if model_config.get("is_do_alignment", False):
            # Build tokenizers for both models
            strategy = model_config.get("alignment_strategy", "first")
            aligner = TokenAligner(slm_tokenizer=slm_tokenizer, llm_tokenizer=llm_tokenizer, strategy=AlignmentStrategy(strategy))
            
        return rosetta_model, slm_tokenizer, aligner, llm_tokenizer

        
def train_step(model: nn.Module, batch: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int, device: str, training_mode: str):
    """Single training step for both baseline and Rosetta models"""
    
    if training_mode == "baseline":
        # Baseline model training
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass for baseline model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
    else:  # rosetta mode
        # Rosetta model training
        if isinstance(batch["input_ids"], list):
            input_ids = [sample_ids.to(device) for sample_ids in batch["input_ids"]]
            attention_mask = [sample_attention_mask.to(device) for sample_attention_mask in batch["attention_mask"]]
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

        position_ids = batch["position_ids"].to(device)
        labels = batch["labels"].to(device)
        kv_cache_index = [x.to(device) for x in batch["kv_cache_index"]]
        
        # Forward pass for Rosetta model
        outputs = model.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=True
        )
        
        loss = outputs.loss

        # Additional loss terms for Rosetta model
        # model_to_use = model.module if hasattr(model, "module") else model
        # for proj in model_to_use.projector_list:
        #     if hasattr(proj, 'gate_logit') and hasattr(proj, 'gate_temperature'):
        #         gate_logit = torch.mean(proj.gate_logit)
        #         gate = torch.sigmoid(gate_logit / proj.gate_temperature)
        #         loss += 0.0025 * gate
                
    return loss


def evaluate_model(model: nn.Module, eval_loader: DataLoader, tokenizer: AutoTokenizer, max_length: int, device: str, training_mode: str) -> float:
    """Evaluate the model and return average loss"""
    model.eval()
    eval_loss_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for eval_batch in eval_loader:
            eval_loss = train_step(model, eval_batch, tokenizer, max_length, device, training_mode)
            eval_loss_total += eval_loss.item()
            num_batches += 1
    
    avg_eval_loss = eval_loss_total / num_batches if num_batches > 0 else 0.0
    model.train()  # Set back to train mode
    return avg_eval_loss


def main():
    """
    Train a model (Rosetta or baseline) using hyper-parameters defined in a JSON
    or YAML configuration file. The mode is automatically detected from the config:
    - If 'baseline_model' is provided: baseline training
    - If 'base_model' and 'teacher_model' are provided: Rosetta training
    Training progress is tracked with Weights & Biases and the original config
    is copied alongside checkpoints for full reproducibility.
    """

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train RosettaModel from a JSON or YAML config")
    parser.add_argument("--config", type=str, default="recipe/all_in_one.yaml", help="Path to JSON or YAML config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs and checkpoints")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only (no training)")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)

    # Extract configuration sections
    model_config = cfg["model"]
    training_config = cfg["training"]
    output_config = cfg["output"]
    data_config = cfg["data"]

    # Set seed for reproducibility and enable stricter determinism
    set_seed(seed = training_config["seed"])
    enable_full_determinism()

    # Create datetime subfolder under output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # timestamped_output_dir = os.path.join(output_config["output_dir"], "3")
    timestamped_output_dir = output_config["output_dir"]
    # timestamped_output_dir = args.output_dir
    
    # Ensure output directory exists and copy config for reproducibility
    os.makedirs(timestamped_output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(timestamped_output_dir, "config.json"))

    # ------------------------------------------------------------------
    # Distributed training setup
    # ------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    local_rank = args.local_rank

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        local_rank = 0
        device = training_config.get("device", "cuda")

    is_main_process = rank == 0

    # ------------------------------------------------------------------
    # Weights & Biases initialisation
    # ------------------------------------------------------------------
    run_name = f"{output_config['wandb_config']['run_name']}_{timestamp}"
    if is_main_process:
        wandb.init(
            project=output_config["wandb_config"]["project"],
            name=run_name,
            config=cfg,
            mode=output_config["wandb_config"]["mode"],
            entity=output_config["wandb_config"]["entity"]
        )
    
    print(f"Outputs will be saved to: {timestamped_output_dir}")

    # ------------------------------------------------------------------
    # Detect training mode and setup models
    # ------------------------------------------------------------------
    training_mode = detect_training_mode(model_config)
    if is_main_process:
        print(f"Training mode: {training_mode}")
        print("Setting up models…")
    
    model, main_tokenizer, aligner, llm_tokenizer = setup_models(model_config, training_mode, device, torch.bfloat16)
    model = model.to(device)

    # Apply freezing/training configuration based on mode
    if training_mode == "baseline":
        # Check for LoRA or partial training configuration
        lora_config = training_config.get("lora", None)
        partial_config = training_config.get("partial_training", None)
        
        if lora_config is not None:
            if is_main_process:
                print("Setting up LoRA training...")
            model = setup_lora_model(model, lora_config)
            if is_main_process:
                print("LoRA setup completed")
        elif partial_config is not None:
            if is_main_process:
                print("Setting up partial parameter training...")
            model = setup_partial_training(model, partial_config)
            if is_main_process:
                print("Partial training setup completed")
        else:
            # Apply freezing based on configuration
            freeze_config = training_config.get("freeze", [])
            if is_main_process:
                print(f"Applying freeze configuration: {freeze_config}")
            
            if "baseline" in freeze_config or "base" in freeze_config:
                freeze_model(model)
            else:
                unfreeze_model(model)
    else:  # rosetta mode
        freeze_config = training_config["freeze"]  # including ["base", "teacher"]
        
        if is_main_process:
            print(f"Applying freeze configuration: {freeze_config}")
        
        if "base" in freeze_config:
            freeze_model(model.model_list[0])  # Base model
        else:
            unfreeze_model(model.model_list[0])
        
        if "teacher" in freeze_config:
            freeze_model(model.model_list[1])  # Teacher model
        else:
            unfreeze_model(model.model_list[1])
        
        if "projector" in freeze_config:
            # Freeze projectors
            for projector in model.projector_list:
                freeze_model(projector)
        else:
            unfreeze_projectors(model)

    # Wrap with DDP if needed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.4f}%")

    # ------------------------------------------------------------------
    # Dataset & dataloaders
    # ------------------------------------------------------------------
    print("Loading dataset…")
    # Create dataset using the auto-registration system
    instruct_ds = create_dataset(
        dataset_type=data_config["type"],
        **data_config["kwargs"]
    )
    
    # Create dataset based on training mode
    if training_mode == "baseline":
        full_dataset = BaselineChatDataset(
            instruct_ds, 
            main_tokenizer, 
            max_length=training_config.get("max_length", 2048)
        )
    elif training_mode == "rosetta":  # rosetta mode
        if model_config.get("is_do_alignment", False) and aligner is not None:
            full_dataset = AlignedChatDataset(instruct_ds, aligner)
        else:
            full_dataset = ChatDataset(instruct_ds, main_tokenizer)
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    train_size = int(data_config["train_ratio"] * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    per_device_batch_size = training_config["per_device_train_batch_size"]
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 1)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, seed=training_config["seed"]
        )
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, seed=training_config["seed"]
        )
    else:
        train_sampler = None
        eval_sampler = None

    # Create collator based on training mode
    if training_mode == "baseline":
        collator = BaselineDataCollator(
            tokenizer=main_tokenizer,
            pad_to_multiple_of=training_config.get("pad_to_multiple_of", None)
        )
    elif training_mode == "rosetta":  # rosetta mode
        collator = RosettaDataCollator(
            slm_tokenizer=main_tokenizer,
            llm_tokenizer=llm_tokenizer,
            pad_to_multiple_of=training_config.get("pad_to_multiple_of", None),
            max_length=training_config.get("max_length", 2048),
            aligner=aligner,
            do_alignment=model_config.get("is_do_alignment", False)
        )
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    # Ensure per-worker seeding if num_workers > 0
    def _worker_init_fn(worker_id):
        worker_seed = training_config["seed"] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        worker_init_fn=_worker_init_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=collator,
        worker_init_fn=_worker_init_fn,
    )

    # ------------------------------------------------------------------
    # Evaluation-only short-circuit
    # ------------------------------------------------------------------
    if args.eval_only:
        if distributed:
            local_eval_loss = evaluate_model(model, eval_loader, main_tokenizer, training_config["max_length"], device, training_mode)
            loss_tensor = torch.tensor([local_eval_loss], device=device, dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_eval_loss = loss_tensor.item()
            if is_main_process:
                print(f"Evaluation (eval_only) loss: {avg_eval_loss:.4f}")
                wandb.log({
                    "eval/loss": avg_eval_loss,
                    "mode": "eval_only",
                }, step=0)
        else:
            eval_loss = evaluate_model(model, eval_loader, main_tokenizer, training_config["max_length"], device, training_mode)
            print(f"Evaluation (eval_only) loss: {eval_loss:.4f}")
            if is_main_process:
                wandb.log({
                    "eval/loss": eval_loss,
                    "mode": "eval_only",
                }, step=0)

        if is_main_process:
            print("Evaluation-only run completed!")
            wandb.finish()
        if distributed:
            dist.destroy_process_group()
        return

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = updates_per_epoch * training_config["num_epochs"]

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    lr = training_config["learning_rate"]

    if training_mode == "baseline":
        # Simple optimizer for baseline mode
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=lr, 
            weight_decay=training_config["weight_decay"]
        )
    else:  # rosetta mode
        # Separate parameter groups for Rosetta mode
        gate_params = []
        weight_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "gate" in name:
                    gate_params.append(param)
                elif "key_weight" in name or "value_weight" in name:
                    weight_params.append(param)
                else:
                    other_params.append(param)

        optimizer = AdamW([
            {"params": gate_params, "lr": lr},
            {"params": weight_params, "lr": lr},
            {"params": other_params, "lr": lr}
            ], weight_decay=training_config["weight_decay"])

    scheduler = get_scheduler(
        training_config["scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(training_config["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("Starting training…")
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(training_config["num_epochs"]):
        if distributed and train_sampler is not None:
            # Ensure different shuffles across epochs in distributed setup
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(total=updates_per_epoch, desc=f"Epoch {epoch + 1}/{training_config['num_epochs']}", disable=not is_main_process)

        macro_step_in_epoch = 0
        accum_true_loss = 0.0
        micro_in_window = 0

        for batch_idx, batch in enumerate(train_loader):
            # Forward/backward with gradient accumulation and DDP no_sync for micro-steps
            is_accum_step = ((batch_idx + 1) % grad_accum_steps) != 0
            sync_ctx = model.no_sync() if distributed and hasattr(model, "no_sync") and is_accum_step else contextlib.nullcontext()

            with sync_ctx:
                loss = train_step(model, batch, main_tokenizer, training_config["max_length"], device, training_mode)
                true_loss_value = loss.detach().item()
                scaled_loss = loss / grad_accum_steps  # Gradient accumulation
                scaled_loss.backward()

            # accumulate true (unscaled) loss for averaging/printing
            epoch_loss += true_loss_value
            accum_true_loss += true_loss_value
            micro_in_window += 1

            # Optimizer step on boundaries or at last batch of the epoch
            did_step = (not is_accum_step) or (batch_idx + 1 == len(train_loader))
            grad_norm_value = None
            if did_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=training_config["max_grad_norm"]
                )
                grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                macro_step_in_epoch += 1

                # Update temperatures for Rosetta models
                if training_mode == "rosetta":
                    model_to_use = model.module if hasattr(model, "module") else model
                    for proj in model_to_use.projector_list:
                        if hasattr(proj, 'update_temperature') and callable(proj.update_temperature):
                            proj.update_temperature(global_step)

            # Progress bar and logging
            if is_main_process and did_step:
                # Calculate fractional epoch based on macro steps
                fractional_epoch = epoch + (macro_step_in_epoch / updates_per_epoch)

                avg_window_loss = accum_true_loss / max(1, micro_in_window)
                postfix = {
                    "loss": f"{avg_window_loss:.4f}",
                    "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                progress_bar.set_postfix(postfix)
                progress_bar.update(1)

                wandb.log({
                    "train/loss": avg_window_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm_value,
                    "train/epoch": fractional_epoch,
                }, step=global_step)

                # reset window accumulators
                accum_true_loss = 0.0
                micro_in_window = 0

            # Evaluation and checkpointing only on real optimizer steps
            if did_step:
                # Calculate fractional epoch based on macro steps
                fractional_epoch = epoch + (macro_step_in_epoch / updates_per_epoch)
                # Evaluation at regular intervals under DDP using broadcasted decision
                want_eval = (global_step % output_config["eval_steps"] == 0)
                want_eval = broadcast_decision_from_rank0(want_eval, distributed, device, rank)
                if want_eval:
                    if distributed:
                        # All ranks evaluate their shard and average
                        local_eval_loss = evaluate_model(model, eval_loader, main_tokenizer, training_config["max_length"], device, training_mode)
                        loss_tensor = torch.tensor([local_eval_loss], device=device, dtype=torch.float32)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                        avg_eval_loss = loss_tensor.item()
                        if is_main_process:
                            print(f"\nEvaluation (mid-epoch) at step {global_step}: {avg_eval_loss:.4f}")
                            wandb.log({
                                "eval/loss": avg_eval_loss,
                                "eval/step": global_step,
                                "eval/epoch": fractional_epoch
                            }, step=global_step)
                    else:
                        eval_loss = evaluate_model(model, eval_loader, main_tokenizer, training_config["max_length"], device, training_mode)
                        print(f"\nEvaluation loss at step {global_step}: {eval_loss:.4f}")
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/step": global_step,
                            "eval/epoch": fractional_epoch
                        }, step=global_step)

                # Checkpointing under DDP using broadcasted decision
                want_save = (global_step % output_config["save_steps"] == 0)
                want_save = broadcast_decision_from_rank0(want_save, distributed, device, rank)
                if want_save:
                    if is_main_process:
                        checkpoint_dir = os.path.join(timestamped_output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        # Unwrap DDP to access underlying model
                        base_model_ref = model.module if isinstance(model, DistributedDataParallel) else model

                        if training_mode == "baseline":
                            # Save baseline model
                            if hasattr(base_model_ref, 'save_pretrained'):
                                # LoRA model - save LoRA weights
                                base_model_ref.save_pretrained(checkpoint_dir)
                                if hasattr(base_model_ref, 'config'):
                                    base_model_ref.config.save_pretrained(checkpoint_dir)
                            else:
                                # Regular model - save full state dict
                                torch.save(base_model_ref.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
                            main_tokenizer.save_pretrained(checkpoint_dir)
                        else:  # rosetta mode
                            # Save Rosetta components
                            for i, proj in enumerate(base_model_ref.projector_list):
                                # We save both the trainable weights and the constructor config
                                torch.save(proj.state_dict(), os.path.join(checkpoint_dir, f"projector_{i}.pt"))
                                save_projector(proj, os.path.join(checkpoint_dir, f"projector_{i}.json"))
                            base_model_ref.save_projector_config(os.path.join(checkpoint_dir, "projector_config.json"))

                        torch.save({
                            "step": global_step,
                            "epoch": epoch,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": true_loss_value,  # true loss for this batch window
                        }, os.path.join(checkpoint_dir, "training_state.pt"))
                        print(f"\nCheckpoint saved at step {global_step}")

        avg_epoch_loss = epoch_loss / len(train_loader)

        # ------------------------------------------------------------------
        # Evaluation phase
        # ------------------------------------------------------------------
        if distributed:
            # Run eval on all ranks and average for deterministic sync
            local_eval_loss = evaluate_model(model, eval_loader, main_tokenizer, training_config["max_length"], device, training_mode)
            loss_tensor = torch.tensor([local_eval_loss], device=device, dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_eval_loss = loss_tensor.item()
            if is_main_process:
                print(f"Epoch {epoch + 1} completed. Train loss: {avg_epoch_loss:.4f} | Eval loss: {avg_eval_loss:.4f}")
                wandb.log({
                    "eval/epoch_loss": avg_eval_loss,
                    "epoch": epoch + 1,
                    "train/epoch_avg_loss": avg_epoch_loss
                }, step=global_step)
        else:
            print(f"Running end-of-epoch evaluation for epoch {epoch + 1}...")
            avg_eval_loss = evaluate_model(model, eval_loader, main_tokenizer, training_config["max_length"], device, training_mode)
            print(f"Epoch {epoch + 1} completed. Train loss: {avg_epoch_loss:.4f} | Eval loss: {avg_eval_loss:.4f}")
            wandb.log({
                "eval/epoch_loss": avg_eval_loss,
                "epoch": epoch + 1,
                "train/epoch_avg_loss": avg_epoch_loss
            }, step=global_step)

    # ------------------------------------------------------------------
    # Save final artefacts
    # ------------------------------------------------------------------
    if is_main_process:
        final_dir = os.path.join(timestamped_output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        base_model_ref = model.module if isinstance(model, DistributedDataParallel) else model

        if training_mode == "baseline":
            # Save final baseline model
            if hasattr(base_model_ref, 'save_pretrained'):
                # LoRA model - save LoRA weights
                base_model_ref.save_pretrained(final_dir)
                if hasattr(base_model_ref, 'config'):
                    base_model_ref.config.save_pretrained(final_dir)
            else:
                # Regular model - save full state dict
                torch.save(base_model_ref.state_dict(), os.path.join(final_dir, "model.pt"))
            main_tokenizer.save_pretrained(final_dir)
        else:  # rosetta mode
            # Save final Rosetta components
            for i, proj in enumerate(base_model_ref.projector_list):
                torch.save(proj.state_dict(), os.path.join(final_dir, f"projector_{i}.pt"))
                save_projector(proj, os.path.join(final_dir, f"projector_{i}.json"))
            base_model_ref.save_projector_config(os.path.join(final_dir, "projector_config.json"))

    if is_main_process:
        print("Training completed!")
        wandb.finish()

    # Clean up distributed training
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, running...")
    main()
