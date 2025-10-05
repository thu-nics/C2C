"""
Use SFT trainer to train rosetta model with hidden states loss
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
import argparse
import shutil
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import math
import contextlib

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import create_projector, save_projector, TrivialProjector
from rosetta.model.aggregator import save_aggregator, get_aggregator_class
from rosetta.train.dataset_adapters import ChatDataset, RosettaDataCollator, create_dataset
from rosetta.train.model_utils import k_nearest_sources, last_aligned_sources

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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

def enable_full_determinism():
    """Enable stricter determinism settings for reproducibility."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
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
        if isinstance(projector, TrivialProjector):
            for param in projector.parameters():
                param.requires_grad = False
        else:
            for param in projector.parameters():
                param.requires_grad = True

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
        
    layers.append(nn.Linear(source_dim, hidden_dim, dtype=dtype))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim, dtype=dtype))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout))
        
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, dtype=dtype))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
    if num_layers > 1:
        layers.append(nn.Linear(hidden_dim, target_dim, dtype=dtype))
    else:
        layers = [nn.Linear(source_dim, target_dim, dtype=dtype)]
        
    return nn.Sequential(*layers)
    
def setup_models(model_config: Dict[str, Any], device: str = "cuda", dtype: torch.dtype = torch.bfloat16, chosen_target_layer_idx: int = None):
    """Setup base and teacher models with projectors"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=dtype,
        output_hidden_states=True  # Enable hidden states output
    )
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_config["teacher_model"],
        torch_dtype=dtype,
        output_hidden_states=True  # Enable hidden states output
    )
    
    base_dim = int(base_model.model.layers[0].self_attn.k_proj.out_features)
    teacher_dim = int(teacher_model.model.layers[0].self_attn.k_proj.out_features)
    slm_num_layers = base_model.config.num_hidden_layers
    llm_num_layers = teacher_model.config.num_hidden_layers
    
    projector_config = model_config["projector"]
    projector_params = projector_config["params"].copy()
    projector_params["dtype"] = dtype
    projector_list = []
    num_projectors = slm_num_layers

    if chosen_target_layer_idx is None:
        chosen_target_layer_idx = slm_num_layers - 1

    for proj_idx in range(num_projectors):
        if proj_idx == chosen_target_layer_idx:
            print(f"-----------------create replace projector {proj_idx}------------------")
            projector = create_projector(
                projector_config["type"],
                source_dim=teacher_dim,
                target_dim=base_dim,
                **projector_params
            )
        # else:
        #     projector = create_projector(
        #         "TrivialProjector",
        #         source_dim=teacher_dim,
        #         target_dim=base_dim,
        #         **projector_params
        #     )
            projector_list.append(projector.to(device))
    
    K = 1
    aggregator_list = []

    rosetta_model = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,
        projector_list=projector_list,
        aggregator_list=aggregator_list,
    ).to(device).eval()
    
    if model_config["mapping"] == "last_aligned":
        source_target_mapping = last_aligned_sources(slm_num_layers, llm_num_layers, K)
    elif model_config["mapping"] == "k_nearest":
        source_target_mapping = k_nearest_sources(slm_num_layers, llm_num_layers, K)
    else:
        raise ValueError(f"Invalid mapping strategy: {model_config['mapping']}")
    print(f"Using {model_config['mapping']} mapping strategy (target: [sources])")

    for target_layer_idx, src_list in source_target_mapping.items():
        # if target_layer_idx == chosen_target_layer_idx:
        if len(aggregator_list) > 0:
            rosetta_model.set_aggregator_idx(
                source_model_idx=1,
                target_model_idx=0,
                target_model_layer_idx=target_layer_idx,
                aggregator_idx=0,
            )
        for source_layer_idx in src_list:
            rosetta_model.set_projector_config(
                source_model_idx=1,
                source_model_layer_idx=source_layer_idx,
                target_model_idx=0,
                target_model_layer_idx=target_layer_idx,
                projector_idx=0,
            )
    return rosetta_model, tokenizer, chosen_target_layer_idx

def compute_single_layer_hidden_states_loss(rosetta_hidden_state: torch.Tensor, 
                                           base_hidden_state: torch.Tensor,
                                           attention_mask: torch.Tensor,
                                           loss_type: str = "mse") -> torch.Tensor:
    """
    Compute loss between Rosetta and base model hidden states for a single layer
    
    Args:
        rosetta_hidden_state: Hidden state from Rosetta model for the chosen layer
        base_hidden_state: Hidden state from base model for the chosen layer
        attention_mask: Attention mask to ignore padding tokens
        loss_type: Type of loss to use ("mse", "cosine", "l1")
    
    Returns:
        Computed loss tensor
    """
    # Expand attention mask for broadcasting
    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
    
    if rosetta_hidden_state.size() != base_hidden_state.size():
        raise ValueError(f"Hidden states size mismatch: "
                       f"Rosetta {rosetta_hidden_state.size()} vs Base {base_hidden_state.size()}")
    
    if loss_type == "mse":
        layer_loss = torch.nn.functional.mse_loss(rosetta_hidden_state, base_hidden_state, reduction='none')
    elif loss_type == "l1":
        layer_loss = torch.nn.functional.l1_loss(rosetta_hidden_state, base_hidden_state, reduction='none')
    elif loss_type == "cosine":
        # Cosine similarity loss (1 - cosine similarity)
        rosetta_norm = torch.nn.functional.normalize(rosetta_hidden_state, p=2, dim=-1)
        base_norm = torch.nn.functional.normalize(base_hidden_state, p=2, dim=-1)
        cosine_sim = (rosetta_norm * base_norm).sum(dim=-1)
        layer_loss = 1.0 - cosine_sim
        layer_loss = layer_loss.unsqueeze(-1)  # Add dimension for consistency
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Mask out padding tokens and average over valid tokens
    masked_loss = layer_loss * attention_mask_expanded
    num_valid_tokens = attention_mask_expanded.sum()
    
    if num_valid_tokens > 0:
        loss_value = masked_loss.sum() / num_valid_tokens
    else:
        loss_value = torch.tensor(0.0, device=rosetta_hidden_state.device)
    
    return loss_value

def train_step(model: RosettaModel, batch: List[Tuple[str]], tokenizer: AutoTokenizer, 
              max_length: int, device: str, loss_config: Dict[str, Any], chosen_target_layer_idx: int) -> torch.Tensor:
    """Single training step with hidden states loss for chosen layer only"""
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    labels = batch["labels"].to(device)
    kv_cache_index = [x.to(device) for x in batch["kv_cache_index"]]
    
    # Forward pass through base model to get reference hidden states
    
    # Forward pass through Rosetta model
    rosetta_outputs, loss = model.module.oracle_forward(
        kv_cache_index=kv_cache_index,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        output_hidden_states=True,  # Return hidden states
        use_cache=True
    )
    
    return loss.mean()

def evaluate_model(model: RosettaModel, eval_loader: DataLoader, tokenizer: AutoTokenizer, 
                  max_length: int, device: str, loss_config: Dict[str, Any], chosen_target_layer_idx: int) -> float:
    """Evaluate the model and return average loss"""
    model.eval()
    eval_loss_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for eval_batch in eval_loader:
            eval_loss = train_step(model, eval_batch, tokenizer, max_length, device, loss_config, chosen_target_layer_idx)
            eval_loss_total += eval_loss.item()
            num_batches += 1
    
    avg_eval_loss = eval_loss_total / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_eval_loss

def main():
    parser = argparse.ArgumentParser(description="Train RosettaModel from a JSON config")
    parser.add_argument("--config", type=str, default="recipe/default_config.json", help="Path to JSON config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs and checkpoints")
    parser.add_argument("--chosen_target_layer_idx", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    model_config = cfg["model"]
    training_config = cfg["training"]
    output_config = cfg["output"]
    data_config = cfg["data"]
    
    # Extract loss configuration
    loss_config = training_config.get("loss", {
        "type": "mse",
        "use_original_loss": False,
        "hidden_states_weight": 1.0,
        "original_loss_weight": 0.0,
        "gate_regularization": 0.0025
    })

    set_seed(seed=training_config["seed"])
    enable_full_determinism()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = os.path.join(output_config["output_dir"], timestamp)
    os.makedirs(timestamped_output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(timestamped_output_dir, "config.json"))

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

    run_name = f"{output_config['wandb_config']['run_name']}_{timestamp}"
    if is_main_process:
        wandb.init(
            project=output_config["wandb_config"]["project"],
            name=run_name,
            config=cfg,
            mode=output_config["wandb_config"]["mode"]
        )
    
    print(f"Outputs will be saved to: {timestamped_output_dir}")

    if is_main_process:
        print("Setting up models…")
    rosetta_model, tokenizer, chosen_target_layer_idx = setup_models(model_config, device, torch.bfloat16, chosen_target_layer_idx = args.chosen_target_layer_idx)
    
    if is_main_process:
        print(f"Using target layer index: {chosen_target_layer_idx} for loss calculation")

    freeze_config = training_config["freeze"]
    
    if is_main_process:
        print(f"Applying freeze configuration: {freeze_config}")
    
    if "base" in freeze_config:
        freeze_model(rosetta_model.model_list[0])
    else:
        unfreeze_model(rosetta_model.model_list[0])
    
    if "teacher" in freeze_config:
        freeze_model(rosetta_model.model_list[1])
    else:
        unfreeze_model(rosetta_model.model_list[1])
    
    if "projector" in freeze_config:
        for projector in rosetta_model.projector_list:
            freeze_model(projector)
    else:
        unfreeze_projectors(rosetta_model)

    if distributed:
        rosetta_model = torch.nn.parallel.DistributedDataParallel(
            rosetta_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    total_params = sum(p.numel() for p in rosetta_model.parameters())
    trainable_params = sum(p.numel() for p in rosetta_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.4f}%")

    print("Loading dataset…")
    instruct_ds = create_dataset(
        dataset_type=data_config["type"],
        **data_config["kwargs"]
    )
    full_dataset = ChatDataset(instruct_ds, tokenizer)

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

    collator = RosettaDataCollator(
        tokenizer,
        pad_to_multiple_of=training_config.get("pad_to_multiple_of", None),
        max_length=2048
    )

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

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = updates_per_epoch * training_config["num_epochs"]

    gate_params = []
    weight_params = []
    other_params = []

    for name, param in rosetta_model.named_parameters():
        if param.requires_grad:
            if "gate" in name:
                gate_params.append(param)
            elif "key_weight" in name or "value_weight" in name:
                weight_params.append(param)
            else:
                other_params.append(param)

    optimizer = AdamW([
        {"params": gate_params, "lr": 3e-4},
        {"params": weight_params, "lr": 3e-4},
        {"params": other_params, "lr": 3e-4}
        ], weight_decay=training_config["weight_decay"])

    scheduler = get_scheduler(
        training_config["scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(training_config["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    print("Starting training…")
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(training_config["num_epochs"]):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        rosetta_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(total=updates_per_epoch, desc=f"Epoch {epoch + 1}/{training_config['num_epochs']}", disable=not is_main_process)

        macro_step_in_epoch = 0
        accum_true_loss = 0.0
        micro_in_window = 0

        for batch_idx, batch in enumerate(train_loader):
            model_to_use = rosetta_model.module if hasattr(rosetta_model, "module") else rosetta_model
            for proj in model_to_use.projector_list:
                if hasattr(proj, 'update_temperature') and callable(proj.update_temperature):
                    proj.update_temperature(batch_idx)

            for agg in model_to_use.aggregator_list:
                if hasattr(agg, 'update_temperature') and callable(agg.update_temperature):
                    agg.update_temperature(batch_idx)

            is_accum_step = ((batch_idx + 1) % grad_accum_steps) != 0
            sync_ctx = rosetta_model.no_sync() if distributed and hasattr(rosetta_model, "no_sync") and is_accum_step else contextlib.nullcontext()

            with sync_ctx:
                loss = train_step(rosetta_model, batch, tokenizer, training_config["max_length"], device, loss_config, chosen_target_layer_idx)
                true_loss_value = loss.detach().item()
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

            epoch_loss += true_loss_value
            accum_true_loss += true_loss_value
            micro_in_window += 1

            did_step = (not is_accum_step) or (batch_idx + 1 == len(train_loader))
            grad_norm_value = None
            if did_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in rosetta_model.parameters() if p.requires_grad],
                    max_norm=training_config["max_grad_norm"]
                )
                grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                macro_step_in_epoch += 1

            if is_main_process and did_step:
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

                accum_true_loss = 0.0
                micro_in_window = 0

            if did_step:
                fractional_epoch = epoch + (macro_step_in_epoch / updates_per_epoch)
                want_eval = (global_step % output_config["eval_steps"] == 0)
                want_eval = broadcast_decision_from_rank0(want_eval, distributed, device, rank)
                if want_eval:
                    if distributed:
                        local_eval_loss = evaluate_model(rosetta_model, eval_loader, tokenizer, training_config["max_length"], device, loss_config, chosen_target_layer_idx)
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
                        eval_loss = evaluate_model(rosetta_model, eval_loader, tokenizer, training_config["max_length"], device, loss_config, chosen_target_layer_idx)
                        print(f"\nEvaluation loss at step {global_step}: {eval_loss:.4f}")
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/step": global_step,
                            "eval/epoch": fractional_epoch
                        }, step=global_step)

                want_save = (global_step % output_config["save_steps"] == 0)
                want_save = broadcast_decision_from_rank0(want_save, distributed, device, rank)
                if want_save:
                    if is_main_process:
                        checkpoint_dir = os.path.join(timestamped_output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        base_model_ref = rosetta_model.module if isinstance(rosetta_model, DistributedDataParallel) else rosetta_model

                        for i, proj in enumerate(base_model_ref.projector_list):
                            torch.save(proj.state_dict(), os.path.join(checkpoint_dir, f"projector_{i}.pt"))
                            save_projector(proj, os.path.join(checkpoint_dir, f"projector_{i}.json"))
                        for i, agg in enumerate(base_model_ref.aggregator_list):
                            torch.save(agg.state_dict(), os.path.join(checkpoint_dir, f"aggregator_{i}.pt"))
                            save_aggregator(agg, os.path.join(checkpoint_dir, f"aggregator_{i}.json"))
                        base_model_ref.save_projector_config(os.path.join(checkpoint_dir, "projector_config.json"))
                        base_model_ref.save_aggregator_config(os.path.join(checkpoint_dir, "aggregator_config.json"))

                        torch.save({
                            "step": global_step,
                            "epoch": epoch,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": true_loss_value,
                        }, os.path.join(checkpoint_dir, "training_state.pt"))
                        print(f"\nCheckpoint saved at step {global_step}")

        avg_epoch_loss = epoch_loss / len(train_loader)

        if distributed:
            local_eval_loss = evaluate_model(rosetta_model, eval_loader, tokenizer, training_config["max_length"], device, loss_config, chosen_target_layer_idx)
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
            avg_eval_loss = evaluate_model(rosetta_model, eval_loader, tokenizer, training_config["max_length"], device, loss_config, chosen_target_layer_idx)
            print(f"Epoch {epoch + 1} completed. Train loss: {avg_epoch_loss:.4f} | Eval loss: {avg_eval_loss:.4f}")
            wandb.log({
                "eval/epoch_loss": avg_eval_loss,
                "epoch": epoch + 1,
                "train/epoch_avg_loss": avg_epoch_loss
            }, step=global_step)

    if is_main_process:
        final_dir = os.path.join(timestamped_output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        base_model_ref = rosetta_model.module if isinstance(rosetta_model, DistributedDataParallel) else rosetta_model

        for i, proj in enumerate(base_model_ref.projector_list):
            torch.save(proj.state_dict(), os.path.join(final_dir, f"projector_{i}.pt"))
            save_projector(proj, os.path.join(final_dir, f"projector_{i}.json"))
        for i, agg in enumerate(base_model_ref.aggregator_list):
            torch.save(agg.state_dict(), os.path.join(final_dir, f"aggregator_{i}.pt"))
            save_aggregator(agg, os.path.join(final_dir, f"aggregator_{i}.json"))
        base_model_ref.save_projector_config(os.path.join(final_dir, "projector_config.json"))
        base_model_ref.save_aggregator_config(os.path.join(final_dir, "aggregator_config.json"))

    if is_main_process:
        print("Training completed!")
        wandb.finish()

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()