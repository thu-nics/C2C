"""
Use SFT trainer to train baseline model (plain HF model without Rosetta)
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
import torch.distributed as dist  # Added for Distributed Data Parallel support
from torch.nn.parallel import DistributedDataParallel  # For type checking
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import math
import contextlib

# PEFT imports for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT library not available. LoRA training will be disabled.")
    print("Install with: pip install peft")

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from rosetta.train.dataset_adapters import create_dataset


class BaselineDataCollator:
    """Custom data collator for baseline model training"""
    
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad sequences
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            # Pad input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_length - len(ids))
            batch_input_ids.append(padded_ids)
            
            # Pad labels (use -100 for padding)
            padded_labels = lbls + [-100] * (max_length - len(lbls))
            batch_labels.append(padded_labels)
            
            # Create attention mask
            attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


class BaselineChatDataset(Dataset):
    """Simple dataset for baseline model training without Rosetta-specific features"""
    
    def __init__(self, chat_dataset, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.chat_dataset = chat_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.chat_dataset)
    
    def __getitem__(self, idx):
        messages = self.chat_dataset[idx]
        
        # Get instruction (first message)
        instruction = self.tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Get full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize instruction and full text
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # Truncate if necessary
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # Create labels (-100 for instruction tokens, actual tokens for response)
        labels = [-100] * len(instruction_tokens) + full_tokens[len(instruction_tokens):]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        return {
            "input_ids": full_tokens,
            "labels": labels,
        }

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


def setup_models(model_config: Dict[str, Any], device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Setup baseline model"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["baseline_model"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_config["baseline_model"],
        torch_dtype=dtype
    )

    return baseline_model, tokenizer


def train_step(model: nn.Module, batch: Dict[str, torch.Tensor], tokenizer: AutoTokenizer, max_length: int, device: str):
    """Single training step"""
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    return loss


def evaluate_model(model: nn.Module, eval_loader: DataLoader, tokenizer: AutoTokenizer, max_length: int, device: str) -> float:
    """Evaluate the model and return average loss"""
    model.eval()
    eval_loss_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for eval_batch in eval_loader:
            eval_loss = train_step(model, eval_batch, tokenizer, max_length, device)
            eval_loss_total += eval_loss.item()
            num_batches += 1
    
    avg_eval_loss = eval_loss_total / num_batches if num_batches > 0 else 0.0
    model.train()  # Set back to train mode
    return avg_eval_loss


def main():
    """
    Train a baseline model using hyper-parameters defined in a JSON configuration
    file. The CLI is only used to specify the path to the config; all other
    settings live in the JSON. Training progress is tracked with Weights &
    Biases and the original config is copied alongside checkpoints for full
    reproducibility.
    """

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train baseline model from a JSON config")
    parser.add_argument("--config", type=str, default="recipe/default_config.json", help="Path to JSON config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs and checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    # Extract configuration sections
    model_config = cfg["model"]
    training_config = cfg["training"]
    output_config = cfg["output"]
    data_config = cfg["data"]

    # Set seed for reproducibility and enable stricter determinism
    set_seed(seed = training_config.get("seed", 42))
    enable_full_determinism()

    # Create datetime subfolder under output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = output_config["output_dir"]
    
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
    run_name = f"{output_config.get('wandb_config', {}).get('run_name', 'baseline_run')}_{timestamp}"
    if is_main_process:
        wandb_config = output_config.get("wandb_config", {})
        wandb.init(
            project=wandb_config.get("project", "baseline_training"),
            name=run_name,
            config=cfg,
            mode=wandb_config.get("mode", "offline")
        )
    
    print(f"Outputs will be saved to: {timestamped_output_dir}")

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    if is_main_process:
        print("Setting up baseline model…")
    baseline_model, tokenizer = setup_models(model_config, device, torch.bfloat16)
    baseline_model = baseline_model.to(device)

    # Check for LoRA or partial training configuration
    lora_config = training_config.get("lora", None)
    partial_config = training_config.get("partial_training", None)
    
    if lora_config is not None:
        if is_main_process:
            print("Setting up LoRA training...")
        baseline_model = setup_lora_model(baseline_model, lora_config)
        if is_main_process:
            print("LoRA setup completed")
    elif partial_config is not None:
        if is_main_process:
            print("Setting up partial parameter training...")
        baseline_model = setup_partial_training(baseline_model, partial_config)
        if is_main_process:
            print("Partial training setup completed")
    else:
        # Apply freezing based on configuration (original logic)
        freeze_config = training_config.get("freeze", [])
        
        if is_main_process:
            print(f"Applying freeze configuration: {freeze_config}")
        
        if "base" in freeze_config or "baseline" in freeze_config:
            freeze_model(baseline_model)
        else:
            unfreeze_model(baseline_model)

    # Wrap with DDP if needed
    if distributed:
        baseline_model = torch.nn.parallel.DistributedDataParallel(
            baseline_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    total_params = sum(p.numel() for p in baseline_model.parameters())
    trainable_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
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
    full_dataset = BaselineChatDataset(
        instruct_ds, 
        tokenizer, 
        max_length=training_config.get("max_length", 2048)
    )

    train_size = int(data_config["train_ratio"] * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    per_device_batch_size = training_config["per_device_train_batch_size"]
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 1)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, seed=training_config.get("seed", 42)
        )
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, seed=training_config.get("seed", 42)
        )
    else:
        train_sampler = None
        eval_sampler = None

    collator = BaselineDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=training_config.get("pad_to_multiple_of", None)
    )

    # Ensure per-worker seeding if num_workers > 0
    def _worker_init_fn(worker_id):
        worker_seed = training_config.get("seed", 42) + worker_id
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

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        [p for p in baseline_model.parameters() if p.requires_grad], 
        lr=training_config["learning_rate"], 
        weight_decay=training_config["weight_decay"]
    )
    
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
        baseline_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(total=updates_per_epoch, desc=f"Epoch {epoch + 1}/{training_config['num_epochs']}", disable=not is_main_process)

        macro_step_in_epoch = 0
        accum_true_loss = 0.0
        micro_in_window = 0

        for batch_idx, batch in enumerate(train_loader):
            # Forward/backward with gradient accumulation and DDP no_sync for micro-steps
            is_accum_step = ((batch_idx + 1) % grad_accum_steps) != 0
            sync_ctx = baseline_model.no_sync() if distributed and hasattr(baseline_model, "no_sync") and is_accum_step else contextlib.nullcontext()

            with sync_ctx:
                loss = train_step(baseline_model, batch, tokenizer, training_config["max_length"], device)
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
                    [p for p in baseline_model.parameters() if p.requires_grad],
                    max_norm=training_config["max_grad_norm"]
                )
                grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                macro_step_in_epoch += 1

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
                        local_eval_loss = evaluate_model(baseline_model, eval_loader, tokenizer, training_config["max_length"], device)
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
                        eval_loss = evaluate_model(baseline_model, eval_loader, tokenizer, training_config["max_length"], device)
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
                        base_model_ref = baseline_model.module if isinstance(baseline_model, DistributedDataParallel) else baseline_model

                        # Save model based on type
                        if hasattr(base_model_ref, 'save_pretrained'):
                            # LoRA model - save LoRA weights
                            base_model_ref.save_pretrained(checkpoint_dir)
                            if hasattr(base_model_ref, 'config'):
                                base_model_ref.config.save_pretrained(checkpoint_dir)
                        else:
                            # Regular model - save full state dict
                            torch.save(base_model_ref.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

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
            local_eval_loss = evaluate_model(baseline_model, eval_loader, tokenizer, training_config["max_length"], device)
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
            avg_eval_loss = evaluate_model(baseline_model, eval_loader, tokenizer, training_config["max_length"], device)
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

        base_model_ref = baseline_model.module if isinstance(baseline_model, DistributedDataParallel) else baseline_model

        # Save final model based on type
        if hasattr(base_model_ref, 'save_pretrained'):
            # LoRA model - save LoRA weights
            base_model_ref.save_pretrained(final_dir)
            if hasattr(base_model_ref, 'config'):
                base_model_ref.config.save_pretrained(final_dir)
        else:
            # Regular model - save full state dict
            torch.save(base_model_ref.state_dict(), os.path.join(final_dir, "model.pt"))

    if is_main_process:
        print("Training completed!")
        wandb.finish()

    # Clean up distributed training
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":

    # debug mode
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Comment out for distributed training
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, running...")
    
    main()

