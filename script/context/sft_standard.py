"""
SFT Training on UltraChat Dataset

Fine-tunes model on stingning/ultrachat with loss only on assistant responses.

Usage:
    # Train on HuggingFace ultrachat
    accelerate launch --num_processes 8 script/train/sft_ultrachat.py --model_name Qwen/Qwen3-1.7B

    # Train on generated dataset
    accelerate launch --num_processes 8 script/train/sft_ultrachat.py \
        --model_name Qwen/Qwen3-1.7B \
        --dataset_path local/ultrachat_qwen3_8b_output/dataset
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from rosetta.context import format_conversation, load_eval_examples, run_evaluation


def tokenize_with_labels(messages, tokenizer, max_length=2048):
    """
    Tokenize conversation and create labels that mask user turns.
    Labels are -100 for tokens we don't want to compute loss on.
    """
    # Build full conversation text with chat template
    full_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False,
        enable_thinking=False
    )
    
    # Tokenize full conversation
    full_tokens = tokenizer(
        full_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
        add_special_tokens=False
    )
    input_ids = full_tokens.input_ids[0]
    
    # Create labels: -100 for user turns, actual tokens for assistant turns
    labels = torch.full_like(input_ids, -100)
    
    # Find assistant response boundaries
    # Strategy: tokenize incrementally to find where each assistant response starts/ends
    current_pos = 0
    for i, msg in enumerate(messages):
        # Tokenize up to and including this message
        partial_messages = messages[:i+1]
        partial_text = tokenizer.apply_chat_template(
            partial_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        partial_tokens = tokenizer(
            partial_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        end_pos = partial_tokens.input_ids.shape[1]
        
        # If assistant turn, unmask these tokens for loss
        if msg["role"] == "assistant":
            # Unmask from current_pos to end_pos
            labels[current_pos:min(end_pos, len(labels))] = input_ids[current_pos:min(end_pos, len(labels))]
        
        current_pos = end_pos
    
    return input_ids, labels


def collate_fn(batch, tokenizer, max_length=2048):
    """Collate batch of examples."""
    input_ids_list = []
    labels_list = []
    
    for example in batch:
        messages = format_conversation(example["data"], tokenizer)
        input_ids, labels = tokenize_with_labels(messages, tokenizer, max_length)
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    # Pad to same length
    max_len = max(ids.shape[0] for ids in input_ids_list)
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for input_ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - input_ids.shape[0]
        
        # Pad input_ids with pad_token_id
        padded_input_ids.append(
            torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id)])
        )
        # Pad labels with -100
        padded_labels.append(
            torch.cat([labels, torch.full((pad_len,), -100)])
        )
        # Attention mask
        attention_masks.append(
            torch.cat([torch.ones_like(input_ids), torch.zeros(pad_len, dtype=torch.long)])
        )
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft_ultrachat")
    parser.add_argument("--dataset_path", type=str, default=None, 
                        help="Path to local generated dataset. If None, loads from stingning/ultrachat.")
    parser.add_argument("--eval_examples", type=str, default="script/train/examples.yaml",
                        help="Path to evaluation examples YAML file")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluate every N steps")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epoch)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    args = parser.parse_args()
    
    # Initialize accelerator
    # Note: dispatch_batches will be configured after we know if streaming or not
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
    )
    set_seed(args.seed)
    
    # Initialize wandb
    if accelerator.is_main_process:
        run_name = args.wandb_run_name or f"sft-{args.model_name.split('/')[-1]}"
        accelerator.init_trackers(
            project_name="contextual",
            config=vars(args),
            init_kwargs={"wandb": {"name": run_name}},
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    if accelerator.is_main_process:
        print(f"Loading model {args.model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    
    # Load dataset (streaming mode)
    if accelerator.is_main_process:
        print("Loading dataset...")
    
    if args.dataset_path:
        # Load from local generated dataset
        if accelerator.is_main_process:
            print(f"Loading from local path: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
        dataset_size = len(dataset)
        
        if accelerator.is_main_process:
            print(f"Loaded {dataset_size} examples")
    else:
        # Load from HuggingFace Hub (streaming)
        if accelerator.is_main_process:
            print("Loading from HuggingFace Hub: stingning/ultrachat (streaming)")
        dataset = load_dataset("stingning/ultrachat", split="train", streaming=True)
        dataset_size = 1400000  # Approximate size
    
    # Create dataloader
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer, args.max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_wrapper,
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Calculate total steps
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        # Calculate from actual dataset size
        total_steps = dataset_size // (args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes) * args.num_epochs
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Prepare with accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # Load evaluation examples
    eval_examples = []
    if os.path.exists(args.eval_examples):
        eval_examples = load_eval_examples(args.eval_examples)
        if accelerator.is_main_process:
            print(f"Loaded {len(eval_examples)} evaluation examples from {args.eval_examples}")
    
    # Training loop
    if accelerator.is_main_process:
        print(f"Starting training on {accelerator.num_processes} GPUs...")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
        print(f"  Total steps: {total_steps}")
        print(f"  Eval steps: {args.eval_steps}")
    
    # Initial evaluation before training
    if eval_examples:
        run_evaluation(model, tokenizer, eval_examples, accelerator, global_step=0)
    
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    
    progress_bar = tqdm(
        total=total_steps if args.max_steps > 0 else None, 
        desc="Training",
        disable=not accelerator.is_main_process
    )
    
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accumulated_loss += loss.detach().item() / args.gradient_accumulation_steps
                
                # Backward pass
                accelerator.backward(loss)
                
                # Update weights (handled by accelerator.accumulate context)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # After accumulation
            if accelerator.sync_gradients:
                global_step += 1
                
                # Logging
                if global_step % args.log_steps == 0:
                    avg_loss = accumulated_loss / args.log_steps
                    lr = lr_scheduler.get_last_lr()[0]
                    
                    if accelerator.is_main_process:
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                    
                    # Log to wandb
                    accelerator.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/global_step": global_step,
                    }, step=global_step)
                    
                    accumulated_loss = 0.0
                
                progress_bar.update(1)
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = f"{args.output_dir}/checkpoint-{global_step}"
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        print(f"\nSaved checkpoint to {save_path}")
                
                # Evaluation
                if global_step % args.eval_steps == 0 and eval_examples:
                    run_evaluation(model, tokenizer, eval_examples, accelerator, global_step)
                
                # Check max steps
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    # Save final model
    if accelerator.is_main_process:
        final_path = f"{args.output_dir}/final"
        os.makedirs(final_path, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Training complete. Model saved to {final_path}")
    
    # End wandb tracking
    accelerator.end_training()


if __name__ == "__main__":
    main()
