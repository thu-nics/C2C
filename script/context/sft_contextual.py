"""
Contextual SFT Training on UltraChat Dataset

Fine-tunes model with random KV dropping simulation using attention masks.
Trains the model to generate good responses even when conversation history is "dropped".

Usage:
    # Train on HuggingFace ultrachat
    accelerate launch --num_processes 8 script/train/sft_contextual.py --model_name Qwen/Qwen3-1.7B

    # Train on generated dataset
    accelerate launch --num_processes 8 script/train/sft_contextual.py \
        --model_name Qwen/Qwen3-1.7B \
        --dataset_path local/ultrachat_qwen3_8b_output/dataset
"""

import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from rosetta.context import (
    format_conversation, 
    load_eval_examples, 
    run_evaluation,
    get_round_boundaries,
    build_contextual_attention_mask,
)


def tokenize_with_labels_and_mask(messages, tokenizer, max_length=2048, drop_prob=0.3):
    """
    Tokenize conversation, create labels, and build contextual attention mask.
    
    Simulates the ContextualModel behavior where rounds are dropped immediately
    after their user turn is prefilled. Each dropped round is gone for all
    subsequent rounds.
    
    Returns:
        input_ids: Token IDs
        labels: Labels (-100 for non-assistant tokens)
        attention_mask_4d: Custom 4D attention mask for contextual training
        num_dropped: Number of rounds dropped
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
    seq_len = input_ids.shape[0]
    
    # Get round boundaries
    boundaries = get_round_boundaries(tokenizer, messages, max_length)
    
    # Create labels: -100 for user turns, actual tokens for assistant turns
    labels = torch.full_like(input_ids, -100)
    
    # Find unique rounds
    unique_rounds = sorted(set(b[3] for b in boundaries))
    num_rounds = len(unique_rounds)
    
    # Build drop_at_round config: {round_where_drop_happens: [rounds_to_drop]}
    # Each round can be dropped at the immediately following round
    # E.g., round 0 can be dropped at round 1, round 1 at round 2, etc.
    drop_at_round = {}
    total_dropped = 0
    
    for i, round_idx in enumerate(unique_rounds[:-1]):  # Can't drop the last round
        next_round = unique_rounds[i + 1]
        if random.random() < drop_prob:
            if next_round not in drop_at_round:
                drop_at_round[next_round] = []
            drop_at_round[next_round].append(round_idx)
            total_dropped += 1
    
    # Build attention mask with fine-grained drop control
    attention_mask_4d = build_contextual_attention_mask(
        seq_len=seq_len,
        round_boundaries=boundaries,
        rounds_to_drop=[],  # Not used when drop_at_round is provided
        device=input_ids.device,
        dtype=torch.float32,
        drop_at_round=drop_at_round if drop_at_round else None,
    )
    
    # Set labels for assistant tokens
    for start, end, role, round_idx in boundaries:
        if role == "assistant":
            end = min(end, len(labels))
            labels[start:end] = input_ids[start:end]
    
    return input_ids, labels, attention_mask_4d, total_dropped


def collate_fn(batch, tokenizer, max_length=2048, drop_prob=0.3):
    """Collate batch of examples with contextual attention masks."""
    input_ids_list = []
    labels_list = []
    attention_masks_4d = []
    num_dropped_list = []
    
    for example in batch:
        messages = format_conversation(example["data"], tokenizer)
        input_ids, labels, attn_mask_4d, num_dropped = tokenize_with_labels_and_mask(
            messages, tokenizer, max_length, drop_prob
        )
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks_4d.append(attn_mask_4d.squeeze(0).squeeze(0))  # (seq, seq)
        num_dropped_list.append(num_dropped)
    
    # Pad to same length
    max_len = max(ids.shape[0] for ids in input_ids_list)
    
    padded_input_ids = []
    padded_labels = []
    padded_attention_masks_4d = []
    
    for input_ids, labels, attn_mask in zip(input_ids_list, labels_list, attention_masks_4d):
        seq_len = input_ids.shape[0]
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input_ids.append(
            torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id)])
        )
        # Pad labels
        padded_labels.append(
            torch.cat([labels, torch.full((pad_len,), -100)])
        )
        
        # Pad attention mask (4D)
        # New shape: (max_len, max_len)
        # Pad with -inf for positions that shouldn't be attended
        padded_mask = torch.full((max_len, max_len), float('-inf'), dtype=attn_mask.dtype)
        padded_mask[:seq_len, :seq_len] = attn_mask
        padded_attention_masks_4d.append(padded_mask)
    
    # Stack into batch
    # attention_mask_4d shape: (batch, 1, max_len, max_len)
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask_4d": torch.stack(padded_attention_masks_4d).unsqueeze(1),
        "num_dropped": sum(num_dropped_list) / len(num_dropped_list),  # avg for logging
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft_contextual")
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
    parser.add_argument("--drop_prob", type=float, default=0.3, help="Probability of dropping each round")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
    )
    set_seed(args.seed)
    
    # Initialize wandb
    if accelerator.is_main_process:
        run_name = args.wandb_run_name or f"contextual-{args.model_name.split('/')[-1]}-drop{args.drop_prob}"
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
        if accelerator.is_main_process:
            print(f"Loading from local path: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
        dataset_size = len(dataset)
        if accelerator.is_main_process:
            print(f"Loaded {dataset_size} examples")
    else:
        if accelerator.is_main_process:
            print("Loading from HuggingFace Hub: stingning/ultrachat (streaming)")
        dataset = load_dataset("stingning/ultrachat", split="train", streaming=True)
        dataset_size = 1400000  # Approximate size
    
    # Create dataloader
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer, args.max_length, args.drop_prob)
    
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
        print(f"Starting contextual training on {accelerator.num_processes} GPUs...")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
        print(f"  Drop probability: {args.drop_prob}")
        print(f"  Total steps: {total_steps}")
        print(f"  Eval steps: {args.eval_steps}")
    
    # Initial evaluation before training (with dropping enabled)
    if eval_examples:
        run_evaluation(model, tokenizer, eval_examples, accelerator, global_step=0, use_drop=True)
    
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    accumulated_dropped = 0.0
    
    progress_bar = tqdm(
        total=total_steps if args.max_steps > 0 else None, 
        desc="Training",
        disable=not accelerator.is_main_process
    )
    
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # Get the 4D attention mask and convert to model dtype
                attention_mask_4d = batch["attention_mask_4d"].to(accelerator.unwrap_model(model).dtype)
                
                # Forward pass with custom attention mask
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=attention_mask_4d,
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accumulated_loss += loss.detach().item() / args.gradient_accumulation_steps
                accumulated_dropped += batch["num_dropped"] / args.gradient_accumulation_steps
                
                # Backward pass
                accelerator.backward(loss)
                
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
                    avg_dropped = accumulated_dropped / args.log_steps
                    lr = lr_scheduler.get_last_lr()[0]
                    
                    if accelerator.is_main_process:
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}", 
                            "lr": f"{lr:.2e}",
                            "dropped": f"{avg_dropped:.2f}"
                        })
                    
                    # Log to wandb
                    accelerator.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/global_step": global_step,
                        "train/avg_rounds_dropped": avg_dropped,
                    }, step=global_step)
                    
                    accumulated_loss = 0.0
                    accumulated_dropped = 0.0
                
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
                
                # Evaluation (with dropping enabled)
                if global_step % args.eval_steps == 0 and eval_examples:
                    run_evaluation(model, tokenizer, eval_examples, accelerator, global_step, use_drop=True)
                
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

