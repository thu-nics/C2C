"""
Common utilities for contextual training and evaluation.
"""

import os
import yaml
import torch
from typing import List, Dict, Any, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Global list for accumulating eval results across steps
# We store data as a list and create a fresh Table each time to avoid immutability issues
_eval_table_data: Optional[List[List]] = None

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p) sampling.

    Args:
        logits: (batch, vocab) logits.
        top_k: Keep only top_k tokens with highest probability (0 disables).
        top_p: Keep the smallest set of tokens with cumulative probability >= top_p (1.0 disables).
    """
    if top_k is None:
        top_k = 0
    if top_p is None:
        top_p = 1.0

    if top_k <= 0 and top_p >= 1.0:
        return logits

    filtered = logits

    if top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        values, _ = torch.topk(filtered, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        filtered = torch.where(
            filtered < min_values,
            torch.full_like(filtered, -float("inf")),
            filtered,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least 1 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        mask = torch.zeros_like(filtered, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
        filtered = filtered.masked_fill(mask, -float("inf"))

    return filtered

def format_conversation(data: List[str], tokenizer=None) -> List[Dict[str, str]]:
    """
    Convert ultrachat data format to chat messages.
    
    Args:
        data: list of alternating [user, assistant, user, assistant, ...]
        tokenizer: Optional tokenizer (unused, kept for compatibility)
        
    Returns:
        List of message dicts with 'role' and 'content'
    """
    messages = []
    for i, text in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})
    return messages


def load_eval_examples(yaml_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation examples from YAML file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        List of example dicts with 'name', 'messages', 'drop_rounds'
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("examples", [])





def _print_eval_results(eval_results: List[Dict[str, Any]], global_step: int, use_drop: bool = False):
    """Print evaluation results to console."""
    print(f"\n{'='*60}")
    mode = "with drop" if use_drop else "no drop"
    print(f"📋 Evaluation at step {global_step} ({mode})")
    print(f"{'='*60}")
    
    for r in eval_results:
        print(f"\n[{r['name']}] (drop_rounds={r['drop_rounds']})")
        for turn_idx, (user, response) in enumerate(zip(r['user_messages'], r['generated_responses'])):
            print(f"  Turn {turn_idx + 1}:")
            print(f"    Q: {user[:80]}{'...' if len(user) > 80 else ''}")
            print(f"    A: {response[:150]}{'...' if len(response) > 150 else ''}")
    
    print(f"{'='*60}\n")


def _log_to_wandb(accelerator, eval_results: List[Dict[str, Any]], global_step: int):
    """Log evaluation results to wandb.
    
    Accumulates data in a list and creates a new table each time to avoid
    the 'mutating immutable table' warning.
    """
    global _eval_table_data
    
    # Initialize data list on first call
    if "_eval_table_data" not in globals() or _eval_table_data is None:
        globals()["_eval_table_data"] = []
    
    # Append new rows to the data list
    for r in eval_results:
        for turn_idx, (user, response) in enumerate(zip(r['user_messages'], r['generated_responses'])):
            _eval_table_data.append([
                global_step,
                r["name"],
                turn_idx + 1,
                user,
                response,
                str(r["drop_rounds"]),
            ])
    
    # Create a fresh table with all accumulated data
    table = wandb.Table(
        columns=["step", "name", "turn", "question", "response", "drop_rounds"],
        data=_eval_table_data,
    )
    
    # Log the table
    wandb_tracker = accelerator.get_tracker("wandb")
    if wandb_tracker:
        wandb_tracker.log({"eval/responses": table}, step=global_step)
    else:
        accelerator.log({"eval/responses": table}, step=global_step)


def reset_eval_table():
    """Reset the evaluation table data. Call at the start of a new training run."""
    global _eval_table_data
    _eval_table_data = None

