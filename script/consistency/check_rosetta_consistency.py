"""
Rosetta Model Consistency Check

This script computes the label consistency between Rosetta model (SLM with projectors)
and the teacher LLM model. It measures how well the Rosetta model's predictions align
with the teacher model's predictions on the response/label portion.

Usage:
    python check_rosetta_consistency.py --config rosetta_consistency_config.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.train.dataset_adapters import create_dataset
from rosetta.utils.evaluate import load_rosetta_model

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute label consistency between Rosetta model and LLM")
    parser.add_argument("--config", type=str, default="consistency_scripts/rosetta_consistency_config.json")
    args = parser.parse_args()
    
    # Load all configurations from config file
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    models_cfg = config.get("models", {})
    training_cfg = config.get("training", {})
    rosetta_cfg = config.get("rosetta", {})
    
    # Build configuration
    defaults = {
        "dataset_type": dataset_cfg.get("type"),
        "dataset_kwargs": dataset_cfg.get("kwargs", {}),
        "slm": models_cfg.get("slm"),
        "llm": models_cfg.get("llm"),
        "checkpoints_dir": rosetta_cfg.get("checkpoints_dir"),
        "include_response": rosetta_cfg.get("include_response", False),
        "is_do_alignment": rosetta_cfg.get("is_do_alignment", False),
        "alignment_strategy": rosetta_cfg.get("alignment_strategy", "first"),
        "device": training_cfg.get("device", "cuda"),
        "dtype": training_cfg.get("dtype", "bfloat16"),
        "max_length": training_cfg.get("max_length", 2048),
        "max_samples": training_cfg.get("max_samples"),
        "trust_remote_code": training_cfg.get("trust_remote_code", False),
        "print_every": training_cfg.get("print_every", 50),
        # Prefer command line arguments, use config file if not specified
        "debug": training_cfg.get("debug", False),
        "debug_samples": training_cfg.get("debug_samples", 3),
    }
    
    # Convert to Namespace
    return argparse.Namespace(**defaults)


def _to_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def build_plain_inputs(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build tokenizer inputs and return label start/end positions."""
    instr_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    instr_ids = tokenizer(instr_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if len(full_ids) == 0 or len(instr_ids) == 0:
        return None

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    label_start = len(instr_ids)
    label_end = len(full_ids)
    if label_start >= label_end:
        return None

    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_start": label_start,
        "label_end": label_end,
    }


def build_kv_cache_index(label_start: int, label_end: int, device: torch.device) -> List[torch.Tensor]:
    """
    Build kv_cache_index required by Rosetta model.
    
    kv_cache_index is a list, each element has shape (B, seq_len, 2):
    - [1, 0]: Use projector (project KV cache from large model to small model)
    - [-1, 0]: Don't use projector (directly use small model)
    
    For consistency check:
    - Instruction part uses [1, 0] (use rosetta projection)
    - Label part uses [-1, 0] (no projection, directly use small model)
    """
    # Instruction part: use projector [1, 0]
    instruction_length = label_start
    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(instruction_length - 1, 1).unsqueeze(0).to(device)
    
    # Label part: don't use projector [-1, 0]
    label_length = label_end - label_start
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(label_length + 1, 1).unsqueeze(0).to(device)
    
    return [instruction_index, label_index]


def greedy_from_logits(logits: torch.Tensor, label_start: int, label_end: int) -> torch.Tensor:
    """
    Extract greedy predictions for label segment from logits.
    logits: [1, seq_len, vocab]
    Returns: [label_len]
    """
    shift_logits = logits[:, :-1, :]  # Align with input_ids[:, 1:]
    start = max(label_start - 1, 0)
    end = max(label_end - 1, start)
    return shift_logits[0, start:end].argmax(dim=-1)


def compare_models(
    messages: List[Dict[str, str]],
    rosetta_model,
    llm_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    debug: bool = False,
) -> Tuple[int, int, int, int]:
    """Compare greedy prediction consistency of Rosetta model, SLM and LLM on label segment.
    
    Returns:
        (rosetta_match, slm_match, total_len, rosetta_win_count)
    """
    inputs = build_plain_inputs(messages, tokenizer, device, max_length)
    if inputs is None:
        return 0, 0, 0, 0

    label_start = inputs["label_start"]
    label_end = inputs["label_end"]
    
    # Build kv_cache_index
    kv_cache_index = build_kv_cache_index(label_start, label_end, device)
    
    # Build position_ids
    seq_len = inputs["input_ids"].shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.inference_mode():
        # 1. Rosetta model forward
        rosetta_output = rosetta_model(
            kv_cache_index=kv_cache_index,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids,
        )
        rosetta_logits = rosetta_output.logits
        
        # 2. LLM model forward
        llm_output = llm_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        llm_logits = llm_output.logits
        
        # 3. SLM (Base Model) forward - directly use base model inside Rosetta
        slm_model = rosetta_model.model_list[0]
        slm_output = slm_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        slm_logits = slm_output.logits

    rosetta_preds = greedy_from_logits(rosetta_logits, 1, label_end - label_start + 1)
    llm_preds = greedy_from_logits(llm_logits, label_start, label_end)
    slm_preds = greedy_from_logits(slm_logits, label_start, label_end)

    if debug:
        print("\n" + "="*80)
        print("🔍 DEBUG INFO (Comparison: Rosetta vs SLM vs LLM)")
        print("="*80)
        
        # 1. Input Context (Instruction)
        input_ids = inputs["input_ids"][0]
        instr_text = tokenizer.decode(input_ids[:label_start], skip_special_tokens=False)
        print(f"\n📝 [Instruction (Last 200 chars)]:\n...{instr_text[-200:]!r}")
        
        # 2. Target Label (Ground Truth)
        label_ids = input_ids[label_start:label_end]
        label_text = tokenizer.decode(label_ids, skip_special_tokens=False)
        print(f"\n🎯 [Target Label]:\n{label_text!r}")

        # 3. Predictions
        rosetta_text = tokenizer.decode(rosetta_preds, skip_special_tokens=False)
        slm_text = tokenizer.decode(slm_preds, skip_special_tokens=False)
        llm_text = tokenizer.decode(llm_preds, skip_special_tokens=False)
        
        print(f"\n🤖 [Rosetta Pred]:\n{rosetta_text!r}")
        print(f"\n👶 [SLM Pred]:\n{slm_text!r}")
        print(f"\n🧠 [LLM Pred]:\n{llm_text!r}")
        
        # 4. Token-level Diff
        print("\n⚖️  [Token Comparison (First 20)]")
        print(f"{'Pos':<5} {'Label':<15} {'Rosetta':<15} {'SLM':<15} {'LLM':<15} {'R==L':<5} {'S==L':<5} {'Note'}")
        print("-" * 100)
        
        limit = min(20, len(label_ids), len(rosetta_preds))
        for i in range(limit):
            lbl_tok = tokenizer.decode([label_ids[i]])
            ros_tok = tokenizer.decode([rosetta_preds[i]])
            slm_tok = tokenizer.decode([slm_preds[i]])
            llm_tok = tokenizer.decode([llm_preds[i]])
            
            r_eq_l = "✅" if rosetta_preds[i] == llm_preds[i] else "❌"
            s_eq_l = "✅" if slm_preds[i] == llm_preds[i] else "❌"
            
            note = ""
            if r_eq_l == "✅" and s_eq_l == "❌":
                note = "🎉 Rosetta Win"
            elif r_eq_l == "❌" and s_eq_l == "✅":
                note = "📉 SLM Win"
            
            print(f"{i:<5} {repr(lbl_tok):<15} {repr(ros_tok):<15} {repr(slm_tok):<15} {repr(llm_tok):<15} {r_eq_l:<5} {s_eq_l:<5} {note}")
        print("="*80 + "\n")

    # Assuming same tokenizer is used, lengths must be the same
    length = len(rosetta_preds)
    if length == 0:
        return 0, 0, 0, 0
        
    rosetta_match = sum(int(a == b) for a, b in zip(rosetta_preds, llm_preds))
    slm_match = sum(int(a == b) for a, b in zip(slm_preds, llm_preds))
    
    # Rosetta Win: Rosetta is correct, SLM is wrong
    rosetta_win = sum(int(r == l and s != l) for r, s, l in zip(rosetta_preds, slm_preds, llm_preds))
    
    return rosetta_match, slm_match, length, rosetta_win


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = create_dataset(args.dataset_type, **args.dataset_kwargs)
    print(f"Loading dataset: {args.dataset_type}, entries={len(dataset)}")

    # Build Rosetta model configuration
    model_config = {
        "model_name": "Rosetta",
        "rosetta_config": {
            "checkpoints_dir": args.checkpoints_dir,
            "base_model": args.slm,
            "teacher_model": args.llm,
            "include_response": args.include_response,
            "is_do_alignment": args.is_do_alignment,
            "alignment_strategy": args.alignment_strategy,
        }
    }
    
    eval_config = {
        "checkpoints_dir": args.checkpoints_dir
    }
    
    # Load Rosetta model
    print(f"Loading Rosetta model...")
    print(f"  - SLM: {args.slm}")
    print(f"  - LLM: {args.llm}")
    print(f"  - Checkpoints: {args.checkpoints_dir}")
    
    rosetta_model, tokenizer = load_rosetta_model(model_config, eval_config, device)
    rosetta_model.eval()
    
    # Directly use LLM inside Rosetta model to avoid duplicate loading
    # model_list[0] is SLM (base model), model_list[1] is LLM (teacher model)
    llm_model = rosetta_model.model_list[1]
    print(f"Using LLM model inside Rosetta for comparison")

    rosetta_match_count = 0
    slm_match_count = 0
    rosetta_win_count = 0
    total_count = 0

    max_samples = args.max_samples or len(dataset)
    for idx in range(min(max_samples, len(dataset))):
        messages = dataset[idx]
        
        # Debug mode: print first N samples
        is_debug = args.debug and (idx < args.debug_samples)
        
        r_match, s_match, total, r_win = compare_models(
            messages, rosetta_model, llm_model, tokenizer, device, args.max_length, debug=is_debug
        )
        rosetta_match_count += r_match
        slm_match_count += s_match
        rosetta_win_count += r_win
        total_count += total

        if (idx + 1) % args.print_every == 0:
            r_ratio = rosetta_match_count / total_count if total_count else 0.0
            s_ratio = slm_match_count / total_count if total_count else 0.0
            print(f"[{idx + 1}] Rosetta一致率={r_ratio:.4f} | SLM一致率={s_ratio:.4f} | Rosetta提升(Win)={rosetta_win_count}")

    r_ratio = rosetta_match_count / total_count if total_count else 0.0
    s_ratio = slm_match_count / total_count if total_count else 0.0
    
    print("\n" + "="*40)
    print("📊 统计结果 (Statistics)")
    print("="*40)
    print(f"总Token数: {total_count}")
    print(f"Rosetta 一致率: {r_ratio:.4f} ({rosetta_match_count}/{total_count})")
    print(f"SLM (Base) 一致率: {s_ratio:.4f} ({slm_match_count}/{total_count})")
    print(f"Rosetta 胜出数 (Rosetta对且SLM错): {rosetta_win_count}")
    print("-" * 40)
    
    diff = r_ratio - s_ratio
    print(f"相对提升: {diff:+.4f}")
    if diff > 0:
        print("✅ Rosetta 模型效果优于 Base 模型")
    else:
        print("❌ Rosetta 模型效果不如 Base 模型 (退化)")


if __name__ == "__main__":
    main()
