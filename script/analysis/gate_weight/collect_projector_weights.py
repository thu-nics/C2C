import argparse
import os
import json
from typing import Dict, List, Any

import torch
import pandas as pd

from rosetta.utils.evaluate import load_rosetta_model
from rosetta.train.dataset_adapters import MMLUChatDataset


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_inputs(tokenizer, conversation: List[Dict[str, str]], device: torch.device):
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def load_model_and_tokenizer(checkpoint_dir: str, device: torch.device):
    model_config = {
        "model_name": "Rosetta",
        "rosetta_config": {
            "checkpoints_dir": checkpoint_dir,
            "base_model": "Qwen/Qwen3-0.6B",
            "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
        },
    }
    eval_config = {"checkpoints_dir": checkpoint_dir}

    rosetta_model, tokenizer = load_rosetta_model(model_config, eval_config, device)
    rosetta_model.eval()

    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n{% endif %}{% endfor %}<|assistant|>\n"

    return rosetta_model, tokenizer


@torch.no_grad()
def run_and_collect(model, dataset, tokenizer, device: torch.device, num_samples: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for sample_idx in range(min(num_samples, len(dataset))):
        conv = dataset[sample_idx]
        inputs = build_inputs(tokenizer, conv, device)

        # Prepare kv_cache indexes similar to compare_projector_terms.py to trigger forward once
        full_length = inputs["input_ids"].shape[1]
        instruction_length = full_length - 1
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(instruction_length, 1).unsqueeze(0).to(device)
        response_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
        kv_cache_list = [instruction_index, response_index]

        _ = model.generate(
            input_ids=inputs["input_ids"],
            kv_cache_index=kv_cache_list,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )

        # After generation, each projector should expose capture attributes
        for proj_idx, proj in enumerate(model.projector_list):
            # Expect attributes set in C2CProjector.forward
            norm_key_scalar = getattr(proj, "last_norm_key_scalar", None)
            norm_value_scalar = getattr(proj, "last_norm_value_scalar", None)
            key_gate_logit = getattr(proj, "last_key_gate_logit", None)
            value_gate_logit = getattr(proj, "last_value_gate_logit", None)

            # Convert tensors to nested Python lists for CSV (JSON-encoded)
            norm_key_scalar_list = norm_key_scalar.tolist() if norm_key_scalar is not None else None
            norm_value_scalar_list = norm_value_scalar.tolist() if norm_value_scalar is not None else None

            row = {
                "sample_index": sample_idx,
                "projector_index": proj_idx,
                "norm_key_scalar": json.dumps(norm_key_scalar_list) if norm_key_scalar_list is not None else None,
                "norm_value_scalar": json.dumps(norm_value_scalar_list) if norm_value_scalar_list is not None else None,
                "key_gate_logit": key_gate_logit,
                "value_gate_logit": value_gate_logit,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-projector scalar weights and gate logits into CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use (e.g., validation)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--output_csv", type=str, default="projector_weights.csv", help="Output CSV path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    dataset = MMLUChatDataset(split=args.split, num_samples=args.num_samples)

    df = run_and_collect(model, dataset, tokenizer, device, args.num_samples)

    ensure_dir(os.path.dirname(args.output_csv) or ".")
    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()


