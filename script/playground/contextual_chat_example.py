"""
Contextual Chat Example

Demonstrates a 'ContextualModel' wrapper for KV cache manipulation:
1. append(input_ids, id) - Prefill tokens into cache with a segment ID.
2. generate(input_ids, id) - Generate response, tagging all tokens with ID.
3. drop_context(remove_ids) - Remove cached tokens by segment ID.

Usage:
    python script/playground/contextual_chat_example.py --model_name /share/public/public_models/Qwen3-8B
"""

import argparse
from collections import Counter
import torch
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM

from rosetta.context.context import ContextualModel


def load_examples(yaml_path: str) -> list:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f).get("examples", [])


def run_example(ctx_model: ContextualModel, example: dict):
    name = example.get("name", "unnamed")
    system_prompt = example.get("system_prompt", "You are a helpful assistant.")
    user_messages = [m["content"] for m in example.get("messages", []) if m["role"] == "user"]
    drop_rounds_config = example.get("drop_rounds", {})
    
    # Convert to dict if it's a list (backward compatibility)
    if isinstance(drop_rounds_config, list):
        # Old format: [0, 1] means drop these rounds at the final round
        drop_rounds_config = {len(user_messages): drop_rounds_config}
    
    print(f"\n{'='*60}")
    print(f"Example: {name} | Drop config: {drop_rounds_config}")
    print(f"{'='*60}")
    
    ctx_model.reset()
    messages = [{"role": "system", "content": system_prompt}]
    
    # System prompt (ID=0)
    sys_ids = ctx_model.apply_chat_template(messages, add_generation_prompt=False, enable_thinking=False)
    ctx_model.append(sys_ids, id=0)
    
    # Process each user message
    for round_id, user_content in enumerate(user_messages, start=1):
        messages.append({"role": "user", "content": user_content})
        print(f"\n[Round {round_id}] User: {user_content}")
        
        full_ids = ctx_model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        # Use seq_length (not cache_length) to slice - cache_length may decrease after drops
        new_ids = full_ids[:, ctx_model.seq_length:]
        
        # Get drop_ids for this round from config (keys may be int or str from YAML)
        drop_ids = drop_rounds_config.get(round_id, drop_rounds_config.get(str(round_id), []))
        
        response = ctx_model.generate(new_ids, id=round_id, drop_ids_after_prefill=drop_ids)
        messages.append({"role": "assistant", "content": response})
        
        suffix = f" (dropped IDs {drop_ids} after prefill)" if drop_ids else ""
        print(f"Assistant{suffix}: {response}")
    
    print(f"\nFinal cache: {Counter(ctx_model.round_ids)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        # default="checkpoints/sft_contextual/final",
        default="/share/public/public_models/Qwen3-1.7B",
        help="HuggingFace model name"
    )
    
    parser.add_argument("--examples_yaml", type=str, default="script/train/examples.yaml")
    parser.add_argument("--example_name", type=str, default=None)
    args = parser.parse_args()

    examples = load_examples(args.examples_yaml)
    if not examples:
        print(f"No examples found in {args.examples_yaml}")
        return
    
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    ctx_model = ContextualModel(model, tokenizer)
    
    if args.example_name:
        examples = [e for e in examples if e.get("name") == args.example_name]
    
    for example in examples:
        run_example(ctx_model, example)


if __name__ == "__main__":
    main()
