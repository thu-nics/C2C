"""
Contextual Chat Example

Demonstrates a 'ContextualModel' wrapper for KV cache manipulation:
1. append(input_ids, id) - Prefill tokens into cache with a segment ID.
2. generate(input_ids, id) - Generate response, tagging all tokens with ID.
3. drop_context(remove_ids) - Remove cached tokens by segment ID.

Segment IDs are assigned by message index:
- ID 0 = system prompt
- ID 1 = first user message
- ID 2 = first assistant response (generated)
- ID 3 = second user message
- ID 4 = second assistant response (generated)
- etc.

Usage:
    python script/context/contextual_chat_example.py --model_name /share/public/public_models/Qwen3-8B
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
    user_messages = example.get("user_messages", [])
    drop_messages_config = example.get("drop_messages", {})
    
    print(f"\n{'='*60}")
    print(f"Example: {name} | Drop config: {drop_messages_config}")
    print(f"{'='*60}")
    
    ctx_model.reset()
    
    # Build list of all messages with their IDs
    # ID 0 = system, ID 1 = first user, ID 2 = first assistant (generated), etc.
    all_messages = [{"role": "system", "content": system_prompt}]
    
    # System prompt (ID=0)
    sys_ids = ctx_model.apply_chat_template(all_messages, add_generation_prompt=False, enable_thinking=False)
    ctx_model.append(sys_ids, id=0)
    
    message_id = 1  # Next message ID (system was 0)
    
    for user_round, user_content in enumerate(user_messages, start=1):
        user_id = message_id
        all_messages.append({"role": "user", "content": user_content})
        
        print(f"\n[Round {user_round}] User (ID={user_id}): {user_content}")
        
        # Append user message tokens with user_id (no generation prompt)
        full_ids_no_gen = ctx_model.apply_chat_template(all_messages, add_generation_prompt=False, enable_thinking=False)
        user_new_ids = full_ids_no_gen[:, ctx_model.seq_length:]
        if user_new_ids.numel() > 0:
            ctx_model.append(user_new_ids, id=user_id)
        
        # Now get the generation prompt tokens
        message_id += 1
        assistant_id = message_id
        
        full_ids_with_gen = ctx_model.apply_chat_template(all_messages, add_generation_prompt=True, enable_thinking=False)
        gen_prompt_ids = full_ids_with_gen[:, ctx_model.seq_length:]
        
        # Get drop_ids for this user message from config (keys may be int or str from YAML)
        drop_ids = drop_messages_config.get(user_id, drop_messages_config.get(str(user_id), []))
        
        # Generate with separate input_id (for gen prompt) and output_id (for generated tokens)
        response = ctx_model.generate_step(
            gen_prompt_ids, 
            input_id=user_id,      # Generation prompt tokens get user's ID
            output_id=assistant_id, # Generated tokens get assistant's ID
            drop_ids_after_prefill=drop_ids
        )
        
        all_messages.append({"role": "assistant", "content": response})
        
        suffix = f" (dropped IDs {drop_ids} after prefill)" if drop_ids else ""
        print(f"Assistant (ID={assistant_id}){suffix}: {response}")
        
        message_id += 1  # Ready for next user message
    
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
    
    parser.add_argument("--examples_yaml", type=str, default="script/context/examples.yaml")
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
