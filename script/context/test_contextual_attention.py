"""
Chat-style example runner for `generate_with_contextual_mask`.

This intentionally mirrors `script/context/contextual_chat_example.py`'s UX:
- sequential multi-round generation
- message IDs: 0=system, 1=user, 2=assistant, 3=user, ...

Unlike the chat example (which uses `ContextualModel`), this script uses ONLY the
standard HF model + `generate_with_contextual_mask` and does NOT run a no-drop baseline.

Usage:
  python script/context/test_contextual_attention.py --model_name /share/public/public_models/Qwen3-1.7B
"""

import argparse
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.context.context_attention import generate_with_contextual_mask


def load_examples(yaml_path: str) -> list:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f).get("examples", [])


def run_example(model, tokenizer, example: dict, max_new_tokens: int):
    name = example.get("name", "unnamed")
    system_prompt = example.get("system_prompt", "You are a helpful assistant.")
    user_messages = example.get("user_messages", [])
    drop_messages_config = example.get("drop_messages", {}) or {}

    print(f"\n{'='*60}")
    print(f"Example: {name} | Drop config: {drop_messages_config}")
    print(f"{'='*60}")

    # Message IDs:
    #   ID 0 = system prompt
    #   ID 1 = first user message
    #   ID 2 = first assistant response (generated)
    #   ID 3 = second user message
    #   ...
    messages = [{"role": "system", "content": system_prompt}]

    for user_round, user_content in enumerate(user_messages, start=1):
        messages.append({"role": "user", "content": user_content})
        user_id = len(messages) - 1

        print(f"\n[Round {user_round}] User (ID={user_id}): {user_content}")

        drop_ids = drop_messages_config.get(user_id, drop_messages_config.get(str(user_id), []))
        response = generate_with_contextual_mask(
            model,
            tokenizer,
            messages,
            messages_to_drop=drop_messages_config,
            max_new_tokens=max_new_tokens,
        )

        assistant_id = user_id + 1
        messages.append({"role": "assistant", "content": response})

        suffix = f" (dropped IDs {drop_ids} after prefill)" if drop_ids else ""
        print(f"Assistant (ID={assistant_id}){suffix}: {response}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/share/public/public_models/Qwen3-1.7B")
    parser.add_argument("--examples_yaml", type=str, default="script/context/examples.yaml")
    parser.add_argument("--example_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    examples = load_examples(args.examples_yaml)
    if not examples:
        raise SystemExit(f"No examples found in {args.examples_yaml}")

    if args.example_name is not None:
        examples = [e for e in examples if e.get("name") == args.example_name]
        if not examples:
            raise SystemExit(f"No example named {args.example_name} in {args.examples_yaml}")

    print(f"Loading tokenizer/model from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    for ex in examples:
        run_example(model, tokenizer, ex, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()

