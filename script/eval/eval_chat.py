"""
Evaluation Script for Chat Models

Tests a trained model with multi-round conversations and records responses.

Usage:
    python script/eval/eval_chat.py --model_path ./checkpoints/sft_ultrachat/final
    python script/eval/eval_chat.py --model_path Qwen/Qwen3-1.7B  # baseline
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_response(model, tokenizer, messages, max_new_tokens=256):
    """Generate a response given conversation history."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the new tokens
    response_ids = outputs[0, inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response


def run_conversation(model, tokenizer, conversation_turns, conversation_name=""):
    """Run a multi-turn conversation and print results."""
    print(f"\n{'='*60}")
    print(f"Conversation: {conversation_name}")
    print('='*60)
    
    messages = []
    
    for i, user_msg in enumerate(conversation_turns):
        messages.append({"role": "user", "content": user_msg})
        print(f"\n[Turn {i+1}] User: {user_msg}")
        
        response = generate_response(model, tokenizer, messages)
        print(f"[Turn {i+1}] Assistant: {response}")
        
        messages.append({"role": "assistant", "content": response})
    
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/sft_ultrachat/final")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Test conversations
    conversations = [
        {
            "name": "Math Problem",
            "turns": [
                "What is 15 + 27?",
                "Now multiply that result by 3.",
                "What was the original sum before multiplication?",
            ]
        },
        {
            "name": "General Knowledge",
            "turns": [
                "What is the capital of France?",
                "What is a famous landmark there?",
                "How tall is it?",
            ]
        },
        {
            "name": "Coding Help",
            "turns": [
                "Write a Python function to calculate factorial.",
                "Can you add error handling for negative numbers?",
                "Now write a test case for it.",
            ]
        },
        {
            "name": "Context Retention",
            "turns": [
                "My name is Alice and I'm a software engineer.",
                "What programming languages should I learn?",
                "What's my name and profession?",
            ]
        },
    ]
    
    results = []
    for conv in conversations:
        messages = run_conversation(
            model, tokenizer, 
            conv["turns"], 
            conv["name"]
        )
        results.append({
            "name": conv["name"],
            "messages": messages
        })
    
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print('='*60)


if __name__ == "__main__":
    main()

