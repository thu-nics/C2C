"""
Test examples.yaml against the contextual OpenAI-compatible server.

This script reads examples from examples.yaml and sends them to the
contextual HTTP server, demonstrating the drop_messages feature.

Message IDs are assigned per-message:
- ID 0 = system prompt
- ID 1 = first user message
- ID 2 = first assistant response (generated)
- ID 3 = second user message
- ID 4 = second assistant response (generated)
- etc.

Usage:
  1. Start the server first:
     python script/context/launch_server.py --model /share/public/public_models/Qwen3-1.7B --port 30000

  2. Run this test script:
     python script/context/test_server_examples.py --port 30000
     python script/context/test_server_examples.py --port 30000 --example_name math_followup
"""

import argparse
import requests
import yaml
from typing import Any, Dict, List


def load_examples(yaml_path: str) -> list:
    """Load examples from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f).get("examples", [])


def test_example_generate_endpoint(
    base_url: str,
    example: dict,
    *,
    max_new_tokens: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test an example using the /generate endpoint.
    
    Uses the new per-message ID format:
    - user_messages: list of user message contents
    - drop_messages: dict mapping user_message_id -> list of message IDs to drop
    """
    name = example.get("name", "unnamed")
    system_prompt = example.get("system_prompt", "You are a helpful assistant.")
    user_messages = example.get("user_messages", [])
    drop_messages_config = example.get("drop_messages", {})
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Example: {name}")
        print(f"Drop config: {drop_messages_config}")
        print(f"{'='*60}")
    
    results = []
    # Build accumulated messages: system + user/assistant pairs
    accumulated_messages = [{"role": "system", "content": system_prompt}]
    
    # Message ID tracking:
    # ID 0 = system
    # ID 1 = first user, ID 2 = first assistant
    # ID 3 = second user, ID 4 = second assistant
    # etc.
    message_id = 1  # Start after system (ID 0)
    
    for user_round, user_content in enumerate(user_messages, start=1):
        user_id = message_id
        assistant_id = message_id + 1
        
        # Add user message
        accumulated_messages.append({"role": "user", "content": user_content})
        
        if verbose:
            print(f"\n[Round {user_round}] User (ID={user_id}): {user_content}")
        
        # Send request to server.
        # Important: pass the full drop schedule so drops persist across later rounds.
        payload = {
            "messages": accumulated_messages,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "drop_messages": drop_messages_config or None,
            }
        }
        
        try:
            response = requests.post(f"{base_url}/generate", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            assistant_text = result.get("text", "")
            segment_map = result.get("segment_id_map", {})
            
            # Best-effort suffix for the current turn, if configured.
            drop_ids = drop_messages_config.get(user_id, drop_messages_config.get(str(user_id), []))
            suffix = f" (drop_messages[{user_id}]={drop_ids})" if drop_ids else ""
            if verbose:
                print(f"Assistant (ID={assistant_id}){suffix}: {assistant_text}")
                print(f"Segment map: {segment_map}")
            
            # Add assistant response to accumulated messages for next round
            accumulated_messages.append({"role": "assistant", "content": assistant_text})
            
            results.append({
                "round": user_round,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "user": user_content,
                "assistant": assistant_text,
                "drop_ids": drop_ids,
                "segment_map": segment_map,
            })
            
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed: {e}")
            results.append({
                "round": user_round,
                "user_id": user_id,
                "user": user_content,
                "error": str(e),
            })
        
        message_id += 2  # Move to next user/assistant pair
    
    return {"name": name, "results": results}


def test_example_chat_completions_endpoint(
    base_url: str,
    example: dict,
    *,
    max_new_tokens: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test an example using the /v1/chat/completions endpoint (OpenAI-compatible).
    """
    name = example.get("name", "unnamed")
    system_prompt = example.get("system_prompt", "You are a helpful assistant.")
    user_messages = example.get("user_messages", [])
    drop_messages_config = example.get("drop_messages", {})
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Example: {name} (OpenAI-compatible endpoint)")
        print(f"Drop config: {drop_messages_config}")
        print(f"{'='*60}")
    
    results = []
    accumulated_messages = [{"role": "system", "content": system_prompt}]
    message_id = 1  # Start after system (ID 0)
    
    for user_round, user_content in enumerate(user_messages, start=1):
        user_id = message_id
        assistant_id = message_id + 1
        
        accumulated_messages.append({"role": "user", "content": user_content})
        
        if verbose:
            print(f"\n[Round {user_round}] User (ID={user_id}): {user_content}")
        
        # Send request to server (OpenAI format).
        # Important: pass the full drop schedule so drops persist across later rounds.
        payload = {
            "model": "contextual-model",
            "messages": [{"role": m["role"], "content": m["content"]} for m in accumulated_messages],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "drop_messages": drop_messages_config or None,
        }
        
        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            assistant_text = result["choices"][0]["message"]["content"]
            segment_map = result.get("segment_id_map", {})
            
            drop_ids = drop_messages_config.get(user_id, drop_messages_config.get(str(user_id), []))
            suffix = f" (drop_messages[{user_id}]={drop_ids})" if drop_ids else ""
            if verbose:
                print(f"Assistant (ID={assistant_id}){suffix}: {assistant_text}")
                print(f"Segment map: {segment_map}")
            
            # Add assistant response to accumulated messages for next round
            accumulated_messages.append({"role": "assistant", "content": assistant_text})
            
            results.append({
                "round": user_round,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "user": user_content,
                "assistant": assistant_text,
                "drop_ids": drop_ids,
                "segment_map": segment_map,
            })
            
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed: {e}")
            results.append({
                "round": user_round,
                "user_id": user_id,
                "user": user_content,
                "error": str(e),
            })
        
        message_id += 2
    
    return {"name": name, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Test examples.yaml against contextual HTTP server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--examples_yaml", type=str, default="script/context/examples.yaml", 
                        help="Path to examples YAML file")
    parser.add_argument("--example_name", type=str, default=None, 
                        help="Run only a specific example by name")
    parser.add_argument("--endpoint", type=str, choices=["generate", "chat", "both"], default="generate",
                        help="Which endpoint to test: generate, chat (OpenAI), or both")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate per turn")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    verbose = not args.quiet
    
    print(f"Testing against server at {base_url}")
    print(f"Loading examples from {args.examples_yaml}")
    
    examples = load_examples(args.examples_yaml)
    if not examples:
        print(f"No examples found in {args.examples_yaml}")
        return
    
    if args.example_name:
        examples = [e for e in examples if e.get("name") == args.example_name]
        if not examples:
            print(f"No example found with name '{args.example_name}'")
            return
    
    print(f"Found {len(examples)} example(s) to test")
    
    # Test server connectivity first
    try:
        requests.get(f"{base_url}/docs", timeout=5)
        print("Server is reachable!")
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Could not reach server at {base_url}: {e}")
        print("Make sure the server is running with:")
        print(f"  python script/context/launch_server.py --model <model_path> --port {args.port}")
        return
    
    all_results = []
    
    for example in examples:
        if args.endpoint in ["generate", "both"]:
            result = test_example_generate_endpoint(
                base_url, example, max_new_tokens=args.max_new_tokens, verbose=verbose
            )
            all_results.append(result)
        
        if args.endpoint in ["chat", "both"]:
            result = test_example_chat_completions_endpoint(
                base_url, example, max_new_tokens=args.max_new_tokens, verbose=verbose
            )
            all_results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        name = result["name"]
        rounds = len(result["results"])
        errors = sum(1 for r in result["results"] if "error" in r)
        status = "✓ PASSED" if errors == 0 else f"✗ FAILED ({errors} errors)"
        print(f"  {name}: {status} ({rounds} rounds)")


if __name__ == "__main__":
    main()
