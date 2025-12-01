"""
UltraChat Dataset Generation Script

Generates responses for multi-round conversations from stingning/ultrachat using SGLang.
The output dataset maintains the exact same format as the original ultrachat.

Dataset format:
- Original ultrachat has 'data' field: list of alternating [user, assistant, user, assistant, ...]
- This script generates NEW assistant responses for the last user turn

Usage:
1. First launch the SGLang server:
   bash script/dataset/launch_ultrachat_server.sh
   
2. Then run the generation:
   python script/dataset/create_ultrachat.py \
       --model_path /share/public/public_models/Qwen3-8B \
       --api_url http://localhost:30000/v1 \
       --output_dir local/ultrachat_qwen3_8b_output \
       --num_items 10000
"""

from datetime import datetime
import pandas as pd
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import os
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import concurrent.futures
from time import sleep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses for UltraChat dataset using SGLang API"
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="/share/public/public_models/Qwen3-8B",
        help="Path or name of the model to use",
    )

    # API configuration
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:30000/v1",
        help="Base URL for the API endpoint (e.g., http://localhost:30000/v1)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for authentication (if required)",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=256,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=600,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=1.0,
        help="Delay between retries in seconds",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save partial results every N processed items",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="stingning/ultrachat",
        help="UltraChat dataset path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    
    # Generation configuration
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0, 
        help="Temperature for generation (0.0 for greedy/deterministic, 0.7 for sampling)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=1.0, 
        help="Top-p sampling parameter for generation (1.0 to disable)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=-1, 
        help="Top-k sampling parameter for generation (-1 to disable)"
    )
    parser.add_argument(
        "--min_p", 
        type=float, 
        default=0.0, 
        help="Min-p sampling parameter for generation"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for Qwen3 models",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="local/ultrachat_qwen3_8b_output",
        help="Base directory to save results",
    )

    parser.add_argument(
        "--is_print",
        action="store_true",
        default=False,
        help="Print all model responses to standard output",
    )

    # Debug configuration
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only process first item)",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=None,
        help="Number of items to process (None for all)",
    )

    # Recovery configuration
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint, processing only failed or missing items",
    )
    
    # Multi-turn configuration
    parser.add_argument(
        "--regenerate_all_turns",
        action="store_true",
        default=False,
        help="Regenerate all assistant turns (default: only regenerate last turn)",
    )

    args = parser.parse_args()
    return args


def format_conversation_to_messages(data: List[str], include_last_assistant: bool = True) -> List[Dict[str, str]]:
    """
    Convert ultrachat data format to chat messages.
    data: list of alternating [user, assistant, user, assistant, ...]
    
    Args:
        data: List of alternating user/assistant messages
        include_last_assistant: Whether to include the last assistant message (for context)
        
    Returns:
        List of message dictionaries with 'role' and 'content'
    """
    messages = []
    for i, text in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})
    
    # If we don't want the last assistant message (for generation), remove it
    if not include_last_assistant and messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    
    return messages


def initialize_tokenizer(model_path: str):
    """Initialize tokenizer for prompt formatting."""
    print(f"Initializing tokenizer from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_path}: {e}")
        print("Using default tokenizer, prompt formatting may be affected")
        return None


def prepare_prompts_for_generation(
    items: List[Dict[str, Any]], 
    tokenizer, 
    enable_thinking: bool = False,
    regenerate_all_turns: bool = False
) -> List[Dict[str, Any]]:
    """
    Prepare prompts for API requests using the tokenizer's chat template.
    
    NOTE: When regenerate_all_turns=True, this only prepares the FIRST turn.
    Subsequent turns are generated sequentially using the model's own responses as context.
    
    Args:
        items: List of ultrachat items with 'data' field
        tokenizer: Model tokenizer
        enable_thinking: Whether to enable thinking mode
        regenerate_all_turns: If True, will regenerate all turns sequentially (handled in main loop)
        
    Returns:
        List of generation tasks with prompt and metadata
    """
    generation_tasks = []
    
    for idx, item in enumerate(items):
        data = item.get("data", [])
        item_id = item.get("id", f"ultrachat_{idx:08d}")
        
        if not data:
            continue
        
        if regenerate_all_turns:
            # For sequential generation, we only prepare the first user message
            # Subsequent turns will be generated using model's own responses
            first_user_message = data[0] if data else ""
            if not first_user_message:
                continue
            
            messages = [{"role": "user", "content": first_user_message}]
            
            if tokenizer is not None:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            else:
                formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            # Count how many assistant turns we need to generate
            num_assistant_turns = len([i for i in range(len(data)) if i % 2 == 1])
            
            generation_tasks.append({
                "item_idx": idx,
                "item_id": item_id,
                "turn_idx": 1,  # First assistant turn
                "prompt": formatted_prompt,
                "original_data": data,
                "messages": messages,
                "num_turns": num_assistant_turns,  # Total turns to generate
                "user_messages": [data[i] for i in range(0, len(data), 2)],  # All user messages
            })
        else:
            # Only regenerate the last assistant turn
            messages = format_conversation_to_messages(data, include_last_assistant=False)
            
            if not messages or messages[-1]["role"] != "user":
                # No user message to respond to
                continue
            
            if tokenizer is not None:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            else:
                formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            generation_tasks.append({
                "item_idx": idx,
                "item_id": item_id,
                "turn_idx": len(data) - 1 if len(data) % 2 == 0 else len(data),  # Position for new assistant response
                "prompt": formatted_prompt,
                "original_data": data,
                "messages": messages,
            })
    
    return generation_tasks


def generate_sequential_conversation(
    args,
    task: Dict[str, Any],
    tokenizer,
) -> Optional[Dict[str, Any]]:
    """
    Generate a full multi-turn conversation sequentially.
    Each assistant response is generated by the model, then used as context for the next turn.
    
    Args:
        args: Command line arguments
        task: Generation task with user_messages and num_turns
        tokenizer: Model tokenizer
        
    Returns:
        Result dict with all generated responses
    """
    user_messages = task.get("user_messages", [])
    num_turns = task.get("num_turns", 1)
    item_id = task["item_id"]
    item_idx = task["item_idx"]
    
    # Build conversation with model's own responses
    messages = []
    generated_responses = []
    
    for turn_idx in range(num_turns):
        # Add user message
        if turn_idx < len(user_messages):
            user_msg = user_messages[turn_idx]
            messages.append({"role": "user", "content": user_msg})
        else:
            break
        
        # Format prompt with current conversation context
        if tokenizer is not None:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        else:
            formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # Generate assistant response
        try:
            response_content = api_request_with_retry(
                api_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model_path,
                prompt=formatted_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            response_content = response_content.strip()
            
            # Add model's response to conversation context
            messages.append({"role": "assistant", "content": response_content})
            generated_responses.append(response_content)
            
            if args.is_print:
                print(f"\n[{item_id}] Turn {turn_idx + 1}/{num_turns}")
                print(f"USER: {user_msg[:200]}...")
                print(f"ASSISTANT: {response_content[:300]}...")
                
        except Exception as e:
            print(f"Error generating turn {turn_idx + 1} for {item_id}: {str(e)}")
            # Continue with partial results
            break
    
    # Build final data in ultrachat format [user, assistant, user, assistant, ...]
    new_data = []
    for i, user_msg in enumerate(user_messages):
        new_data.append(user_msg)
        if i < len(generated_responses):
            new_data.append(generated_responses[i])
    
    return {
        "item_idx": item_idx,
        "item_id": item_id,
        "new_data": new_data,
        "generated_responses": generated_responses,
        "num_turns_generated": len(generated_responses),
    }


def make_api_request(
    api_url: str, 
    api_key: str, 
    model_name: str, 
    prompt: str, 
    max_new_tokens: int = 2048, 
    temperature: float = 0.7, 
    top_p: float = 0.8, 
    top_k: int = 20,
    min_p: float = 0.0,
    timeout: int = 600
) -> str:
    """Make a single API request to the model."""
    headers = {
        "Content-Type": "application/json",
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Use completions API for pre-formatted prompts
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    
    # Only add sampling parameters if they are enabled
    if top_k > 0:
        payload["top_k"] = top_k
    if min_p > 0:
        payload["min_p"] = min_p
    
    # For greedy decoding (temperature=0), disable top_p as well
    if temperature == 0.0:
        payload["top_p"] = 1.0
    
    try:
        response = requests.post(
            f"{api_url}/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["text"]
        else:
            raise Exception("No response content in API result")
            
    except requests.exceptions.Timeout:
        raise Exception("Request timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing API response: {str(e)}")


def api_request_with_retry(
    api_url: str, 
    api_key: str, 
    model_name: str, 
    prompt: str, 
    max_retries: int = 3, 
    retry_delay: float = 1.0, 
    **kwargs
) -> str:
    """Make an API request with retry logic."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return make_api_request(api_url, api_key, model_name, prompt, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                sleep(retry_delay * (2 ** attempt))
            else:
                print(f"Request failed after {max_retries} attempts: {str(e)}")
    
    raise last_exception


def process_single_task(args, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single generation task."""
    try:
        response_content = api_request_with_retry(
            api_url=args.api_url,
            api_key=args.api_key,
            model_name=args.model_path,
            prompt=task["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        
        # Clean up response (remove special tokens if any)
        response_content = response_content.strip()
        
        if args.is_print:
            print(f"\n===== TASK {task['item_id']} (Turn {task['turn_idx']}) =====")
            print(f"CONTEXT: {task['messages'][-1]['content'][:200]}...")
            print(f"RESPONSE: {response_content[:500]}...")
            print("=" * 50)
        
        return {
            "item_idx": task["item_idx"],
            "item_id": task["item_id"],
            "turn_idx": task["turn_idx"],
            "response": response_content,
            "original_data": task["original_data"],
        }
        
    except Exception as e:
        print(f"Error processing task {task['item_id']}: {str(e)}")
        return None


def process_sequential_conversations(
    args,
    tasks: List[Dict[str, Any]],
    tokenizer,
    max_concurrent: int = 64,
) -> List[Dict[str, Any]]:
    """
    Process multiple conversations with sequential turn generation.
    Each conversation's turns are generated sequentially (model response used as context),
    but different conversations can be processed in parallel.
    
    Args:
        args: Command line arguments
        tasks: List of generation tasks
        tokenizer: Model tokenizer
        max_concurrent: Max number of conversations to process in parallel
        
    Returns:
        List of results with generated conversations
    """
    results = []
    batch_start_time = time.time()
    
    # Reduce concurrency since each task now involves multiple API calls
    effective_concurrent = min(max_concurrent // 4, 64)  # Each conversation makes multiple calls
    effective_concurrent = max(effective_concurrent, 1)
    
    print(f"Processing {len(tasks)} conversations with sequential turn generation")
    print(f"Effective concurrency: {effective_concurrent} (reduced due to multi-turn)")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrent) as executor:
        future_to_task = {}
        for task in tasks:
            future = executor.submit(generate_sequential_conversation, args, task, tokenizer)
            future_to_task[future] = task
        
        completed_count = 0
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task), desc="Processing conversations"):
            task = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                completed_count += 1
                
                # Progress logging
                if completed_count % max(1, len(tasks) // 10) == 0:
                    elapsed = time.time() - batch_start_time
                    avg_time = elapsed / completed_count
                    remaining = len(tasks) - completed_count
                    
                    print(f"\nProgress: {completed_count}/{len(tasks)} ({completed_count/len(tasks)*100:.1f}%)")
                    print(f"Elapsed: {elapsed/60:.1f}min, Avg: {avg_time:.2f}s/conversation")
                    print(f"Estimated remaining: {avg_time * remaining / 60:.1f}min")
                    
            except Exception as e:
                print(f"Error processing conversation {task['item_id']}: {str(e)}")
    
    batch_time = time.time() - batch_start_time
    print(f"\nBatch completed in {batch_time/60:.1f}min")
    print(f"Successfully processed: {len(results)}/{len(tasks)} conversations")
    
    # Calculate total turns generated
    total_turns = sum(r.get("num_turns_generated", 0) for r in results)
    print(f"Total turns generated: {total_turns}")
    
    return results


def concurrent_api_requests(
    args, 
    tasks: List[Dict[str, Any]], 
    max_concurrent: int = 256,
    start_time: float = None
) -> List[Dict[str, Any]]:
    """Process multiple tasks concurrently using API requests."""
    results = []
    partial_buffer = []
    batch_start_time = time.time()
    
    print(f"Starting concurrent processing of {len(tasks)} tasks, max concurrent: {max_concurrent}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_task = {}
        for task in tasks:
            future = executor.submit(process_single_task, args, task)
            future_to_task[future] = task
        
        completed_count = 0
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task), desc="Processing"):
            task = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    partial_buffer.append(result)
                completed_count += 1
                
                # Progress logging
                if completed_count % max(1, len(tasks) // 10) == 0:
                    elapsed = time.time() - batch_start_time
                    avg_time_per_task = elapsed / completed_count
                    remaining = len(tasks) - completed_count
                    estimated_remaining = avg_time_per_task * remaining
                    
                    print(f"\nProgress: {completed_count}/{len(tasks)} ({completed_count/len(tasks)*100:.1f}%)")
                    print(f"Elapsed: {elapsed/60:.1f}min, Avg: {avg_time_per_task:.2f}s/task")
                    print(f"Estimated remaining: {estimated_remaining/60:.1f}min")
                
                # Periodic partial save
                if args.save_every and len(partial_buffer) >= args.save_every:
                    save_partial_results(partial_buffer, args.output_dir)
                    partial_buffer = []
                    
            except Exception as e:
                print(f"Error processing task {task['item_id']}: {str(e)}")
    
    # Flush remaining partials
    if partial_buffer:
        save_partial_results(partial_buffer, args.output_dir)
    
    batch_time = time.time() - batch_start_time
    print(f"\nBatch completed in {batch_time/60:.1f}min")
    print(f"Successfully processed: {len(results)}/{len(tasks)} tasks")
    
    return results


def save_partial_results(results: List[Dict[str, Any]], output_dir: str):
    """Save partial results to CSV."""
    if not results:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ultrachat_partial_results.csv")
    
    df = pd.DataFrame([{
        "item_idx": r["item_idx"],
        "item_id": r["item_id"],
        "turn_idx": r["turn_idx"],
        "response": r["response"],
    } for r in results])
    
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)
    print(f"[Partial Save] Appended {len(df)} rows to {csv_path}")


def build_ultrachat_format_dataset(
    original_items: List[Dict[str, Any]], 
    results: List[Dict[str, Any]],
    regenerate_all_turns: bool = False
) -> List[Dict[str, Any]]:
    """
    Build the final dataset in ultrachat format with generated responses.
    
    Args:
        original_items: Original ultrachat items
        results: Generation results
        regenerate_all_turns: Whether all turns were regenerated
        
    Returns:
        List of items in ultrachat format (with 'data' field)
    """
    # Group results by item_idx
    results_by_item = {}
    for r in results:
        item_idx = r["item_idx"]
        if item_idx not in results_by_item:
            results_by_item[item_idx] = []
        results_by_item[item_idx].append(r)
    
    new_dataset = []
    
    for idx, original_item in enumerate(original_items):
        original_data = original_item.get("data", [])
        item_id = original_item.get("id", f"ultrachat_{idx:08d}")
        
        if idx not in results_by_item:
            # No results for this item, keep original
            new_dataset.append({
                "id": item_id,
                "data": original_data,
            })
            continue
        
        item_results = results_by_item[idx]
        
        if regenerate_all_turns:
            # For sequential generation, the result already contains the full new_data
            if item_results and "new_data" in item_results[0]:
                new_data = item_results[0]["new_data"]
            else:
                # Fallback: rebuild from individual turn results
                new_data = []
                # Sort by turn_idx
                sorted_results = sorted(item_results, key=lambda x: x.get("turn_idx", 0))
                for i, text in enumerate(original_data):
                    if i % 2 == 0:  # User turn
                        new_data.append(text)
                    else:  # Assistant turn
                        turn_result = next((r for r in sorted_results if r.get("turn_idx") == i), None)
                        if turn_result and "response" in turn_result:
                            new_data.append(turn_result["response"])
                        else:
                            new_data.append(text)
        else:
            # Only replace the last assistant turn
            new_data = list(original_data)
            if item_results:
                last_result = item_results[-1]
                turn_idx = last_result.get("turn_idx", len(new_data) - 1)
                response = last_result.get("response", "")
                
                # If turn_idx equals len(original_data), we're adding a new response
                if turn_idx >= len(new_data):
                    new_data.append(response)
                else:
                    # Replace the existing assistant response
                    new_data[turn_idx] = response
        
        new_dataset.append({
            "id": item_id,
            "data": new_data,
        })
    
    return new_dataset


def save_final_dataset(
    new_dataset: List[Dict[str, Any]], 
    output_dir: str,
    model_name: str
):
    """Save the final dataset in HuggingFace format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create HuggingFace dataset with ultrachat format
    # ultrachat format: {"id": str, "data": [str, str, ...]}
    dataset_dict = {
        "id": [item["id"] for item in new_dataset],
        "data": [item["data"] for item in new_dataset],
    }
    
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    # Save to disk
    dataset_path = os.path.join(output_dir, "dataset")
    hf_dataset.save_to_disk(dataset_path)
    print(f"Saved HuggingFace dataset with {len(hf_dataset)} conversations to {dataset_path}")
    
    # Also save as JSON for easy inspection
    json_path = os.path.join(output_dir, "ultrachat_generated.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON format to {json_path}")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "num_conversations": len(new_dataset),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_dataset": "stingning/ultrachat",
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return hf_dataset


def get_completed_items(output_dir: str) -> set:
    """Get set of item IDs that have been successfully processed."""
    completed = set()
    csv_path = os.path.join(output_dir, "ultrachat_partial_results.csv")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "item_id" in df.columns:
                completed.update(df["item_id"].unique())
        except Exception as e:
            print(f"Error reading partial results: {e}")
    
    return completed


def save_args_to_json(args, output_dir: str):
    """Save command line arguments to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    args_dict = vars(args)
    args_dict["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = os.path.join(output_dir, "run_args.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    
    print(f"Arguments saved to: {json_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load UltraChat dataset
    print(f"Loading UltraChat dataset from {args.dataset_path}")
    
    if args.num_items:
        items_to_load = args.num_items
    elif args.debug:
        items_to_load = 5
    else:
        items_to_load = None
    
    # Load dataset (streaming for large datasets)
    if items_to_load and items_to_load < 10:
        dataset = load_dataset(args.dataset_path, split=args.split)
        if items_to_load:
            dataset = dataset.select(range(min(items_to_load, len(dataset))))
    else:
        # Use streaming for very large datasets
        dataset = load_dataset(args.dataset_path, split=args.split, streaming=True)
        if items_to_load:
            dataset = dataset.take(items_to_load)
    
    # Convert to list with IDs
    all_items = []
    for idx, item in enumerate(tqdm(dataset, desc="Loading dataset", total=items_to_load)):
        item_with_id = dict(item)
        item_with_id["id"] = f"ultrachat_{idx:08d}"
        all_items.append(item_with_id)
        
        if items_to_load and len(all_items) >= items_to_load:
            break
    
    print(f"Loaded {len(all_items)} conversations from UltraChat")
    
    # Print first item as example
    if all_items:
        print("\n" + "=" * 60)
        print("📋 FIRST DATA ITEM EXAMPLE:")
        print("=" * 60)
        first_item = all_items[0]
        first_data = first_item.get("data", [])
        print(f"ID: {first_item.get('id', 'N/A')}")
        print(f"Number of messages: {len(first_data)}")
        print(f"Number of turns: {len(first_data) // 2}")
        for i, msg in enumerate(first_data):
            role = "USER" if i % 2 == 0 else "ASSISTANT"
            # Truncate long messages for display
            display_msg = msg[:500] + "..." if len(msg) > 500 else msg
            print(f"\n[{role}] (Turn {i // 2 + 1}):")
            print(display_msg)
        print("=" * 60 + "\n")
    
    # Get completed items if resuming
    completed_items = get_completed_items(args.output_dir) if args.resume else set()
    
    if args.resume and completed_items:
        print(f"Resuming: {len(completed_items)} items already completed")
        all_items = [item for item in all_items if item["id"] not in completed_items]
        print(f"Remaining items to process: {len(all_items)}")
    
    if not all_items:
        print("No items to process!")
        return
    
    print(f"\nProcessing {len(all_items)} conversations")
    print(f"Output directory: {args.output_dir}")
    print(f"API URL: {args.api_url}")
    print(f"Max concurrent requests: {args.max_concurrent_requests}")
    print(f"Enable thinking: {args.enable_thinking}")
    print(f"Regenerate all turns: {args.regenerate_all_turns}")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(args.model_path)
    
    # Prepare generation tasks
    print("Preparing generation tasks...")
    tasks = prepare_prompts_for_generation(
        all_items, 
        tokenizer, 
        enable_thinking=args.enable_thinking,
        regenerate_all_turns=args.regenerate_all_turns
    )
    print(f"Prepared {len(tasks)} generation tasks")
    
    # Process tasks
    processing_start_time = time.time()
    
    if args.regenerate_all_turns:
        # Sequential generation: each conversation is processed turn-by-turn
        # where model's responses are used as context for subsequent turns
        print("\n🔄 Using SEQUENTIAL generation mode (model responses used as context)")
        results = process_sequential_conversations(
            args=args,
            tasks=tasks,
            tokenizer=tokenizer,
            max_concurrent=args.max_concurrent_requests,
        )
    else:
        # Single-turn generation: can be fully parallelized
        results = concurrent_api_requests(
            args=args,
            tasks=tasks,
            max_concurrent=args.max_concurrent_requests,
            start_time=start_time,
        )
    
    # Build final dataset
    print("\nBuilding final dataset in UltraChat format...")
    new_dataset = build_ultrachat_format_dataset(
        all_items, 
        results,
        regenerate_all_turns=args.regenerate_all_turns
    )
    
    # Save final dataset
    hf_dataset = save_final_dataset(
        new_dataset, 
        args.output_dir,
        model_name=args.model_path
    )
    
    # Save arguments
    save_args_to_json(args, args.output_dir)
    
    # Final timing summary
    total_time = time.time() - start_time
    processing_time = time.time() - processing_start_time
    
    print("\n" + "=" * 60)
    print("🎉 Generation completed!")
    print("=" * 60)
    print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/60:.1f}min ({total_time:.1f}s)")
    print(f"Processing time: {processing_time/60:.1f}min ({processing_time:.1f}s)")
    print(f"Successfully processed: {len(results)} tasks")
    print(f"Final dataset size: {len(new_dataset)} conversations")
    
    if results and total_time > 0:
        print(f"Average time per task: {total_time/len(results):.2f}s")
        print(f"Processing speed: {len(results)/(total_time/60):.1f} tasks/min")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

