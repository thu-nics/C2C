"""
Direct Dolly dataset generation script - builds user prompts from Dolly fields and lets the model generate responses.

Inputs:
- Dolly dataset directly from HuggingFace (databricks/databricks-dolly-15k)

Outputs:
- A csv file with the model responses.
- A huggingface dataset with the model responses.
    - Contains columns: "id", "input_text", "model_response", and "is_finished". Each row corresponds to a query.
"""

from datetime import datetime
import pandas as pd
import requests
from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import os
import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import concurrent.futures
import threading
from time import sleep

# os.environ["HF_DATASETS_OFFLINE"] = "1"

@dataclass
class InputItem:
    """Data class representing a general input item for model processing."""

    id: str
    input_text: str
    model_reasoning: Optional[str] = None
    model_response: Optional[str] = None
    is_finished: Optional[bool] = None

    def __str__(self) -> str:
        return f"Item {self.id}:\n{self.input_text}\n\nResponse:\n{self.model_response}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process OpenHermes inputs with models using API requests"
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-30BA3B-Instruct-2507",
        help="Path or name of the model to use",
    )

    # API configuration
    parser.add_argument(
        "--api_url",
        type=str,
        required=True,
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
        default=16,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=6000,
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
        default="databricks/databricks-dolly-15k",
        help="Dolly dataset path (e.g., databricks/databricks-dolly-15k)",
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
        default=32768,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p sampling parameter for generation"
    )
    parser.add_argument(
        "--top_k", type=int, default=-1, help="Top-k sampling parameter for generation"
    )
    parser.add_argument(
        "--min_p", type=float, default=-1.0, help="Min-p sampling parameter for generation"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="openhermes_output",
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
        help="Number of items to process (for testing)",
    )

    # Recovery configuration
    parser.add_argument(
        "--item_ids",
        type=str,
        default=None,
        help="Comma-separated list of specific item IDs to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint, processing only failed or missing items",
    )

    args = parser.parse_args()

    # Convert item IDs string to list if provided
    if args.item_ids:
        args.item_ids = [id.strip() for id in args.item_ids.split(",")]

    return args


def save_results(problems, output_dir, append_mode=True):
    """Finalize: read accumulated CSV (from partial saves) and build HF dataset.
    If the CSV doesn't exist, create it from current problems once.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "Dolly_generated_results.csv")

    if os.path.exists(csv_path):
        try:
            combined_df = pd.read_csv(csv_path, low_memory=False)
            print(f"Found existing CSV with {len(combined_df)} records. Skipping re-append and building dataset directly.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            combined_df = None
    else:
        # No CSV yet; create from current in-memory problems if provided
        if not problems:
            print("No CSV found and no in-memory results to save. Nothing to finalize.")
            return
        new_df = pd.DataFrame([problem.__dict__ for problem in problems])
        new_df["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_df.to_csv(csv_path, index=False)
        combined_df = new_df
        print(f"Created new CSV with {len(new_df)} results at {csv_path}")

    if combined_df is None or len(combined_df) == 0:
        print("No data available to build HuggingFace dataset.")
        return

    # Filter out rows with empty or None model_response to prevent ArrowTypeError
    before_count = len(combined_df)
    if "model_response" in combined_df.columns:
        combined_df = combined_df[
            combined_df["model_response"].notna()
            & (combined_df["model_response"].astype(str).str.strip() != "")
        ]
    after_count = len(combined_df)
    removed = before_count - after_count
    if removed > 0:
        print(f"Filtered out {removed} rows with empty model_response")

    if len(combined_df) == 0:
        print("No valid rows remain after filtering. Skipping dataset build.")
        return

    # Normalize dtypes to avoid pyarrow ArrowTypeError (e.g., floats/None in string fields)
    try:
        if "id" in combined_df.columns:
            combined_df["id"] = combined_df["id"].astype(str)
        if "input_text" in combined_df.columns:
            combined_df["input_text"] = combined_df["input_text"].fillna("").astype(str)
        if "model_reasoning" in combined_df.columns:
            combined_df["model_reasoning"] = combined_df["model_reasoning"].fillna("").astype(str)
        else:
            combined_df["model_reasoning"] = ""
        if "model_response" in combined_df.columns:
            combined_df["model_response"] = combined_df["model_response"].fillna("").astype(str)
        if "is_finished" in combined_df.columns:
            combined_df["is_finished"] = combined_df["is_finished"].fillna(True).astype(bool)
        else:
            combined_df["is_finished"] = True
    except Exception as e:
        print(f"Warning: Failed to normalize dtypes: {e}")

    # Convert combined data to HuggingFace dataset
    dataset_dict = {
        "id": combined_df["id"].tolist(),
        "input_text": combined_df["input_text"].tolist(),
        "model_reasoning": combined_df.get("model_reasoning", pd.Series([None]*len(combined_df))).tolist(),
        "model_response": combined_df["model_response"].tolist(),
        "is_finished": combined_df.get("is_finished", pd.Series([True]*len(combined_df))).tolist(),
    }

    hf_dataset = Dataset.from_dict(dataset_dict)

    dataset_path = os.path.join(output_dir, "dataset")
    hf_dataset.save_to_disk(dataset_path)
    print(f"Saved HuggingFace dataset with {len(hf_dataset)} problems to {dataset_path}")

    hf_dataset_finished = hf_dataset.filter(lambda x: x["is_finished"] == True)
    print(f"Filtered dataset with {len(hf_dataset_finished)} finished problems")

    dataset_finished_path = os.path.join(output_dir, "dataset_finished")
    hf_dataset_finished.save_to_disk(dataset_finished_path)
    print(f"Saved finished HuggingFace dataset with {len(hf_dataset_finished)} problems to {dataset_finished_path}")


def save_partial_results_csv(partials, output_dir):
    """Append partial results to CSV quickly (no dataset conversion)."""
    if not partials:
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "Dolly_generated_results.csv")
    df = pd.DataFrame([p.__dict__ for p in partials])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)
    print(f"[Partial Save] Appended {len(df)} rows to {csv_path}")


def get_completed_items(output_dir: str) -> set:
    """Get set of item IDs that have been successfully processed."""
    completed = set()
    try:
        # Look for all CSV files in subdirectories
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".csv"):
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        if "id" in df.columns:
                            completed.update(df["id"].unique())
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
    except Exception as e:
        print(f"Error getting completed items: {e}")
    return completed


def initialize_tokenizer(model_path):
    """Initialize tokenizer for prompt formatting."""
    print(f"Initializing tokenizer from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_path}: {e}")
        print("Using default tokenizer, prompt formatting may be affected")
        return None


def prepare_prompts(items, tokenizer=None, enable_thinking=False):
    """
    Prepare prompts for API requests using the tokenizer's chat template if available.
    
    Dolly format fields:
    - instruction: user instruction text
    - context: optional context to prepend
    """
    prompts = []
    item_ids = []

    for item in items:
        try:
            # Get the item ID
            item_id = item["id"]

            instruction = str(item.get("instruction", "")).strip()
            context = str(item.get("context", "") or "").strip()

            if not instruction and not context:
                continue

            # Build user prompt similar to dataset_adapters
            if context:
                user_input = f"{context}\n\n{instruction}"
            else:
                user_input = instruction

            # Format as a chat message
            if tokenizer is not None:
                messages = [{"role": "user", "content": user_input}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                # Fallback to simple formatting if tokenizer is not available
                formatted_prompt = user_input
            
            prompts.append(formatted_prompt)
            item_ids.append(item_id)
            
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            continue

    return prompts, item_ids


def make_api_request(api_url, api_key, model_name, prompt, max_new_tokens=8192, temperature=0.0, top_p=1.0, top_k=-1, timeout=300):
    """Make a single API request to the model."""
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add authorization header if API key is provided
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Prepare request payload (OpenAI-compatible format)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    
    # Add top_k if specified
    if top_k != -1:
        payload["top_k"] = top_k
    
    # Make the API request
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the response content
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception("No response content in API result")
            
    except requests.exceptions.Timeout:
        raise Exception("Request timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing API response: {str(e)}")


def api_request_with_retry(api_url, api_key, model_name, prompt, max_retries=3, retry_delay=1.0, **kwargs):
    """Make an API request with retry logic."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return make_api_request(api_url, api_key, model_name, prompt, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"Request failed after {max_retries} attempts: {str(e)}")
    
    raise last_exception


def process_single_item(args, item_data, prompt, item_id):
    """Process a single item with API request."""
    try:
        # Make API request with retry
        response_content = api_request_with_retry(
            api_url=args.api_url,
            api_key=args.api_key,
            model_name=args.model_path,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        
        return process_single_response(response_content, item_id, item_data, args.is_print)
        
    except Exception as e:
        print(f"Error processing item {item_id}: {str(e)}")
        return None


def process_single_response(response_content, item_id, item_data, is_print=False):
    """Process a single model response."""
    instruction = str(item_data.get("instruction", "")).strip()
    context = str(item_data.get("context", "") or "").strip()

    if context:
        input_text = f"{context}\n\n{instruction}"
    else:
        input_text = instruction

    is_finished = True

    # Print full responses if requested
    if is_print:
        print(f"\n===== INPUT =====\n{input_text}\n")
        print(f"===== RESPONSE =====\n{response_content}\n")
        print(f"{'='*50}\n")

    if response_content:
        return InputItem(
            id=item_id,
            input_text=input_text,
            model_response=response_content,
            is_finished=is_finished,
        )
    
    return None


def concurrent_api_requests(args, items, prompts, item_ids, max_concurrent=10, start_time=None):
    """Process multiple items concurrently using API requests."""
    processed_items = []
    partial_buffer = []
    batch_start_time = time.time()
    
    print(f"开始并发处理 {len(items)} 个样本，并发数: {max_concurrent}")
    
    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all tasks
        future_to_item = {}
        for i, (item_data, prompt, item_id) in enumerate(zip(items, prompts, item_ids)):
            future = executor.submit(process_single_item, args, item_data, prompt, item_id)
            future_to_item[future] = (item_id, i)
        
        # Process completed requests with detailed progress
        completed_count = 0
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item), desc="处理API请求"):
            item_id, index = future_to_item[future]
            try:
                result = future.result()
                if result is not None:
                    processed_items.append(result)
                    partial_buffer.append(result)
                completed_count += 1
                
                # Show progress every 10% or every 100 items
                if completed_count % max(1, len(items) // 10) == 0 or completed_count % 100 == 0:
                    elapsed = time.time() - batch_start_time
                    avg_time_per_item = elapsed / completed_count
                    remaining_items = len(items) - completed_count
                    estimated_remaining = avg_time_per_item * remaining_items
                    
                    print(f"\n进度: {completed_count}/{len(items)} ({completed_count/len(items)*100:.1f}%)")
                    print(f"已用时: {elapsed/60:.1f}分钟, 平均每样本: {avg_time_per_item:.2f}秒")
                    print(f"预计剩余时间: {estimated_remaining/60:.1f}分钟")
                    if start_time:
                        total_elapsed = time.time() - start_time
                        print(f"总用时: {total_elapsed/60:.1f}分钟")
                    print("-" * 40)

                # Periodic partial save
                if args.save_every and len(partial_buffer) >= args.save_every:
                    save_partial_results_csv(partial_buffer, args.output_dir)
                    partial_buffer = []
                    
            except Exception as e:
                print(f"Error processing item {item_id}: {str(e)}")
    
    # Flush remaining partials
    if partial_buffer:
        save_partial_results_csv(partial_buffer, args.output_dir)
        partial_buffer = []

    batch_time = time.time() - batch_start_time
    print(f"\n批次完成! 用时: {batch_time/60:.1f}分钟")
    print(f"成功处理: {len(processed_items)}/{len(items)} 个样本")
    print(f"平均处理速度: {len(processed_items)/(batch_time/60):.1f} 样本/分钟")
    
    return processed_items


def save_args_to_json(args, output_dir):
    """Save command line arguments to a JSON file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert args to dictionary
    args_dict = vars(args)

    # Handle non-serializable objects
    if "year_range" in args_dict:
        args_dict["year_range"] = list(args_dict["year_range"])

    # Add timestamp
    args_dict["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to JSON
    json_path = os.path.join(output_dir, "run_args.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=2)

    print(f"Arguments saved to: {json_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    print(f"开始处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load Dolly dataset directly
    print(f"Loading Dolly dataset from {args.dataset_path}")
    
    # Determine how many items to load
    if args.num_items:
        items_to_load = args.num_items
    elif args.debug:
        items_to_load = 1
    else:
        items_to_load = None  # Load all
    
    if items_to_load:
        print(f"Loading first {items_to_load} items from Dolly dataset")
        dataset = load_dataset(args.dataset_path, split=args.split)
        dataset = dataset.select(range(items_to_load))
    else:
        print(f"Loading all items from Dolly dataset")
        dataset = load_dataset(args.dataset_path, split=args.split)
    
    print(f"Loaded Dolly dataset with {len(dataset)} items")
    
    # Convert to list and add IDs
    all_items = []
    for idx, item in enumerate(dataset):
        item_with_id = dict(item)
        item_with_id["id"] = f"dolly_{idx:06d}"
        all_items.append(item_with_id)

    # Get completed items if resuming
    completed_items = get_completed_items(args.output_dir) if args.resume else set()

    # Filter items based on arguments
    if args.item_ids:
        # Only process specified items
        items = [p for p in all_items if p["id"] in args.item_ids]
        print(f"Processing {len(items)} specified items")
    elif args.resume:
        # Only process items that haven't been completed
        items = [p for p in all_items if p["id"] not in completed_items]
        print(f"Resuming with {len(items)} remaining items")
    else:
        items = all_items

    # Handle debug mode and num_items
    if args.debug:
        print("Debug mode: processing only first item")

    if not items:
        print("No items to process!")
        return

    print(f"处理 {len(items)} 个样本从 Dolly 数据集")
    print(f"输出目录: {args.output_dir}")
    print(f"API URL: {args.api_url}")
    print(f"最大并发请求数: {args.max_concurrent_requests}")
    print(f"预计批次数: {(len(items) + args.max_concurrent_requests - 1) // args.max_concurrent_requests}")
    print("=" * 60)

    # Initialize tokenizer (optional, for prompt formatting)
    tokenizer = initialize_tokenizer(args.model_path)

    # Prepare prompts
    prompts, item_ids = prepare_prompts(items=items, tokenizer=tokenizer)
    print(f"Prepared {len(prompts)} prompts")

    # Process items using concurrent API requests
    processing_start_time = time.time()
    processed_items = concurrent_api_requests(
        args=args,
        items=items,
        prompts=prompts,
        item_ids=item_ids,
        max_concurrent=args.max_concurrent_requests,
        start_time=start_time,
    )

    # Save results
    save_results(processed_items, args.output_dir, append_mode=True)

    # Save arguments to JSON
    save_args_to_json(args, args.output_dir)

    # Final timing summary
    total_time = time.time() - start_time
    processing_time = time.time() - processing_start_time
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*60)
    print("🎉 处理完成！")
    print("="*60)
    print(f"开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time}")
    print(f"总用时: {total_time/60:.1f}分钟 ({total_time:.1f}秒)")
    print(f"纯处理用时: {processing_time/60:.1f}分钟 ({processing_time:.1f}秒)")
    print(f"成功处理样本数: {len(processed_items)}")
    print(f"处理成功率: {len(processed_items)/len(items)*100:.1f}%")
    if processed_items and total_time > 0:
        print(f"平均每样本用时: {total_time/len(processed_items):.2f}秒")
        print(f"整体处理速度: {len(processed_items)/(total_time/60):.1f} 样本/分钟")
        print(f"纯处理速度: {len(processed_items)/(processing_time/60):.1f} 样本/分钟")
    
    # 如果是处理完整数据集，给出预估
    if not args.debug and not args.num_items:
        print(f"\n📊 完整数据集处理统计:")
        print(f"如果这是完整数据集的一部分，基于当前速度:")
        if processed_items and total_time > 0:
            speed_per_minute = len(processed_items) / (total_time / 60)
            # OpenHermes-2.5 大约有 100万+ 样本
            full_dataset_size = 1000000
            estimated_full_time = full_dataset_size / speed_per_minute
            print(f"处理 100万样本 预计需要: {estimated_full_time/60:.1f}小时")
    
    print("="*60)


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, running...")
    main()