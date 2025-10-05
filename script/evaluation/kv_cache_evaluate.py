import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 1. Setup Argument Parser
# NOTE: For this specific test, we'll override some args in the main block.
parser = argparse.ArgumentParser(description='MMLU KV-Cache Optimization Test')
parser.add_argument('--model_name', type=str, default='Qwen3-0.6B', choices=['Qwen3-0.6B', 'Qwen3-1.7B'], help='The name of the model to evaluate.')
parser.add_argument('--gpu_id', type=int, default=0, help='The GPU ID to use for the evaluation.')
parser.add_argument('--batch_size', type=int, default=2, help='The batch size for inference. Keep it small for this test.')
# Add a specific argument for our test subject
parser.add_argument('--use_cot', action='store_true', help='Whether to use Chain-of-Thought reasoning.')
parser.add_argument('--subject', type=str, default='high_school_psychology', help='MMLU subject to run the test on.')

args = parser.parse_args()

# 2. Environment and Path Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
torch.cuda.set_device(args.gpu_id)
device = torch.device(f"cuda:{args.gpu_id}")
print(f"Using device: {device} | GPU ID: {args.gpu_id}")

BASE_DIR = Path("/mnt/public")
MODEL_DIR = BASE_DIR / "public_models"
DATA_DIR = BASE_DIR / "public_datasets" / "mmlu"
OUTPUT_DIR = BASE_DIR / "yanjichao" / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "Qwen3-0.6B": MODEL_DIR / "Qwen3-0.6B",
    "Qwen3-1.7B": MODEL_DIR / "Qwen3-1.7B"
}

print(f"Loading model: {args.model_name}")
model_path = MODEL_PATHS[args.model_name]

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    padding_side='left'
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.bfloat16,
    device_map=f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "auto"
).eval()
print(f"Model loaded successfully! Parameters: {model.num_parameters():,}")

# MODIFIED: Split prompt generation into two parts for easier token length calculation.
def build_prompts_for_kv_test(question, choices):
    """
    Builds prompts for the KV-Cache test.
    Returns the few-shot example part and the question part separately.
    """
    example_prompt = (
        "Follow the example to answer the multiple-choice question in JSON format.\n\n"
        "--- Example ---\n"
        "Question: Which of the following is a primary color?\n"
        "Options:\n"
        "A. Green\n"
        "B. Orange\n"
        "C. Blue\n"
        "D. Purple\n"
        "Answer (in JSON format):\n"
        '{"answer": "C"}\n\n'
    )
    
    current_question_prompt = (
        "--- Now, answer this question ---\n"
        f"Question: {question}\n"
        "Options:\n"
    )
    for i, choice in enumerate(choices):
        current_question_prompt += f"{chr(65+i)}. {choice}\n"
    
    current_question_prompt += "\nAnswer (in JSON format):"
    
    return example_prompt, current_question_prompt

def load_mmlu_dataset(subject):
    """Loads and formats the MMLU dataset for a given subject."""
    try:
        dataset = load_dataset(
            "parquet",
            data_files={
                "test": str(DATA_DIR / subject / "test-00000-of-00001.parquet"),
                "dev": str(DATA_DIR / subject / "dev-00000-of-00001.parquet")
            }
        )
        
        def format_answer(example):
            if isinstance(example['answer'], int) and 0 <= example['answer'] < 4:
                example['answer'] = chr(65 + example['answer'])
            elif isinstance(example['answer'], str):
                example['answer'] = example['answer'].strip().upper()
                if example['answer']:
                    example['answer'] = example['answer'][0]
            return example
        
        return dataset.map(format_answer)
    except Exception as e:
        print(f"Error loading dataset for {subject} from Hugging Face Hub. Make sure you are online. Error: {str(e)}")
        return None

def extract_answer_from_json(text: str) -> str:
    """Extracts the answer (A, B, C, or D) from the model's output."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not match:
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index > start_index:
            json_str = text[start_index : end_index + 1]
        else:
            json_str = None
    else:
        json_str = match.group(1)
    
    if json_str:
        try:
            data = json.loads(json_str)
            answer = data.get('answer')
            if isinstance(answer, str):
                answer = answer.strip().upper()
                if answer in ['A', 'B', 'C', 'D']:
                    return answer
        except (json.JSONDecodeError, AttributeError):
            pass

    for char in reversed(text):
        if char in {'A', 'B', 'C', 'D'}:
            return char
            
    return 'X' # Return 'X' for failure

# 4. NEW CORE TEST FUNCTION (CORRECTED)

def evaluate_kv_cache_optimization(subject):
    """
    Performs the KV-Cache optimization test on a given MMLU subject.
    Compares a standard prefill with a few-shot optimized prefill.
    """
    is_thinking_mode = args.use_cot
    # BUG FIX: `repetition_penalty` must be >= 1.0 to prevent repetition. 1.1 is a safe value.
    sampling_params = dict(
        do_sample=True,
        temperature=0.6 if is_thinking_mode else 0.7,
        top_p=0.95 if is_thinking_mode else 0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.5,
    )

    print(f"\n{'='*60}")
    print(f"Starting KV-Cache Optimization Test on Subject: {subject}")
    print(f"{'='*60}")

    # The loader returns a DatasetDict, e.g., {'test': ..., 'dev': ...}
    dataset = load_mmlu_dataset(subject)
    if dataset is None:
        return None, None

    # CORRECTED: First, select the 'test' split, which is a Dataset object.
    test_data = dataset['test']
    
    # Use a smaller subset for a quick test. Now .select() is called on a Dataset object.
   
    total = len(test_data)
    
    correct_optimized = 0
    correct_standard = 0
    results_log = []

    # Generation config
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    qwen_eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if qwen_eot_id is not None and qwen_eot_id not in stop_token_ids:
        stop_token_ids.append(qwen_eot_id)
        
    if not stop_token_ids:
        raise ValueError("EOS token ID not found in tokenizer.")

    # CORRECTED: Iterate over the correct variable `test_dataset`
    for i in tqdm(range(0, total, args.batch_size), desc=f"Testing {subject}"):
        # CORRECTED: Select batches from `test_dataset`
        batch = test_data.select(range(i, min(i + args.batch_size, total)))
        # --- Prepare prompts for the batch ---
        example_prompts, question_prompts = [], []
        for ex in batch:
            ex_prompt, q_prompt = build_prompts_for_kv_test(ex['question'], ex['choices'])
            example_prompts.append(ex_prompt)
            question_prompts.append(q_prompt)

        full_prompts = [ex + q for ex, q in zip(example_prompts, question_prompts)]
        
        true_answers = [ex['answer'] for ex in batch]
        ##fp,ta
        # --- Tokenization ---
        example_tokens = tokenizer(example_prompts[0], return_tensors="pt", add_special_tokens=False)
        example_tokens_len = example_tokens.input_ids.shape[1]
        question_tokens = tokenizer(question_prompts[0], return_tensors="pt", add_special_tokens=False)
        question_tokens_len = question_tokens.input_ids.shape[1]

        full_inputs = tokenizer(full_prompts, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
        question_only_inputs = tokenizer(question_prompts, padding=True, return_tensors="pt", truncation=True, max_length=400).to(device)

        with torch.no_grad():
            # --- STRATEGY 1: Optimized KV-Cache ---
            outputs_full = model(
                input_ids=full_inputs.input_ids,
                attention_mask=full_inputs.attention_mask,
                use_cache=True
            )
            full_past_key_values = outputs_full.past_key_values
            #print(outputs_full)
            optimized_past_key_values = []
            for layer_past in full_past_key_values:
                key_states, value_states = layer_past
                sliced_key = key_states[:, :, example_tokens_len:, :]
                sliced_value = value_states[:, :, example_tokens_len:, :]
                optimized_past_key_values.append((sliced_key, sliced_value))
            optimized_past_key_values = tuple(optimized_past_key_values)
            #print(full_inputs)
            optimized_attention_mask = full_inputs.attention_mask[:, example_tokens_len:]
            last_token_ids = full_inputs.input_ids[:,-1:]

            cache_position = torch.arange(
                question_tokens_len, 
                question_tokens_len + last_token_ids.shape[1], 
                device=device
            )
            #print(optimized_past_key_values)
            outputs_optimized = model.generate(
                input_ids=last_token_ids,
                past_key_values=optimized_past_key_values,
                attention_mask=optimized_attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
                    
                cache_position=cache_position,  # 明确传递缓存位置
                **sampling_params
            )
            

            # --- STRATEGY 2: Standard KV-Cache (Baseline) ---
            outputs_question_only = model(
                input_ids=question_only_inputs.input_ids,
                attention_mask=question_only_inputs.attention_mask,
                use_cache=True
            )
            standard_past_key_values = outputs_question_only.past_key_values

            last_token_ids_standard = question_only_inputs.input_ids[:, -1:]
            standard_cache_position = torch.arange(
                question_only_inputs.input_ids.shape[1] - 1,  # 已处理的token数（排除当前token）
                question_only_inputs.input_ids.shape[1],      # 当前token位置
                device=device
            )
            outputs_standard = model.generate(
                input_ids=last_token_ids_standard,
                past_key_values=standard_past_key_values,
                attention_mask=question_only_inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
                    
                cache_position=standard_cache_position ,  # 明确传递缓存位置
                **sampling_params
            )

        # --- Decode and Compare ---
        decoded_optimized = tokenizer.batch_decode(outputs_optimized, skip_special_tokens=True)
        decoded_standard = tokenizer.batch_decode(outputs_standard, skip_special_tokens=True)
        
        preds_optimized = [extract_answer_from_json(text) for text in decoded_optimized]
        preds_standard = [extract_answer_from_json(text) for text in decoded_standard]
        
        for j, true_ans in enumerate(true_answers):
            if preds_optimized[j] == chr(65+true_ans):
                correct_optimized += 1
            if preds_standard[j] == chr(65+true_ans):
                correct_standard += 1
            
            results_log.append({
                "question": batch[j]['question'],
                "true_answer": true_ans,
                "optimized_output": decoded_optimized[j],
                "optimized_prediction": preds_optimized[j],
                "standard_output": decoded_standard[j],
                "standard_prediction": preds_standard[j],
            })
        print(results_log)
    # --- Report Results ---
    accuracy_optimized = (correct_optimized / total) * 100 if total > 0 else 0
    accuracy_standard = (correct_standard / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*60}")
    print("Test Complete. Final Results:")
    print(f"Subject: {subject} ({total} samples)")
    print(f"  - Accuracy with Optimized KV-Cache: {accuracy_optimized:.2f}% ({correct_optimized}/{total})")
    print(f"  - Accuracy with Standard KV-Cache:  {accuracy_standard:.2f}% ({correct_standard}/{total})")
    print(f"{'='*60}")

    if accuracy_optimized > accuracy_standard:
        print("Conclusion: The KV-Cache optimization yielded a BETTER result.")
    elif accuracy_optimized < accuracy_standard:
        print("Conclusion: The KV-Cache optimization yielded a WORSE result.")
    else:
        print("Conclusion: Both methods yielded the SAME result.")
        
    output_file = OUTPUT_DIR / f"kv_cache_test_{args.model_name}_{subject}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_summary": {
                "subject": subject,
                "total_samples": total,
                "accuracy_optimized": accuracy_optimized,
                "accuracy_standard": accuracy_standard,
            },
            "detailed_results": results_log
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed log saved to: {output_file}")
# 5. Main Execution Block
if __name__ == "__main__":
    # For this targeted test, we will run on a single subject.
    # This avoids the complexity of iterating through all subjects and focuses on the core task.
    print("--- Running AI Test: KV-Cache Optimization ---")
    evaluate_kv_cache_optimization(subject=args.subject)
    print("--- Test Finished ---")


