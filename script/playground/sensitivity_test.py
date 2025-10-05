import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_scheduler
from tqdm import tqdm
import os
import sys
import json
import argparse
import shutil
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datasets import DatasetDict, load_dataset
from transformers.cache_utils import DynamicCache
import re

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import create_projector

BASE_DIR = Path("/mnt/public")
DATA_DIR1 = BASE_DIR / "public_datasets" / "mmlu"
DATA_DIR2 = BASE_DIR / "public_datasets" / "mmlu-redux-2.0"
OUTPUT_DIR = BASE_DIR / "hanling" / "rosetta_evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_models(model_config: Dict[str, Any], num_layers_to_map: List[int] = [0, ], device: str = "cuda", dtype: torch.dtype = torch.bfloat16, args = None):
    """Setup base and teacher models with projectors"""
    
    # Load tokenizer (use base model tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=dtype,
        device_map=device
    )
    
    # Load teacher model  
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_config["teacher_model"],
        torch_dtype=dtype,
        device_map=device
    )
    
    # Create projector from config
    projector_config = model_config["projector"]
    projector_params = projector_config["params"].copy()
    projector_params["dtype"] = dtype
    projector = create_projector(
        projector_config["type"],
        mean = args.mean,
        std = args.std,
        **projector_params
    )

    # Setup RosettaModel
    model_list = [base_model, teacher_model]
    rosetta_model = RosettaModel(
        model_list=model_list,
        base_model_idx=0,  # Base model is index 0
        projector_list=[projector]
    ).to(device)
    
    # Configure projector mappings (map teacher layers to base layers)
    # For simplicity, map first few layers
    
    for layer_idx in num_layers_to_map:
        rosetta_model.set_projector_config(
            source_model_idx=1,  # Teacher model
            source_model_layer_idx=layer_idx,
            target_model_idx=0,  # Base model
            target_model_layer_idx=layer_idx,
            projector_idx=0
        )
    
    return rosetta_model, tokenizer


def get_subjects():
    subjects = []
    excluded = {'all', 'auxiliary_train', '.git'}
    
    for item in DATA_DIR2.iterdir():
        if item.is_dir() and item.name not in excluded:
            test_path = DATA_DIR2 / item.name / "data-00000-of-00001.arrow"
            
            if test_path.exists():
                subjects.append(item.name)
    return sorted(subjects)


def parse_answer(answer_str):
    """
    仅从字符串中提取数字0/1/2/3并转换为A/B/C/D
    参数:
        answer_str: 可能包含数字的字符串（如 "0", "0,1", "either 0 or 1"）
    返回:
        排序后的唯一选项字母列表（如 ['A'], ['A','B']）
    """
    if not isinstance(answer_str, str):
        return []
    
    # 只提取0-3的数字字符
    valid_digits = [c for c in answer_str if c in {'0','1','2','3'}]
    
    # 转换为字母并去重排序
    return sorted(list({
        chr(65 + int(d))  # 0->A, 1->B, 2->C, 3->D
        for d in valid_digits
    }))


def format_example(example, include_answer=True):
    """优化后的提示词格式，最大化模型输出选项字母的概率"""
    prompt = "Question: " + example['question'] + "\n"
    prompt += "Options:\n"
    for i, choice in enumerate(example['choices']):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    # 关键指令 - 强化输出要求

    prompt += "Answer: "
    if include_answer:
        prompt += f"{chr(65+example['answer'])}\n\n"  # 训练示例中提供正确答案
    return prompt


def gen_prompt(dev_data, subject, k=-1):
    """生成包含k个示例的提示"""
    if(k==0):return ""
    prompt = f"The following are single choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"
    k = len(dev_data) if k == -1 else min(k, len(dev_data))
    for i in range(k):
        prompt += format_example(dev_data[i])
    return prompt


# MODIFIED: Added a few-shot example to guide the model.
def build_prompt_for_direct_answer(question, choices):
    """
    Builds a prompt to get a direct answer in JSON format using a few-shot example.
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
    
    return example_prompt + current_question_prompt

def build_messages_for_cot(question, choices):
    prompt_content = (
        "You are a strict multiple-choice assistant.\n"
        "Follow these steps:\n"
        "1. In the <think> tag, reason step-by-step in English to solve the problem concisely.\n"
        "2. Immediately after </think>, give your final answer as a single uppercase letter (A/B/C/D), and nothing else.\n"
        "Do NOT include any explanation outside <think>. Your answer must be in the format: A\n\n"
        f"Question: {question}\nOptions:"
    )
    for i, choice in enumerate(choices):
        prompt_content += f"\n{chr(65+i)}. {choice}"

    return [{"role": "user", "content": prompt_content}]


def load_dataset_files(subject):
    """加载指定学科的数据集，支持parquet和arrow格式"""
    try:
        dev_path = DATA_DIR1 / subject / "dev-00000-of-00001.parquet"
        test_path = DATA_DIR2 / subject / "data-00000-of-00001.arrow"
        
        # 检查文件是否存在
        if not dev_path.exists():
            print(f"Dev file not found: {dev_path}")
            return None
        if not test_path.exists():
            print(f"Test file not found: {test_path}")
            return None

        # 分别加载dev和test
        dev_dataset = load_dataset("parquet", data_files=str(dev_path))['train']
        test_dataset = load_dataset("arrow", data_files=str(test_path))['train']
        
        # 重命名split
        dataset = DatasetDict({
            "dev": dev_dataset,
            "test": test_dataset
        })
        return dataset
    except Exception as e:
        print(f"Error loading {subject}: {str(e)}")
        return None


def extract_answer_from_json(text: str, args) -> str:
    """
    Extracts the answer from a model output expected to contain a JSON object.
    Includes a fallback to find a single letter if JSON parsing fails.
    """
    if args.use_cot:
        # For CoT, look for the content after </think>
        think_end_match = re.search(r'</think>\s*([A-D])', text, re.IGNORECASE)
        if think_end_match:
            return think_end_match.group(1).upper()
        
        # If no single letter immediately after </think>, look for the last valid letter
        # in the entire text, assuming the model might just output it somewhere.
        # This is a less strict fallback for CoT.
        for char in reversed(text):
            if char in {'A', 'B', 'C', 'D'}:
                return char
    # Attempt to find a JSON object within ```json ... ``` code blocks
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not match:
        # If no markdown block, find the first '{' and the last '}'
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index > start_index:
            match = text[start_index : end_index + 1]
        else:
            match = None
    
    if match:
        try:
            json_str = match.group(1) if hasattr(match, 'group') else match
            data = json.loads(json_str)
            answer = data.get('answer')
            if isinstance(answer, str):
                answer = answer.strip().upper()
                if answer in ['A', 'B', 'C', 'D']:
                    return answer
        except (json.JSONDecodeError, AttributeError):
            pass # Fall through to the next method if JSON parsing fails

    # Fallback: If JSON fails, search for the last capital letter (A-D) in the string.
    for char in reversed(text):
        if char in {'A', 'B', 'C', 'D'}:
            return char
            
    return 'X'


def evaluate_subject(subject, model, tokenizer, device, args):
    """Evaluates a single MMLU subject."""
    
    is_thinking_mode = args.use_cot
    # BUG FIX: `repetition_penalty` must be >= 1.0 to prevent repetition. 1.1 is a safe value.
    sampling_params = dict(
        do_sample=True,
        temperature=0.6 if is_thinking_mode else 0.7,
        top_p=0.95 if is_thinking_mode else 0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.1,
    )

    print(f"\n{'='*50}")
    print(f"Evaluating subject: {subject}")
    
    dataset = load_dataset_files(subject)
    if dataset is None:
        return None, None
    
    test_data = dataset["test"]
    total = len(test_data)
    correct = 0
    results = []
    
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    qwen_eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if qwen_eot_id is not None and qwen_eot_id not in stop_token_ids:
        stop_token_ids.append(qwen_eot_id)
        
    if not stop_token_ids:
        raise ValueError("EOS token ID not found in tokenizer.")

    for i in tqdm(range(0, total, args.batch_size), desc=f"Evaluating {subject}"):
        batch = test_data.select(range(i, min(i + args.batch_size, total)))
        answers = [ex['answer'] for ex in batch]
        
        if args.use_cot:
            messages_batch = [build_messages_for_cot(ex['question'], ex['choices']) for ex in batch]
            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(device)
            print(prompts)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_token_ids,
                    **sampling_params
                )
        else: # Direct answer mode
            prompts = [build_prompt_for_direct_answer(ex['question'], ex['choices']) for ex in batch]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
            print(prompts)
            with torch.no_grad():
                # BUG FIX: `max_new_tokens` was too small for JSON. Increased from 5 to 30.
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_token_ids,
                    **sampling_params
                )
        
        # We decode the full output for parsing, not just the newly generated part
        full_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = [extract_answer_from_json(text, args=args) for text in full_outputs]

        for j, (pred, ans) in enumerate(zip(predictions, answers)):
            # BUG FIX: `ans` is already a character. No need for chr().
            is_correct = 1 if pred == chr(65+ans) else 0
            correct += is_correct
            results.append({
                "question": batch[j]["question"],
                "choices": batch[j]["choices"],
                "prediction": pred,
                "answer": chr(65+ans), # BUG FIX: Use `ans` directly.
                "is_correct": is_correct,
                "full_output": full_outputs[j]
            })
    
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"Completed! Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if results:
        print("\nResult Examples:")
        for i, r in enumerate(results[:min(3, len(results))]):
            print(f"Question {i+1}: {r['question']}")
            print(f"Model Output: {r['full_output']}")
            print(f"Extracted Answer: {r['prediction']}")
            print(f"Correct Answer: {r['answer']}")
            print(f"Result: {'✓' if r['is_correct'] else '✗'}\n")
    
    return accuracy, results


@torch.no_grad()
def evaluate(subject, model, tokenizer, device, args):
    dataset = load_dataset_files(subject)
    if dataset is None:
        return None, None, None

    dev_data = dataset["dev"]
    test_data = dataset["test"]
    option_ids = []
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        option_ids.append(ids[0] if ids else tokenizer.eos_token_id)
    
    cors = []
    all_probs = []
    total_count = 0
    skip_count = 0
    for i in tqdm(range(len(test_data)), desc=f"Evaluating {subject} ({args.method})"):
        # 准备测试问题
        error_type = test_data[i]['error_type']
        if error_type in ['no_correct_answer', 'expert']:
            skip_count += 1
            continue
                
        try:
            # 统一答案解析逻辑
            if error_type == 'ok':
                # 单选直接使用answer字段的数字
                answer_num = test_data[i]['answer']
                true_answer = [chr(65 + answer_num)] if answer_num is not None else None
            else:
                # 其他所有类型都解析correct_answer字符串
                correct_answer = test_data[i]['correct_answer']
                true_answer = parse_answer(correct_answer) if correct_answer is not None else None
            
            # 获取当前问题的格式化字符串
            question_and_options_text = format_example(test_data[i], include_answer=False)

            if args.method == 'zero_shot':
                # 方法1: 直接不使用few-shot
                input_text = question_and_options_text
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[0, -1]
            elif args.method == 'few_shot_add_noise':
                k = args.ntrain
                # 1. 生成 few-shot 示例部分
                few_shot_text = gen_prompt(dev_data, subject, k)
                ##few-shot trainpro   qao   promp end
                # 动态调整样本数以适应上下文长度 (few_shot_text + question_and_options_text)
                prompt = few_shot_text + question_and_options_text
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
                # 动态减少样本数以适应上下文长度
                
                """
                few_shot_text = gen_prompt(dev_data, subject, k)
                prompt = few_shot_text + question_and_options_text
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)"""
                sampling_params = {
                'do_sample': True,
                'temperature': 0.6 if args.use_cot else 0.7,
                'top_p': 0.95 if args.use_cot else 0.8,
                'top_k': 20,
                'min_p': 0.0,
                'repetition_penalty': 1.1,
                'max_new_tokens': 32768
                }
                # 将 few-shot 示例和问题部分组合，进行第一次前向传播
                # 这里假设问题部分在 `format_example` 中包含了 "Question: ... Options: ... Answer: " 前缀
                # 我们需要将问题和选项分开
                # 首先找到 "Options: " 在 `question_and_options_text` 中的位置
                answer_prefix_start_index = question_and_options_text.rfind("Options:")
                question_part_text = question_and_options_text[:answer_prefix_start_index]
                answer_suffix_text = question_and_options_text[answer_prefix_start_index:] # 包含 "Options: "

                # 第一次前向传播：few-shot + 问题部分 (不带 "Options: ")
                # 目标是获取到问题部分的 KV cache
                first_pass_input_text = few_shot_text + question_part_text
                first_pass_input_ids = tokenizer.encode(first_pass_input_text, return_tensors="pt").to(device)
                ##print(first_pass_input_text,'&',answer_suffix_text)
                # 计算 few_shot 部分和 question 部分的长度
                few_shot_len = len(tokenizer.encode(few_shot_text, add_special_tokens=False))
                question_len = len(tokenizer.encode(question_part_text, add_special_tokens=False))
                full_seq_len = first_pass_input_ids.shape[1]
                #print(few_shot_len,len(few_shot_text))
                ##with torch.no_grad():
                    # 第一次前向传播
                position_ids = torch.arange(0, full_seq_len, device=device).unsqueeze(0)
                    # 返回 past_key_values 以便后续使用
                outputs_first_pass = model(input_ids=first_pass_input_ids,**sampling_params, use_cache=True)
                past_key_values = outputs_first_pass.past_key_values
                

                full_first_pass_seq_len = first_pass_input_ids.shape[1]
                
                # 2. 将选项送入模型进行第二次前向传播
                # 输入是 Answer: 后面的部分
                second_pass_input_ids = tokenizer.encode(answer_suffix_text, return_tensors="pt").to(device)
                position_ids = torch.arange(
                    full_seq_len,  # 从原始序列结束位置开始
                    full_seq_len + second_pass_input_ids.shape[1],  # 到新序列结束
                    device=device
                ).unsqueeze(0)
                # 第二次前向传播，使用第一次保留的 KV cache, 只用base model
                ##with torch.no_grad():
                start_position = full_seq_len
                seq_length = past_key_values.get_usable_length(second_pass_input_ids.shape[1])
                
                outputs_second_pass = model.model_list[0](
                        input_ids = second_pass_input_ids,
                        **sampling_params,
                        past_key_values=past_key_values, # 使用修改后的 KV cache
                        position_ids=position_ids,
                        use_cache=True # 确保继续生成 KV cache 以防后续需要 (虽然这里是最后一步)
                    )
                #print(outputs_second_pass)
                logits = outputs_second_pass.logits[0, -1] # 获取最后一个token的logits

            else:  # few_shot_retain
                # 方法3: 完整的few-shot
                k = args.ntrain
                train_prompt = gen_prompt(dev_data, subject, k)
                input_text = train_prompt + question_and_options_text
                prompt = train_prompt + question_and_options_text
                
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)                
                # 动态调整样本数以适应上下文长度
                while input_ids.shape[1] > 32758:
                    k = max(1, k - 1)
                    train_prompt = gen_prompt(dev_data, subject, k)
                    prompt = train_prompt + question_and_options_text
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)              
                    # 对于few_shot_retain，直接执行一次前向传播
                sampling_params = {
                    'do_sample': True,
                    'temperature': 0.6 if args.use_cot else 0.7,
                    'top_p': 0.95 if args.use_cot else 0.8,
                    'top_k': 20,
                    'min_p': 0.0,
                    'repetition_penalty': 1.1,
                    'max_new_tokens': 32768
                }

                outputs = model.model_list[0](input_ids, **sampling_params) 
                logits = outputs.logits[0, -1]
            # 计算选项概率 (对 few_shot_delete 和 few_shot_retain 都是一样的)
            option_logits = torch.tensor([
                logits[option_ids[0]].item(),
                logits[option_ids[1]].item(),
                logits[option_ids[2]].item(),
                logits[option_ids[3]].item()
            ])
            
            probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
            pred = chr(65 + np.argmax(probs))
            
            is_correct = pred in true_answer if true_answer else True  # 无合法答案自动判对

            cors.append(is_correct)
            all_probs.append(probs)
            total_count += 1
            ##print(pred,answer)
        except Exception as e:
            print(f"Error processing sample {i} in {subject}: {str(e)}")
            skip_count += 1
            continue

    if total_count > 0:
        acc = np.mean(cors)
        print(f"{subject} accuracy: {acc:.3f} (evaluated on {total_count} samples, skipped {skip_count})")
    else:
        acc = 0
        print(f"{subject} skipped all samples ({skip_count} skipped)")
        
    return np.array(cors) if cors else None, acc, np.array(all_probs) if all_probs else None


def main():
    parser = argparse.ArgumentParser(description="Train RosettaModel from a JSON config")
    parser.add_argument('--ntrain', type=int, default=5,
                   help='Number of few-shot examples')
    parser.add_argument('--mean', type=float, default=0)
    parser.add_argument('--std', type=float, default=1)
    parser.add_argument("--config", type=str, default="recipe/sensitivity_config.json", help="Path to JSON config file")
    parser.add_argument("--layer_add_noise", type=int, default=0, help="Layer-wise noise addition")
    parser.add_argument('--use_cot', action='store_true', help='Whether to use Chain-of-Thought reasoning.')
    parser.add_argument('--method', type=str, required=True,
                   choices=['zero_shot', 'few_shot_add_noise', 'few_shot_retain'],
                   help='Generation method: zero_shot, few_shot_add_noise, or few_shot_retain')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    # Set seed for reproducibility
    set_seed(seed = 42)

    # Extract configuration sections
    model_config = cfg["model"]
    eval_config = cfg["evaluation"]
    device = eval_config.get("device", "cuda")

    rosetta_model, tokenizer = setup_models(model_config, [args.layer_add_noise,], device, torch.bfloat16, args)

    # 获取所有学科
    subjects = []
    excluded = {'all', 'auxiliary_train', '.git'}
    
    # 修改点1: 支持检测arrow格式的测试文件
    for item in DATA_DIR2.iterdir():
        if item.is_dir() and item.name not in excluded:
            # 检查可能的测试文件格式
            test_file_parquet = item / "test-00000-of-00001.parquet"
            test_file_arrow = item / "data-00000-of-00001.arrow"
            
            # 任意一种格式存在即可
            if test_file_parquet.exists() or test_file_arrow.exists():
                subjects.append(item.name)
    
    print(f"Found {len(subjects)} subjects to evaluate")
    
    # 结果存储 - 添加方法信息到路径
    if args.method == 'few_shot_add_noise':
        result_dir = OUTPUT_DIR / args.method / f"rosetta_add_noise_{args.layer_add_noise}_{args.ntrain}"
    else:
        result_dir = OUTPUT_DIR / args.method / f"rosetta_{args.method}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    # 学科评估
    for subject in subjects:
        cors, acc, probs = evaluate(subject, rosetta_model, tokenizer, device, args)
        if cors is None:
            continue
            
            # 保存结果
        result_df = pd.DataFrame({
            "correct": cors,
            "prob_A": probs[:, 0] if probs is not None else [],
            "prob_B": probs[:, 1] if probs is not None else [],
            "prob_C": probs[:, 2] if probs is not None else [],
            "prob_D": probs[:, 3] if probs is not None else []
        })
        result_df.to_parquet(result_dir / f"{subject}.parquet")
        
        # 更新分类结果
        all_cors.append(cors)
        for subcat in subcategories.get(subject, []):
            subcat_cors[subcat].append(cors)
            for cat, subcat_list in categories.items():
                if subcat in subcat_list:
                    cat_cors[cat].append(cors)
    
    # 输出分类结果
    print("\n===== Category Results =====")
    for cat in sorted(cat_cors.keys()):
        if cat_cors[cat]:
            acc = np.mean(np.concatenate(cat_cors[cat]))
            print(f"{cat:<20}: {acc:.3f}")

    # 保存汇总结果
    summary = {
        "model": f"rosetta_noise_{args.layer_add_noise}",
        "method": args.method,
        "overall_accuracy": np.mean(np.concatenate(all_cors))if all_cors else 0,
        "categories": {
            cat: np.mean(np.concatenate(cors)) 
            for cat, cors in cat_cors.items() 
        },
        "subcategories": {
            subcat: np.mean(np.concatenate(cors))if cors else 0
            for subcat, cors in subcat_cors.items() 
        }
    }

    if args.method == 'few_shot_add_noise':
        with open(OUTPUT_DIR / f"rosetta_add_noise_{args.layer_add_noise}_{args.method}_{args.ntrain}_{args.mean}_{args.std}_{args.ntrain}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    else:
        with open(OUTPUT_DIR / f"base_{args.layer_add_noise}_{args.method}_{args.ntrain}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {result_dir}")

if __name__ == "__main__":
    main()
    