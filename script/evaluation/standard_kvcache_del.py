import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset,DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None 
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
# ... (保留原有的subcategories和categories定义不变) ...

# 1. 参数配置 - 添加method参数
parser = argparse.ArgumentParser(description='MMLU Evaluation with Parquet Files')
parser.add_argument('--model_name', type=str, required=True,
                   choices=['Qwen3-0.6B', 'Qwen3-1.7B'],
                   help='Model to evaluate')
parser.add_argument('--use_cot', action='store_true', help='Whether to use Chain-of-Thought reasoning.')
parser.add_argument('--gpu_id', type=int, default=5,
                   help='GPU ID to use')
parser.add_argument('--ntrain', type=int, default=5,
                   help='Number of few-shot examples')
parser.add_argument('--max_length', type=int, default=32768,
                   help='Maximum context length')
parser.add_argument('--batch_size', type=int, default=1,
                   help='Inference batch size')
parser.add_argument('--subjects', nargs='+', default=[],
                   help='Specific subjects to evaluate')
parser.add_argument('--method', type=str, required=True,
                   choices=['zero_shot', 'few_shot_delete', 'few_shot_retain'],
                   help='Generation method: zero_shot, few_shot_delete, or few_shot_retain')
args = parser.parse_args()
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Requested GPU ID: {args.gpu_id}")
# 2. 环境设置
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
torch.cuda.set_device(args.gpu_id)
device = torch.device(f"cuda:{args.gpu_id}")

# 3. 路径配置
BASE_DIR = Path("/mnt/public")
MODEL_DIR = BASE_DIR / "public_models"
DATA_DIR1 = BASE_DIR / "public_datasets" / "mmlu"
DATA_DIR2 = BASE_DIR / "public_datasets" / "mmlu-redux-2.0"
OUTPUT_DIR = BASE_DIR / "yanjichao" / "mmlu_results_kv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "Qwen3-0.6B": MODEL_DIR / "Qwen3-0.6B",
    "Qwen3-1.7B": MODEL_DIR / "Qwen3-1.7B"
}

# 4. 加载模型
print(f"Loading {args.model_name}...")
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATHS[args.model_name]),
    trust_remote_code=True,
    padding_side='left'
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATHS[args.model_name]),
    torch_dtype=torch.bfloat16,
    device_map={"": device}
).eval()

# 5. 数据加载函数
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
# 6. 提示工程
def format_example(example, include_answer=True):
    """优化后的提示词格式，最大化模型输出选项字母的概率"""
    prompt = "Question: " + example['question'] + "\n"
    prompt += "Options:\n"
    for i, choice in enumerate(example['choices']):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    # 关键指令 - 强化输出要求

    prompt += "Answer: "
    if include_answer:
        prompt += "Answer: "
        prompt += f"{chr(65+example['answer'])}\n\n"  # 训练示例中提供正确答案
    return prompt
def gen_prompt(dev_data, subject, k=-1):
    """生成包含k个示例的提示"""
    
    prompt = f"The following are single choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"
    if(k==0):return prompt
    k = len(dev_data) if k == -1 else min(k, len(dev_data))
    for i in range(k):
        prompt += format_example(dev_data[i])
    return prompt
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

# 7. 核心评估函数 - 修改以支持三种生成方式
@torch.no_grad()
def evaluate(subject):
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
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1]
            elif args.method == 'few_shot_delete':
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
                # 首先找到 "Answer: " 在 `question_and_options_text` 中的位置
                answer_prefix_start_index = question_and_options_text.rfind("Options:")
                question_part_text = question_and_options_text[:answer_prefix_start_index]
                answer_suffix_text = question_and_options_text[answer_prefix_start_index:] # 包含 "Answer: "

                # 第一次前向传播：few-shot + 问题部分 (不带 "Answer: ")
                # 目标是获取到问题部分的 KV cache
                first_pass_input_text = few_shot_text + question_part_text+answer_suffix_text
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
                outputs_first_pass = model(first_pass_input_ids,**sampling_params, use_cache=True)
                past_key_values = outputs_first_pass.past_key_values
                

                full_first_pass_seq_len = first_pass_input_ids.shape[1]
                
                #few_shot_len=1
                new_past_key_values = DynamicCache()
                for layer_idx in range(len(past_key_values.key_cache)):
                    key = past_key_values.key_cache[layer_idx]
                    value = past_key_values.value_cache[layer_idx]
                    
                    # 保留第一个元素，删除其后的few_shot_len个元素
                    new_key = torch.cat([
                        key[:, :, :15, :],           # 保留第一个元素
                        key[:, :, few_shot_len:, :]  # 保留few_shot_len之后的元素
                    ], dim=2)
                    
                    new_value = torch.cat([
                        value[:, :, :15, :],           # 保留第一个元素
                        value[:, :, few_shot_len:, :]  # 保留few_shot_len之后的元素
                    ], dim=2)
                    
                    #new_key = key[:, :, few_shot_len:, :]
                    #new_value = value[:, :, few_shot_len:, :]
                    new_past_key_values.key_cache.append(new_key)
                    new_past_key_values.value_cache.append(new_value)
                # 2. 将选项送入模型进行第二次前向传播
                # 输入是 Answer: 后面的部分
                second_pass_input_ids = tokenizer.encode("Answer:", return_tensors="pt").to(device)
                position_ids = torch.arange(
                    full_seq_len,  # 从原始序列结束位置开始
                    full_seq_len + second_pass_input_ids.shape[1],  # 到新序列结束
                    device=device
                ).unsqueeze(0)
                # 第二次前向传播，使用第一次保留的 KV cache
                ##with torch.no_grad():
                start_position = full_seq_len
                seq_length = new_past_key_values.get_usable_length(second_pass_input_ids.shape[1])
                
                outputs_second_pass = model(
                        second_pass_input_ids,
                        **sampling_params,
                        
                        past_key_values=new_past_key_values, # 使用修改后的 KV cache
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
                while input_ids.shape[1] > args.max_length - 10:
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

                outputs = model(input_ids, **sampling_params) 
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
# 8. 主函数 - 修改输出路径以包含方法信息
def main():
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
    result_dir = OUTPUT_DIR / args.model_name / args.method
    result_dir.mkdir(parents=True, exist_ok=True)
    
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    # 学科评估
    for subject in subjects:
        cors, acc, probs = evaluate(subject)
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
        "model": args.model_name,
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
    
    with open(OUTPUT_DIR / f"{args.model_name}_{args.method}_{args.ntrain}_summary_wchoices.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {result_dir}")

if __name__ == "__main__":
    main()