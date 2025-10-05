import argparse
import re
import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def load_hf_model(model_name, device):
    """加载HuggingFace模型到指定设备"""
    model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval().to(device)
    return model, tokenizer

def format_example(example):
    """格式化评测样例"""
    TEMPLATE = """
        Accurately answer the following question:

        {{question.question}}

        Choices:
        {{question.multi_choices}}

        Instructions:
        - Carefully read the question and all options.
        - Select the single most correct answer.
        - Respond ONLY with the letter ({{question.letters}}) corresponding to the correct answer.
        - Do not include any explanations, additional text, or punctuation in your response.

        Your answer:
    """
    prompt = TEMPLATE.replace("{{question.question}}", example['question'])
    choices = ""
    for i, choice in enumerate(example['choices']):
        choices += f"{chr(65+i)}. {choice}\n"
    prompt = prompt.replace("{{question.multi_choices}}", choices)
    prompt = prompt.replace("{{question.letters}}", "A, B, C, D")
    return prompt

def generate_answer_with_logits(model, tokenizer, prompt, option_ids, device):
    """使用logits方法生成答案"""
    messages = [{
        "role": "user",
        "content": prompt
    }]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    text += "The correct answer is"
    input_ids = tokenizer(text, return_tensors="pt").to(device)['input_ids']
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits[0, -1]
    option_logits = torch.tensor([
        logits[option_ids[0]].item(),
        logits[option_ids[1]].item(),
        logits[option_ids[2]].item(),
        logits[option_ids[3]].item()
    ])
    
    probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
    pred = chr(65 + np.argmax(probs))
    return pred, probs

def evaluate_subjects(model, tokenizer, model_name, subjects, device, rank):
    """评估指定的学科"""
    correct_ids = set()
    
    for i, subject in enumerate(subjects):
        try:
            dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject)
            test_data = dataset["test"]

            option_ids = []
            for letter in ["A", "B", "C", "D"]:
                ids = tokenizer.encode(" " + letter, add_special_tokens=False)
                option_ids.append(ids[0] if ids else tokenizer.eos_token_id)
            
            # 为了保持一致的ID，需要知道这个subject在总列表中的索引
            all_subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
                           'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
                           'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
                           'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
                           'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
                           'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
                           'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
                           'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 
                           'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 
                           'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 
                           'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 
                           'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 
                           'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
            
            subject_idx = all_subjects.index(subject)
            
            for j in tqdm(range(len(test_data)), desc=f"GPU {rank}: Evaluating {model_name} on {subject}"):
                error_type = test_data[j]['error_type']
                if error_type in ['no_correct_answer', 'expert']:
                    continue
                    
                if error_type == 'wrong_groundtruth':
                    if test_data[j]['correct_answer'] is not None:
                        if test_data[j]['correct_answer']>= '0' and test_data[j]['correct_answer'] <= '3':
                            answer_num = int(test_data[j]['correct_answer'])
                        else:
                            answer_num = ord(test_data[j]['correct_answer']) - ord('A')
                    else:
                        answer_num = int(test_data[j]['answer'])
                    true_answer = [chr(65 + answer_num)] if answer_num is not None else None
                else:
                    answer_num = int(test_data[j]['answer'])
                    true_answer = [chr(65 + answer_num)] if answer_num is not None else None

                prompt = format_example(test_data[j])
                pred, _ = generate_answer_with_logits(model, tokenizer, prompt, option_ids, device)
                is_correct = pred in true_answer if true_answer else True

                if is_correct:
                    correct_ids.add(subject_idx * 100 + j)
                    
        except Exception as e:
            print(f"GPU {rank}: Error processing subject {subject} for model {model_name}: {e}")
            continue
        
    return correct_ids

def worker_process(rank, world_size, model_configs, subject_chunks):
    """每个GPU上运行的工作进程"""
    try:
        # 设置分布式环境
        setup_distributed(rank, world_size)
        device = f'cuda:{rank}'
        
        results = {}
        
        # 为这个GPU上的每个模型进行评估
        for model_key, model_name in model_configs.items():
            print(f"GPU {rank}: Loading model {model_name}")
            
            # 加载模型
            model, tokenizer = load_hf_model(model_name, device)
            
            # 评估分配给这个GPU的学科
            subjects_for_this_gpu = subject_chunks[rank]
            print(f"GPU {rank}: Evaluating {model_name} on subjects: {subjects_for_this_gpu}")
            
            correct_ids = evaluate_subjects(model, tokenizer, model_name, subjects_for_this_gpu, device, rank)
            results[model_key] = correct_ids
            
            # 清理模型以释放显存
            del model, tokenizer
            torch.cuda.empty_cache()
        
        # 保存结果到文件
        result_file = f"results_gpu_{rank}.json"
        serializable_results = {k: list(v) for k, v in results.items()}
        with open(result_file, 'w') as f:
            json.dump(serializable_results, f)
        
        print(f"GPU {rank}: Finished evaluation, saved results to {result_file}")
        
    except Exception as e:
        print(f"GPU {rank}: Error in worker process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理分布式环境
        cleanup_distributed()

def split_subjects_among_gpus(subjects, world_size):
    """将学科平均分配给各个GPU"""
    chunk_size = len(subjects) // world_size
    remainder = len(subjects) % world_size
    
    chunks = []
    start = 0
    for i in range(world_size):
        # 如果有余数，前remainder个GPU多分配1个学科
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        chunks.append(subjects[start:end])
        start = end
    
    return chunks

def collect_results(world_size):
    """收集所有GPU的结果"""
    all_results = {}
    
    for rank in range(world_size):
        result_file = f"results_gpu_{rank}.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                gpu_results = json.load(f)
                for model_key, correct_ids in gpu_results.items():
                    if model_key not in all_results:
                        all_results[model_key] = set()
                    all_results[model_key].update(correct_ids)
            # 删除临时文件
            os.remove(result_file)
        else:
            print(f"Warning: Result file {result_file} not found")
    
    return all_results

def run_multicard_evaluation(model_configs, world_size):
    """运行多卡评估"""
    # 所有学科
    subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
               'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
               'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
               'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
               'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
               'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
               'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
               'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 
               'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 
               'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 
               'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 
               'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 
               'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    
    # 将学科分配给各个GPU
    subject_chunks = split_subjects_among_gpus(subjects, world_size)
    
    print(f"Total subjects: {len(subjects)}")
    print(f"Models to evaluate: {list(model_configs.keys())}")
    for i, chunk in enumerate(subject_chunks):
        print(f"GPU {i}: {len(chunk)} subjects - {chunk[:3]}..." if len(chunk) > 3 else f"GPU {i}: {len(chunk)} subjects - {chunk}")
    
    # 启动多进程
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker_process, args=(rank, world_size, model_configs, subject_chunks))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 收集结果
    all_results = collect_results(world_size)
    
    for model_key in model_configs.keys():
        print(f"Total correct answers for {model_key} ({model_configs[model_key]}): {len(all_results.get(model_key, set()))}")
    
    return all_results

def plot_venn(correct_ids_dict, venn_out_file, json_out_file, model_configs, all_question_ids=None):
    """绘制Venn图"""
    # 获取三个模型的结果集
    model_keys = list(correct_ids_dict.keys())
    if len(model_keys) != 3:
        raise ValueError(f"Expected 3 models for Venn diagram, got {len(model_keys)}")
    
    set1 = correct_ids_dict[model_keys[0]]
    set2 = correct_ids_dict[model_keys[1]]
    set3 = correct_ids_dict[model_keys[2]]
    
    # 获取模型标签
    labels = []
    for key in model_keys:
        model_name = model_configs[key]
        # 提取模型名称的最后部分作为标签
        clean_name = model_name.split("/")[-1] if "/" in model_name else model_name
        labels.append(clean_name)

    # 计算各个区域
    venn_regions = {
        f"{labels[0]}_only": list(set1 - set2 - set3),
        f"{labels[1]}_only": list(set2 - set1 - set3),
        f"{labels[2]}_only": list(set3 - set1 - set2),
        f"{labels[0]}_{labels[1]}": list((set1 & set2) - set3),
        f"{labels[0]}_{labels[2]}": list((set1 & set3) - set2),
        f"{labels[1]}_{labels[2]}": list((set2 & set3) - set1),
        "all_three": list(set1 & set2 & set3),
    }

    # 计算 none 区域（可选，需提供所有题目 ID）
    if all_question_ids is not None:
        all_correct = set1 | set2 | set3
        venn_regions["none"] = list(set(all_question_ids) - all_correct)

    # 保存 JSON 文件
    with open(json_out_file, "w") as f:
        json.dump(venn_regions, f, indent=4)
    print(f"Venn region question IDs saved to {json_out_file}")

    # 绘图
    plt.figure(figsize=(12, 10))
    venn_diagram = venn3([set1, set2, set3], set_labels=labels)
    
    # 设置标题
    title = f"Correct Answer Overlap: {' vs '.join(labels)}"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 添加统计信息
    stats_text = f"Total questions: {len(all_question_ids) if all_question_ids else 'Unknown'}\n"
    stats_text += f"{labels[0]}: {len(set1)} correct\n"
    stats_text += f"{labels[1]}: {len(set2)} correct\n"
    stats_text += f"{labels[2]}: {len(set3)} correct\n"
    stats_text += f"All three: {len(set1 & set2 & set3)} correct"
    
    plt.text(1.2, 0.8, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.savefig(venn_out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Venn diagram saved to {venn_out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-card Venn diagram evaluation for 3 HF models')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--model1', type=str, required=True, help='First HuggingFace model name')
    parser.add_argument('--model2', type=str, required=True, help='Second HuggingFace model name')
    parser.add_argument('--model3', type=str, required=True, help='Third HuggingFace model name')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 配置要评估的三个模型
    model_configs = {
        "model1": args.model1,
        "model2": args.model2,
        "model3": args.model3
    }
    
    print(f"Will evaluate models:")
    for key, model in model_configs.items():
        print(f"  {key}: {model}")
    print(f"Using {args.world_size} GPUs")
    
    # 运行多卡评估
    correct_ids = run_multicard_evaluation(model_configs, args.world_size)
    
    # 生成包含模型信息的文件名
    model_names = []
    for key in ["model1", "model2", "model3"]:
        model_name = model_configs[key].split("/")[-1] if "/" in model_configs[key] else model_configs[key]
        # 清理模型名称，移除特殊字符
        clean_name = re.sub(r'[^\w\-_.]', '_', model_name)
        model_names.append(clean_name)
    
    file_suffix = "_vs_".join(model_names)
    
    # 生成输出文件路径
    venn_out_file = os.path.join(args.output_dir, f"venn_diagram_{file_suffix}.png")
    json_out_file = os.path.join(args.output_dir, f"venn_regions_{file_suffix}.json")
    
    # 绘制Venn图
    plot_venn(correct_ids, venn_out_file, json_out_file, model_configs, all_question_ids=list(range(5700)))
    
    print("Multi-card evaluation completed!")
    print(f"Results saved to:")
    print(f"  Venn diagram: {venn_out_file}")
    print(f"  Region data: {json_out_file}")
