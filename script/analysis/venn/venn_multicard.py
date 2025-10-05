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
import seaborn as sns
from typing import Dict, Any, List, Tuple

from rosetta.model.projector import create_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.train.dataset_adapters import MMLUChatDataset
from rosetta.model.projector import create_projector, load_projector
from rosetta.model.aggregator import WeightedAggregator, load_aggregator

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

def load_rosetta_model(model_config, eval_config, device):
    """加载Rosetta模型"""
    # Create projectors/aggregators and load from checkpoint
    checkpoint_dir = eval_config["checkpoints_dir"]

    rosetta_config = model_config["rosetta_config"]
    slm_model_path = rosetta_config["base_model"]
    llm_model_path = rosetta_config["teacher_model"]

    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # Load projectors
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    projector_list = []
    for t in range(num_projectors):
        # Prefer JSON config if present to allow class/args reconstruction
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg)
        proj = proj.to(device)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)
    
    # Load aggregators
    num_aggregators = len([f for f in os.listdir(checkpoint_dir) if re.match(r"aggregator_\d+\.pt", f)])
    aggregator_list = []
    for t in range(num_aggregators):
        json_cfg = os.path.join(checkpoint_dir, f"aggregator_{t}.json")
        agg_path = os.path.join(checkpoint_dir, f"aggregator_{t}.pt")
        agg = load_aggregator(json_cfg)
        if os.path.exists(agg_path):
            sd = torch.load(agg_path, map_location="cpu")
            agg.load_state_dict(sd, strict=False)
        agg = agg.to(device)
        aggregator_list.append(agg)

    # shared_key_projection=build_shared_mlp(
    #     source_dim=teacher_dim,
    #     hidden_dim=projector_params["hidden_dim"],
    #     target_dim=base_dim,
    #     num_layers=projector_params["num_layers"],
    #     use_layer_norm=projector_params["use_layer_norm"],
    #     dropout=projector_params["dropout"],
    #     dtype=projector_params["dtype"]
    # )
    # shared_value_projection=build_shared_mlp(
    #     source_dim=teacher_dim,
    #     hidden_dim=projector_params["hidden_dim"],
    #     target_dim=base_dim,
    #     num_layers=projector_params["num_layers"],
    #     use_layer_norm=projector_params["use_layer_norm"],
    #     dropout=projector_params["dropout"],
    #     dtype=projector_params["dtype"]
    # )
    
    # 初始化Rosetta模型
    rosetta_model = RosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=projector_list,
        aggregator_list=aggregator_list,
    ).to(device).eval()

    # Load projector/aggregator mapping configs saved during training
    # Load saved mapping configs (preferred)
    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    agg_cfg_path = os.path.join(checkpoint_dir, "aggregator_config.json")
    # if os.path.exists(proj_cfg_path):
    rosetta_model.load_projector_config(proj_cfg_path)
    # if os.path.exists(agg_cfg_path):
    rosetta_model.load_aggregator_config(agg_cfg_path)

    return rosetta_model, slm_tokenizer

def format_example(example):
    """格式化评测样例"""
    TEMPLATE = """Accurately answer the following question:

{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D".
- Do not include any explanations, additional text, or punctuation besides the answer.

The correct answer is"""

    prompt = TEMPLATE.replace("{{question}}", example['question'])
    choices = ""
    for i, choice in enumerate(example['choices']):
        choices += f"{chr(65+i)}. {choice}\n"
    prompt = prompt.replace("{{choices}}", choices)
    return prompt

def generate_answer_with_logits(model, tokenizer, prompt, option_ids, device, model_type="qwen"):
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
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input_ids.shape[1]-4, 1).unsqueeze(0).to(device)
    responce_index = torch.tensor([-1, 0], dtype=torch.long).repeat(4, 1).unsqueeze(0).to(device)

    sampling_params = {
                    'do_sample': True,
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 20,
                    'min_p': 0.0,
                    'repetition_penalty': 1.1,
                    'max_new_tokens': 32768
    }

    if model_type == "rosetta":
        outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, kv_cache_index=[instruction_index, responce_index])
    else:
        outputs = model(input_ids, **sampling_params)
    
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

def evaluate_subjects(model, tokenizer, model_type, subjects, device, rank):
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
            
            for j in tqdm(range(len(test_data)), desc=f"GPU {rank}: Evaluating {subject}"):
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
                pred, _ = generate_answer_with_logits(model, tokenizer, prompt, option_ids, device, model_type=model_type)
                is_correct = pred in true_answer if true_answer else True

                if is_correct:
                    correct_ids.add(subject_idx * 100 + j)
                    
        except Exception as e:
            print(f"GPU {rank}: Error processing subject {subject}: {e}")
            continue
        
    return correct_ids

def worker_process(rank, world_size, model_type, model_name, subject_chunks, model_config=None, eval_config=None):
    """每个GPU上运行的工作进程"""
    try:
        # 设置分布式环境
        setup_distributed(rank, world_size)
        device = f'cuda:{rank}'
        
        print(f"GPU {rank}: Loading model {model_name}")
        
        # 加载模型
        if model_type == "rosetta":
            model, tokenizer = load_rosetta_model(model_config, eval_config, device)
        else:
            model, tokenizer = load_hf_model(model_name, device)
        
        # 评估分配给这个GPU的学科
        subjects_for_this_gpu = subject_chunks[rank]
        print(f"GPU {rank}: Evaluating subjects: {subjects_for_this_gpu}")
        
        correct_ids = evaluate_subjects(model, tokenizer, model_type, subjects_for_this_gpu, device, rank)
        
        # 保存结果到文件
        result_file = f"results_{model_type}_gpu_{rank}.json"
        with open(result_file, 'w') as f:
            json.dump(list(correct_ids), f)
        
        print(f"GPU {rank}: Finished evaluation, saved {len(correct_ids)} correct answers to {result_file}")
        
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

def collect_results(model_type, world_size):
    """收集所有GPU的结果"""
    all_correct_ids = set()
    
    for rank in range(world_size):
        result_file = f"results_{model_type}_gpu_{rank}.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                correct_ids = json.load(f)
                all_correct_ids.update(correct_ids)
            # 删除临时文件
            os.remove(result_file)
        else:
            print(f"Warning: Result file {result_file} not found")
    
    return all_correct_ids

def run_multicard_evaluation(model_configs, world_size, model_config=None, eval_config=None):
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
    for i, chunk in enumerate(subject_chunks):
        print(f"GPU {i}: {len(chunk)} subjects - {chunk[:3]}..." if len(chunk) > 3 else f"GPU {i}: {len(chunk)} subjects - {chunk}")
    
    correct_ids = {}
    
    # 为每个模型运行评估
    for model_type, model_name in model_configs.items():
        print(f"\n=== Evaluating {model_type} ({model_name}) ===")
        
        # 启动多进程
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker_process, args=(rank, world_size, model_type, model_name, subject_chunks, model_config, eval_config))
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 收集结果
        correct_ids[model_type] = collect_results(model_type, world_size)
        print(f"Total correct answers for {model_type}: {len(correct_ids[model_type])}")
    
    return correct_ids

def calculate_subject_accuracy(correct_ids_dict: Dict[str, set], subjects: List[str]) -> Dict[str, Dict[str, float]]:
    """
    计算每个subject的准确率
    
    Args:
        correct_ids_dict: 模型名称到正确答案ID集合的映射
        subjects: 所有subject列表
        
    Returns:
        Dict[subject_name, Dict[model_name, accuracy]]
    """
    subject_accuracy = {}
    
    # 获取每个subject的题目总数
    subject_question_counts = {}
    for subject_idx, subject in enumerate(subjects):
        try:
            dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject)
            test_data = dataset["test"]
            # 过滤掉错误类型的题目
            valid_count = 0
            for data in test_data:
                error_type = data['error_type']
                if error_type not in ['no_correct_answer', 'expert']:
                    valid_count += 1
            subject_question_counts[subject] = valid_count
        except Exception as e:
            print(f"Warning: Could not load subject {subject}: {e}")
            subject_question_counts[subject] = 100  # 默认值
    
    # 计算每个subject的准确率
    for subject_idx, subject in enumerate(subjects):
        subject_accuracy[subject] = {}
        total_questions = subject_question_counts[subject]
        
        for model_name, correct_ids in correct_ids_dict.items():
            # 计算该subject的正确答案数量
            subject_correct_count = 0
            for correct_id in correct_ids:
                # ID格式：subject_idx * 100 + question_idx
                if correct_id // 100 == subject_idx:
                    subject_correct_count += 1
            
            accuracy = subject_correct_count / total_questions if total_questions > 0 else 0
            subject_accuracy[subject][model_name] = accuracy
    
    return subject_accuracy

def plot_subject_accuracy_comparison(subject_accuracy: Dict[str, Dict[str, float]], 
                                   output_dir: str, model_configs: Dict[str, str]):
    """
    绘制每个subject的准确率对比柱状图，每个subject单独显示
    
    Args:
        subject_accuracy: 每个subject的准确率数据
        output_dir: 输出目录
        model_configs: 模型配置
    """
    subjects = sorted(list(subject_accuracy.keys()))
    model_names = list(next(iter(subject_accuracy.values())).keys())
    
    # 设置颜色
    colors = {'rosetta': '#FF6B6B', 'slm': '#4ECDC4', 'llm': '#45B7D1'}
    model_display_names = {
        'rosetta': 'Rosetta',
        'slm': 'SLM',
        'llm': 'LLM'
    }
    
    # 计算需要的子图数量和布局
    n_subjects = len(subjects)
    
    # 每页显示15个subject，如果超过则分页
    subjects_per_page = 15
    n_pages = (n_subjects + subjects_per_page - 1) // subjects_per_page
    
    for page in range(n_pages):
        start_idx = page * subjects_per_page
        end_idx = min(start_idx + subjects_per_page, n_subjects)
        page_subjects = subjects[start_idx:end_idx]
        
        # 计算当前页的子图布局 (3行5列)
        n_rows = 3
        n_cols = 5
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 18))
        
        # 如果只有一个子图，axes不是数组
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, subject in enumerate(page_subjects):
            ax = axes[idx]
            
            # 准备数据
            x_pos = np.arange(len(model_names))
            bar_width = 0.6
            
            # 为每个模型绘制柱状图
            accuracies = []
            colors_list = []
            labels = []
            
            for model_name in model_names:
                if model_name in model_display_names:
                    accuracy = subject_accuracy[subject].get(model_name, 0)
                    accuracies.append(accuracy)
                    colors_list.append(colors.get(model_name, f'C{len(accuracies)-1}'))
                    labels.append(model_display_names[model_name])
            
            bars = ax.bar(x_pos, accuracies, bar_width, 
                         color=colors_list, alpha=0.8)
            
            # 在柱子上显示数值
            for bar, acc in zip(bars, accuracies):
                if acc > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 设置标签和标题
            ax.set_ylabel('Accuracy')
            ax.set_title(subject.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=0)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.0)
            
            # 添加水平线显示平均值
            overall_avg = np.mean(accuracies) if accuracies else 0
            ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # 隐藏多余的子图
        for i in range(len(page_subjects), len(axes)):
            axes[i].axis('off')
        
        # 添加总标题
        model_names_str = " vs ".join([model_display_names.get(name, name) for name in model_names])
        if n_pages > 1:
            fig.suptitle(f'Subject Accuracy Comparison: {model_names_str} (Page {page+1}/{n_pages})', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'Subject Accuracy Comparison: {model_names_str}', 
                        fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 为suptitle留空间
        
        # 生成文件名
        if n_pages > 1:
            output_file = os.path.join(output_dir, f"subject_accuracy_comparison_{model_names_str}_page_{page+1}.png")
        else:
            output_file = os.path.join(output_dir, f"subject_accuracy_comparison_{model_names_str}.png")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Subject accuracy comparison plot (page {page+1}) saved to {output_file}")
    
    # 保存数值数据到JSON
    model_names_str = "_vs_".join([model_display_names.get(name, name) for name in model_names])
    json_output_file = os.path.join(output_dir, f"subject_accuracy_data_{model_names_str}.json")
    with open(json_output_file, 'w') as f:
        json.dump(subject_accuracy, f, indent=4)
    print(f"Subject accuracy data saved to {json_output_file}")

def plot_overall_accuracy_summary(subject_accuracy: Dict[str, Dict[str, float]], 
                                output_dir: str, model_configs: Dict[str, str]):
    """
    绘制总体准确率汇总图
    """
    model_names = list(next(iter(subject_accuracy.values())).keys())
    # 固定图片尺寸与分辨率
    FIGSIZE = (12, 9)
    DPI = 300
    model_display_names = {
        'rosetta': 'C2C',
        'slm': 'Receiver', 
        'llm': 'Sharer'
    }
    colors = {'rosetta': '#FF6B6B', 'slm': '#4ECDC4', 'llm': '#45B7D1'}
    
    # 计算每个模型的平均准确率
    model_avg_accuracy = {}
    for model_name in model_names:
        accuracies = [subject_accuracy[subject][model_name] for subject in subject_accuracy]
        model_avg_accuracy[model_name] = np.mean(accuracies)
    
    # 绘制平均准确率柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    
    # 子图1：总体平均准确率
    models = list(model_avg_accuracy.keys())
    avg_accs = list(model_avg_accuracy.values())
    
    bars = ax1.bar([model_display_names.get(m, m) for m in models], avg_accs,
                   color=[colors.get(m, f'C{i}') for i, m in enumerate(models)],
                   alpha=0.8)
    
    # 添加数值标签
    for bar, acc in zip(bars, avg_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Overall Average Accuracy Across All Subjects')
    ax1.set_ylim(0, max(avg_accs) * 1.2)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：每个模型在所有subject上的准确率分布
    model_data = []
    model_labels = []
    for model_name in model_names:
        accuracies = [subject_accuracy[subject][model_name] for subject in subject_accuracy]
        model_data.append(accuracies)
        model_labels.append(model_display_names.get(model_name, model_name))
    
    box_plot = ax2.boxplot(model_data, labels=model_labels, patch_artist=True)
    
    # 设置箱线图颜色
    for patch, model_name in zip(box_plot['boxes'], model_names):
        patch.set_facecolor(colors.get(model_name, 'lightblue'))
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Distribution Across All Subjects')
    ax2.grid(True, alpha=0.3)
    
    # 优化布局，尽量充满画布
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, wspace=0.25)
    # 保存图片（固定输出尺寸，避免根据内容裁剪导致大小不一致）
    model_names_str = "_vs_".join([model_display_names.get(name, name) for name in model_names])
    output_file = os.path.join(output_dir, f"overall_accuracy_summary_{model_names_str}.png")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=DPI, bbox_inches=None)
    plt.close()
    print(f"Overall accuracy summary plot saved to {output_file}")
    
    return model_avg_accuracy

def plot_venn(correct_ids_dict, venn_out_file, json_out_file, model_configs, all_question_ids=None):
    """绘制Venn图"""
    rosetta_set = correct_ids_dict.get("rosetta", set())
    slm_set = correct_ids_dict.get("slm", set())
    llm_set = correct_ids_dict.get("llm", set())

    # 计算各个区域
    venn_regions = {
        "rosetta_only": list(rosetta_set - slm_set - llm_set),
        "slm_only": list(slm_set - rosetta_set - llm_set),
        "llm_only": list(llm_set - rosetta_set - slm_set),
        "rosetta_slm": list((rosetta_set & slm_set) - llm_set),
        "rosetta_llm": list((rosetta_set & llm_set) - slm_set),
        "slm_llm": list((slm_set & llm_set) - rosetta_set),
        "all_three": list(rosetta_set & slm_set & llm_set),
    }

    # 计算 none 区域（可选，需提供所有题目 ID）
    if all_question_ids is not None:
        all_correct = rosetta_set | slm_set | llm_set
        venn_regions["none"] = list(set(all_question_ids) - all_correct)

    # 保存 JSON 文件
    with open(json_out_file, "w") as f:
        json.dump(venn_regions, f, indent=4)
    print(f"Venn region question IDs saved to {json_out_file}")

    # 创建模型信息用于标题和文件名
    model_info = []
    if "rosetta" in model_configs:
        model_info.append("C2C")
    if "slm" in model_configs:
        slm_name = model_configs["slm"].split("/")[-1] if model_configs["slm"] else "SLM"
        model_info.append(f"Receiver({slm_name})")
    if "llm" in model_configs:
        llm_name = model_configs["llm"].split("/")[-1] if model_configs["llm"] else "LLM"
        model_info.append(f"Sharer({llm_name})")
    
    model_title = " vs ".join(model_info)

    # 绘图（固定图片尺寸与分辨率，与总体汇总图保持一致）
    FIGSIZE = (12, 9)
    DPI = 300
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    ax = fig.add_subplot(111)
    v = venn3([rosetta_set, slm_set, llm_set], set_labels=("C2C", "Receiver", "Sharer"))
    # 放大字体与图形，减少空白
    if v is not None:
        # 集合标签字体
        if hasattr(v, 'set_labels') and v.set_labels is not None:
            for txt in v.set_labels:
                if txt is not None:
                    txt.set_fontsize(16)
        # 区域数字字体
        for sid in ("100","010","001","110","101","011","111"):
            lbl = v.get_label_by_id(sid)
            if lbl is not None:
                lbl.set_fontsize(14)
        # 圈线更清晰
        for sid in ("100","010","001","110","101","011","111"):
            patch = v.get_patch_by_id(sid)
            if patch is not None:
                patch.set_linewidth(1.5)
    # 调整坐标轴占比，尽量充满画布（保留少量边距给标题）
    ax.set_position([0.06, 0.06, 0.88, 0.86])
    plt.title(f"Correct Answer Overlap: {model_title}", fontsize=16, pad=8)
    # 固定输出尺寸，避免根据内容裁剪导致大小不一致
    plt.savefig(venn_out_file, dpi=DPI, bbox_inches=None)
    plt.close()
    print(f"Venn diagram saved to {venn_out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMLU Evaluation with Parquet Files')
    parser.add_argument("--config", type=str, default="script/analysis/venn/venn_config.json", help="Path to JSON config file")
    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    # Extract configuration sections
    model_config = cfg["model"]
    output_config = cfg["output"]
    eval_config = cfg["eval"]
    models_config = cfg.get("models", {})
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 从配置文件中获取GPU设置
    gpu_ids = eval_config.get("gpu_ids", [0, 1, 2, 3])
    world_size = len(gpu_ids)
    
    # 从配置文件中配置要评估的模型
    model_configs = {}
    
    # 检查是否要评估Rosetta模型
    if model_config.get("model_name") == "Rosetta":
        model_configs["rosetta"] = models_config["rosetta"]  # Rosetta不需要model_name
    
    # 从models配置中添加其他模型
    if models_config.get("slm") and models_config["slm"] != "None":
        model_configs["slm"] = models_config["slm"]
    
    if models_config.get("llm") and models_config["llm"] != "None":
        model_configs["llm"] = models_config["llm"]
    
    # 如果没有指定任何模型，默认评估SLM和LLM
    if not model_configs:
        model_configs = {
            "slm": "Qwen/Qwen2.5-0.5B",
            "llm": "Qwen/Qwen2.5-Math-1.5B-Instruct"
        }
    
    print(f"Will evaluate models: {list(model_configs.keys())}")
    print(f"Using {world_size} GPUs: {gpu_ids}")
    
    # 创建输出目录
    output_dir = output_config["output_dir"]
    result_dir = output_config["result_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 运行多卡评估
    correct_ids = run_multicard_evaluation(model_configs, world_size, model_config, eval_config)
    
    # 计算每个subject的准确率
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
    
    # print("\n=== Calculating subject-wise accuracy ===")
    # subject_accuracy = calculate_subject_accuracy(correct_ids, subjects)
    
    # # 绘制subject准确率对比图
    # print("=== Plotting subject accuracy comparison ===")
    # plot_subject_accuracy_comparison(subject_accuracy, output_dir, model_configs)
    
    # # 绘制总体准确率汇总图
    # print("=== Plotting overall accuracy summary ===")
    # model_avg_accuracy = plot_overall_accuracy_summary(subject_accuracy, output_dir, model_configs)
    
    # # 打印总体准确率
    # print("\n=== Overall Results ===")
    # for model_name, avg_acc in model_avg_accuracy.items():
    #     model_display_name = {'rosetta': 'Rosetta', 'slm': 'SLM', 'llm': 'LLM'}.get(model_name, model_name)
    #     print(f"{model_display_name}: {avg_acc:.4f}")
    
    # 生成包含模型信息的文件名
    model_names = []
    if "rosetta" in model_configs:
        model_names.append(models_config["rosetta"])
    if "slm" in model_configs:
        slm_name = model_configs["slm"].split("/")[-1] if model_configs["slm"] else "SLM"
        model_names.append(slm_name)
    if "llm" in model_configs:
        llm_name = model_configs["llm"].split("/")[-1] if model_configs["llm"] else "LLM"
        model_names.append(llm_name)
    
    file_suffix = "_vs_".join(model_names) if model_names else "multicard"
    
    # 生成输出文件路径
    venn_out_file = os.path.join(result_dir, f"venn_graph/venn_diagram_{file_suffix}.png")
    json_out_file = os.path.join(result_dir, f"venn_json/venn_regions_{file_suffix}.json")

    # 绘制Venn图
    plot_venn(correct_ids, venn_out_file, json_out_file, model_configs, all_question_ids=list(range(5700)))
    
    print("Multi-card evaluation completed!")
    print(f"Results saved to: {result_dir}")
    print(f"Venn diagram: {venn_out_file}")
    print(f"Region data: {json_out_file}")
    print(f"Subject accuracy comparison plots saved in: {output_dir}")
