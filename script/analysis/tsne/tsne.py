import argparse
import re
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from rosetta.model.projector import create_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.train.dataset_adapters import MMLUChatDataset

def load_qwen_model(model_name):
    model_path = "Qwen/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval().to(DEVICE)
    return model, tokenizer

def load_rosetta_model(checkpoint_dir):
    """Load Rosetta model using the same approach as evaluate.py"""
    from rosetta.model.projector import load_projector
    import re
    
    slm_model_path = "Qwen/Qwen3-0.6B"
    llm_model_path = "Qwen/Qwen3-4B"

    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(slm_model_path)
    if slm_tokenizer.pad_token is None:
        slm_tokenizer.pad_token = slm_tokenizer.eos_token
    
    # Load models
    slm_model = AutoModelForCausalLM.from_pretrained(
        slm_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE}
    ).eval()
    
    # Load projectors
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    projector_list = []
    for t in range(num_projectors):
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg)
        proj = proj.to(DEVICE)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=DEVICE)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)
    
    # Initialize Rosetta model
    rosetta_model = RosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=projector_list,
    ).to(DEVICE).eval()

    # Load projector mapping configs
    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    rosetta_model.load_projector_config(proj_cfg_path)

    return rosetta_model, slm_tokenizer

def extract_k_cache(model, tokenizer, dataset, layer_idx, num_samples=10):
    all_values = []
    # 选择指定数量的样本
    subset = [dataset[i] for i in range(0, min(num_samples, len(dataset)))]
    for i, sample in enumerate(tqdm(subset, desc=f"Layer {layer_idx}")):
        # Use normal MMLU question without extra prompt
        instruction = tokenizer.apply_chat_template(sample[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input = tokenizer(instruction, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input['input_ids'].shape[1], 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if isinstance(model, RosettaModel):
                output = model(**input, kv_cache_index=[instruction_index], use_cache=True)
            else:
                output = model(**input, use_cache=True)

        # (batch, num_heads, seq_len, head_dim)
        k_value = output.past_key_values[layer_idx][0].squeeze(0).float().cpu()  # (num_heads, seq_len, head_dim)
        
        # For each token, flatten across heads and head_dim
        # k_value: (num_heads, seq_len, head_dim) -> (seq_len, num_heads * head_dim)
        k_value_flat = k_value.permute(1, 0, 2).reshape(k_value.shape[1], -1)  # (seq_len, num_heads * head_dim)
        
        # Add all tokens for this sample
        all_values.append(k_value_flat.numpy())

    return all_values

def extract_v_cache(model, tokenizer, dataset, layer_idx, num_samples=20):
    all_values = []
    # 选择指定数量的样本
    subset = [dataset[i] for i in range(0, min(num_samples, len(dataset)))]
    for i, sample in enumerate(tqdm(subset, desc=f"Layer {layer_idx}")):
        # Use normal MMLU question without extra prompt
        instruction = tokenizer.apply_chat_template(sample[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input = tokenizer(instruction, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input['input_ids'].shape[1], 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if isinstance(model, RosettaModel):
                output = model(**input, kv_cache_index=[instruction_index], use_cache=True)
            else:
                output = model(**input, use_cache=True)

        # (batch, num_heads, seq_len, head_dim)
        v_value = output.past_key_values[layer_idx][1].squeeze(0).float().cpu()  # (num_heads, seq_len, head_dim)
        
        # For each token, flatten across heads and head_dim
        # v_value: (num_heads, seq_len, head_dim) -> (seq_len, num_heads * head_dim)
        v_value_flat = v_value.permute(1, 0, 2).reshape(v_value.shape[1], -1)  # (seq_len, num_heads * head_dim)
        
        # Add all tokens for this sample
        all_values.append(v_value_flat.numpy())

    return all_values

def plot_tsne_per_token(all_embeddings, label, model_names, layer_idx, output_path, show_correspondence=True):
    """绘制每个token的t-SNE图"""
    tsne = TSNE(n_components=2, perplexity=30, random_state=2)
    
    # Flatten all embeddings from all models and samples
    X = np.concatenate([np.concatenate(emb, axis=0) for emb in all_embeddings], axis=0)
    tsne_result = tsne.fit_transform(X)

    # 构建颜色标签 - 为每个 token 分配模型名称
    color_labels = []
    token_indices = []  # 记录每个token在原始数据中的位置信息
    current_idx = 0
    
    for i, emb in enumerate(all_embeddings):
        # emb is a list of arrays, each array has shape (seq_len, num_heads * head_dim)
        for sample_idx, arr in enumerate(emb):
            total_tokens = arr.shape[0]
            color_labels.extend([model_names[i]] * total_tokens)
            # 记录每个token的模型索引和样本索引
            for token_idx in range(total_tokens):
                token_indices.append({
                    'model_idx': i,
                    'model_name': model_names[i],
                    'sample_idx': sample_idx,
                    'token_idx': token_idx,
                    'global_idx': current_idx + token_idx
                })
            current_idx += total_tokens

    # 创建基础散点图
    plt.figure(figsize=(15, 10))
    
    # 绘制所有点
    for model_name in set(color_labels):
        indices = [j for j, lbl in enumerate(color_labels) if lbl == model_name]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                   label=model_name, s=8, alpha=0.7)

    # 先保存无对应关系的版本
    plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Token")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_token.png"), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: tsne_layer_{layer_idx}_{label}_per_token.png")
    
    # 如果启用对应关系，添加连线
    if show_correspondence:
        plot_correspondence_lines(tsne_result, token_indices, model_names, label)
        
        plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Token with Correspondence")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_token_with_correspondence.png"), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: tsne_layer_{layer_idx}_{label}_per_token_with_correspondence.png")


def plot_tsne_per_sequence(all_embeddings, label, model_names, layer_idx, output_path, show_correspondence=True):
    """绘制每个序列的t-SNE图"""
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeddings[0])-1), random_state=42)
    
    # 为每个序列计算平均嵌入
    sequence_embeddings = []
    sequence_indices = []  # 记录每个序列的模型和样本信息
    
    for i, emb in enumerate(all_embeddings):
        # emb is a list of arrays, each array has shape (seq_len, num_heads * head_dim)
        for sample_idx, arr in enumerate(emb):
            # 计算序列的平均嵌入
            seq_avg = np.mean(arr, axis=0)  # (num_heads * head_dim,)
            sequence_embeddings.append(seq_avg)
            sequence_indices.append({
                'model_idx': i,
                'model_name': model_names[i],
                'sample_idx': sample_idx,
                'global_idx': len(sequence_embeddings) - 1
            })
    
    # 转换为numpy数组进行t-SNE
    X = np.array(sequence_embeddings)
    tsne_result = tsne.fit_transform(X)
    
    # 构建颜色标签
    color_labels = [info['model_name'] for info in sequence_indices]
    
    # 创建散点图
    plt.figure(figsize=(15, 10))
    
    # 绘制所有点
    for model_name in set(color_labels):
        indices = [j for j, lbl in enumerate(color_labels) if lbl == model_name]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                   label=model_name, s=50, alpha=0.7)

    # 先保存无对应关系的版本
    plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Sequence")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_sequence.png"), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: tsne_layer_{layer_idx}_{label}_per_sequence.png")
    
    # 如果启用对应关系，添加连线
    if show_correspondence:
        plot_sequence_correspondence_lines(tsne_result, sequence_indices, model_names, label)
        
        plt.title(f"TSNE of {label} Cache (Layer {layer_idx}) - Per Sequence with Correspondence")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"tsne_layer_{layer_idx}_{label}_per_sequence_with_correspondence.png"), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: tsne_layer_{layer_idx}_{label}_per_sequence_with_correspondence.png")


def plot_sequence_correspondence_lines(tsne_result, sequence_indices, model_names, label):
    """绘制序列级别的模型对应关系连线"""
    # 找到模型索引
    model_idx_map = {name: idx for idx, name in enumerate(model_names)}
    
    # 0.6B 和 Rosetta 的对应关系
    if 'Qwen3-0.6B' in model_idx_map and 'Rosetta' in model_idx_map:
        plot_sequence_model_correspondence(tsne_result, sequence_indices, 
                                         'Qwen3-0.6B', 'Rosetta', 'blue', alpha=0.5)
    
    # Rosetta 和 4B 的对应关系
    if 'Rosetta' in model_idx_map and 'Qwen3-4B' in model_idx_map:
        plot_sequence_model_correspondence(tsne_result, sequence_indices, 
                                         'Rosetta', 'Qwen3-4B', 'red', alpha=0.5)


def plot_sequence_model_correspondence(tsne_result, sequence_indices, model1_name, model2_name, color, alpha=0.5):
    """绘制两个模型之间的序列级别对应关系连线"""
    # 按样本索引分组
    model1_sequences = {}
    model2_sequences = {}
    
    for seq_info in sequence_indices:
        sample_idx = seq_info['sample_idx']
        if seq_info['model_name'] == model1_name:
            model1_sequences[sample_idx] = seq_info['global_idx']
        elif seq_info['model_name'] == model2_name:
            model2_sequences[sample_idx] = seq_info['global_idx']
    
    # 绘制对应关系连线
    for sample_idx in model1_sequences:
        if sample_idx in model2_sequences:
            idx1 = model1_sequences[sample_idx]
            idx2 = model2_sequences[sample_idx]
            
            x1, y1 = tsne_result[idx1]
            x2, y2 = tsne_result[idx2]
            
            plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1.5)
    
    # 添加图例说明
    plt.plot([], [], color=color, alpha=alpha, linewidth=2, 
             label=f'{model1_name} ↔ {model2_name} correspondence')


def plot_correspondence_lines(tsne_result, token_indices, model_names, label):
    """绘制模型之间的对应关系连线"""
    # 找到模型索引
    model_idx_map = {name: idx for idx, name in enumerate(model_names)}
    
    # 0.6B 和 Rosetta 的对应关系
    if 'Qwen3-0.6B' in model_idx_map and 'Rosetta' in model_idx_map:
        plot_model_correspondence(tsne_result, token_indices, 
                                'Qwen3-0.6B', 'Rosetta', 'blue', alpha=0.3)
    
    # Rosetta 和 4B 的对应关系
    if 'Rosetta' in model_idx_map and 'Qwen3-4B' in model_idx_map:
        plot_model_correspondence(tsne_result, token_indices, 
                                'Rosetta', 'Qwen3-4B', 'red', alpha=0.3)


def plot_model_correspondence(tsne_result, token_indices, model1_name, model2_name, color, alpha=0.3):
    """绘制两个模型之间的对应关系连线"""
    # 按样本和token位置分组
    model1_tokens = {}
    model2_tokens = {}
    
    for token_info in token_indices:
        key = (token_info['sample_idx'], token_info['token_idx'])
        if token_info['model_name'] == model1_name:
            model1_tokens[key] = token_info['global_idx']
        elif token_info['model_name'] == model2_name:
            model2_tokens[key] = token_info['global_idx']
    
    # 绘制对应关系连线
    for key in model1_tokens:
        if key in model2_tokens:
            idx1 = model1_tokens[key]
            idx2 = model2_tokens[key]
            
            x1, y1 = tsne_result[idx1]
            x2, y2 = tsne_result[idx2]
            
            plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)
    
    # 添加图例说明
    plt.plot([], [], color=color, alpha=alpha, linewidth=2, 
             label=f'{model1_name} ↔ {model2_name} correspondence')


def main(args):
    # 更新全局设备设置
    global DEVICE
    if 'device' in args and args['device'] is not None:
        DEVICE = args['device']
        print(f"Using specified device: {DEVICE}")
    
    dataset = MMLUChatDataset(split="validation", num_samples=None)

    all_models = []
    all_tokenizers = []
    for model_path in args['models']:
        print(f"Loading model: {model_path}")
        if "Rosetta" in model_path:
            model, tokenizer = load_rosetta_model("local/checkpoints/0.6B_4B_context/final")
        else:
            model, tokenizer = load_qwen_model(model_path)
        model.eval()
        all_models.append(model)
        all_tokenizers.append(tokenizer)

    os.makedirs(args['output_dir'], exist_ok=True)

    # 只分析几个关键层
    layers_to_analyze = [21,22,23,24,25,26,27]  # 选择几个代表性的层
    layer_idx_offset_list = [0, 0, 8]
    for layer_idx in layers_to_analyze:
        # 根据模式选择样本数量
        if args.get('mode', 'both') in ['sequence', 'both']:
            # per sequence模式需要更多样本以获得更好的可视化效果
            num_samples = args.get('num_samples', 50)
        else:
            # per token模式样本数可以少一些
            num_samples = args.get('num_samples', 10)
        
        k_layer_embeddings = []
        v_layer_embeddings = []
        for model, tokenizer, layer_idx_offset in zip(all_models, all_tokenizers, layer_idx_offset_list):
            k_values = extract_k_cache(model, tokenizer, dataset, layer_idx=layer_idx + layer_idx_offset, num_samples=num_samples)
            # v_values = extract_v_cache(model, tokenizer, dataset, layer_idx=layer_idx, num_samples=num_samples)
            k_layer_embeddings.append(k_values)
            # v_layer_embeddings.append(v_values)

        # 根据模式选择绘制方式
        if args.get('mode', 'both') in ['token', 'both']:
            plot_tsne_per_token(k_layer_embeddings, "k", args['models'], layer_idx, args['output_dir'], 
                               args.get('show_correspondence', True))
        
        if args.get('mode', 'both') in ['sequence', 'both']:
            plot_tsne_per_sequence(k_layer_embeddings, "k", args['models'], layer_idx, args['output_dir'], 
                                  args.get('show_correspondence', True))
        
        # plot_tsne(v_layer_embeddings, "v", args['models'], layer_idx, args['output_dir'], 
                #  args.get('show_correspondence', True))


if __name__ == "__main__":
    """
    使用示例:
    python tsne.py                                    # 自动检测设备，显示对应关系，两种模式都生成
    python tsne.py --device cuda                     # 指定使用CUDA
    python tsne.py --device cpu                      # 指定使用CPU
    python tsne.py --output_dir my_plots             # 指定输出目录
    python tsne.py --models Rosetta Qwen3-0.6B       # 指定要分析的模型
    python tsne.py --no-correspondence               # 不显示对应关系连线
    python tsne.py --mode token                      # 只生成token级别的图（10个样本）
    python tsne.py --mode sequence                   # 只生成序列级别的图（50个样本）
    python tsne.py --mode both                       # 生成两种模式的图（默认）
    python tsne.py --num_samples 100                 # 指定使用100个样本
    python tsne.py --mode sequence --num_samples 30  # 序列模式使用30个样本
    """
    parser = argparse.ArgumentParser(description='Generate t-SNE plots for KV cache analysis')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda, mps, cpu). If not specified, auto-detect.')
    parser.add_argument('--output_dir', type=str, default="new_tsne_outputs",
                       help='Output directory for t-SNE plots')
    parser.add_argument('--models', nargs='+', 
                       default=["Rosetta", "Qwen3-0.6B", "Qwen3-4B"],
                       help='Models to analyze')
    parser.add_argument('--no-correspondence', action='store_true',
                       help='Disable correspondence lines between models')
    parser.add_argument('--mode', type=str, choices=['token', 'sequence', 'both'], 
                       default='both',
                       help='Visualization mode: token (per-token), sequence (per-sequence), or both (default)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to use. If not specified, uses 10 for token mode and 50 for sequence mode')
    
    args = parser.parse_args()
    
    # Convert to dict for compatibility
    args_dict = {
        'device': args.device,
        'output_dir': args.output_dir,
        'models': args.models,
        'show_correspondence': not args.no_correspondence,
        'mode': args.mode,
        'num_samples': args.num_samples
    }

    main(args_dict)
