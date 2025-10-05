#!/usr/bin/env python3
"""
分析CoT评估结果，按length ratio分段统计准确率并绘制折线图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict
import seaborn as sns

def load_data(filepath):
    """加载JSON数据文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_question_data(data):
    """从JSON数据中提取所有题目的数据"""
    questions = []
    
    for category, category_data in data.items():
        if isinstance(category_data, list):
            for item in category_data:
                if 'length_ratio' in item and 'is_correct' in item:
                    questions.append({
                        'category': category,
                        'subject': item.get('subject', ''),
                        'question_id': item.get('question_id', ''),
                        'length_ratio': item['length_ratio'],
                        'is_correct': item['is_correct'],
                        'input_length': item.get('input_length', 0),
                        'gen_length': item.get('gen_length', 0)
                    })
    
    return questions

def create_length_ratio_bins(questions, num_bins=20, min_questions_per_bin=10, coverage_percentile=95):
    """创建length ratio的分段，排除极值，确保每段有足够题目数"""
    length_ratios = [q['length_ratio'] for q in questions]
    length_ratios = np.array(sorted(length_ratios))
    
    # 使用百分位数来确定切段范围，排除极值
    lower_bound = np.percentile(length_ratios, (100 - coverage_percentile) / 2)
    upper_bound = np.percentile(length_ratios, coverage_percentile + (100 - coverage_percentile) / 2)
    
    print(f"使用 {coverage_percentile}% 覆盖范围: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"原始范围: {length_ratios.min():.2f} - {length_ratios.max():.2f}")
    
    # 过滤出在范围内的数据
    filtered_ratios = length_ratios[(length_ratios >= lower_bound) & (length_ratios <= upper_bound)]
    print(f"过滤后的题目数量: {len(filtered_ratios)} / {len(length_ratios)}")
    
    # 创建初始等距分段
    initial_bins = np.linspace(lower_bound, upper_bound, num_bins + 1)
    
    # 优化分段，确保每段有足够的题目数
    optimized_bins = [initial_bins[0]]
    current_count = 0
    
    for i in range(1, len(initial_bins)):
        # 统计当前段的题目数
        segment_count = np.sum((filtered_ratios >= optimized_bins[-1]) & 
                             (filtered_ratios < initial_bins[i]))
        current_count += segment_count
        
        # 如果是最后一段或者当前累积的题目数足够多，就添加这个分段点
        if i == len(initial_bins) - 1 or current_count >= min_questions_per_bin:
            optimized_bins.append(initial_bins[i])
            current_count = 0
    
    # 确保最后一个点是上边界
    if optimized_bins[-1] != upper_bound:
        optimized_bins[-1] = upper_bound
    
    print(f"优化后的分段数: {len(optimized_bins) - 1}")
    return np.array(optimized_bins), lower_bound, upper_bound

def calculate_accuracy_by_bins(questions, bins, range_bounds=None):
    """计算每个分段的准确率"""
    bin_stats = []
    
    # 如果提供了范围边界，先过滤题目
    if range_bounds:
        lower_bound, upper_bound = range_bounds
        questions = [q for q in questions if lower_bound <= q['length_ratio'] <= upper_bound]
    
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        
        # 找到在这个分段内的题目
        bin_questions = [q for q in questions if bin_min <= q['length_ratio'] < bin_max]
        
        if i == len(bins) - 2:  # 最后一个分段包含上边界
            bin_questions = [q for q in questions if bin_min <= q['length_ratio'] <= bin_max]
        
        if bin_questions:
            correct_count = sum(1 for q in bin_questions if q['is_correct'])
            total_count = len(bin_questions)
            accuracy = correct_count / total_count
            
            bin_stats.append({
                'bin_min': bin_min,
                'bin_max': bin_max,
                'bin_center': (bin_min + bin_max) / 2,
                'accuracy': accuracy,
                'total_questions': total_count,
                'correct_questions': correct_count
            })
        else:
            bin_stats.append({
                'bin_min': bin_min,
                'bin_max': bin_max,
                'bin_center': (bin_min + bin_max) / 2,
                'accuracy': 0,
                'total_questions': 0,
                'correct_questions': 0
            })
    
    return bin_stats

def match_questions_across_models(rosetta_questions, qwen_questions):
    """匹配不同模型间的相同题目"""
    # 创建题目索引 (subject, question_id)
    rosetta_index = {(q['subject'], q['question_id']): q for q in rosetta_questions}
    
    matched_questions = []
    for qwen_q in qwen_questions:
        key = (qwen_q['subject'], qwen_q['question_id'])
        if key in rosetta_index:
            matched_questions.append({
                'rosetta': rosetta_index[key],
                'qwen': qwen_q
            })
    
    return matched_questions

def plot_accuracy_vs_length_ratio():
    """绘制准确率vs长度比的折线图"""
    # 加载数据
    print("加载数据文件...")
    rosetta_data = load_data('/share/minzihan/unified_memory/cot_eval_results/Rosetta_generate_20250819_014730_detailed_length.json')
    qwen06b_data = load_data('/share/minzihan/unified_memory/cot_eval_results/Qwen3-0.6B_generate_20250818_210815_detailed_length.json')
    qwen4b_data = load_data('/share/minzihan/unified_memory/cot_eval_results/Qwen3-4B_generate_20250819_013930_detailed_length.json')
    
    # 提取题目数据
    print("提取题目数据...")
    rosetta_questions = extract_question_data(rosetta_data)
    qwen06b_questions = extract_question_data(qwen06b_data)
    qwen4b_questions = extract_question_data(qwen4b_data)
    
    print(f"Rosetta题目数量: {len(rosetta_questions)}")
    print(f"Qwen-0.6B题目数量: {len(qwen06b_questions)}")
    print(f"Qwen-4B题目数量: {len(qwen4b_questions)}")
    
    # 匹配相同题目
    print("匹配相同题目...")
    matched_06b = match_questions_across_models(rosetta_questions, qwen06b_questions)
    matched_4b = match_questions_across_models(rosetta_questions, qwen4b_questions)
    
    print(f"与Qwen-0.6B匹配的题目数量: {len(matched_06b)}")
    print(f"与Qwen-4B匹配的题目数量: {len(matched_4b)}")
    
    # 基于Rosetta的length ratio创建分段，使用更细粒度的划分
    bins, lower_bound, upper_bound = create_length_ratio_bins(
        rosetta_questions, 
        num_bins=20, 
        min_questions_per_bin=100, 
        coverage_percentile=95
    )
    
    range_bounds = (lower_bound, upper_bound)
    print(f"使用范围: {lower_bound:.2f} - {upper_bound:.2f}")
    
    # 计算每个模型的分段准确率
    print("计算分段准确率...")
    rosetta_stats = calculate_accuracy_by_bins(rosetta_questions, bins, range_bounds)
    
    # 对于匹配的题目，使用Rosetta的length_ratio进行分段
    qwen06b_matched_questions = []
    qwen4b_matched_questions = []
    
    # 只保留在范围内的匹配题目
    for match in matched_06b:
        rosetta_ratio = match['rosetta']['length_ratio']
        if lower_bound <= rosetta_ratio <= upper_bound:
            qwen_q = match['qwen'].copy()
            qwen_q['length_ratio'] = rosetta_ratio
            qwen06b_matched_questions.append(qwen_q)
    
    for match in matched_4b:
        rosetta_ratio = match['rosetta']['length_ratio']
        if lower_bound <= rosetta_ratio <= upper_bound:
            qwen_q = match['qwen'].copy()
            qwen_q['length_ratio'] = rosetta_ratio
            qwen4b_matched_questions.append(qwen_q)
    
    qwen06b_stats = calculate_accuracy_by_bins(qwen06b_matched_questions, bins)
    qwen4b_stats = calculate_accuracy_by_bins(qwen4b_matched_questions, bins)
    
    print(f"过滤后匹配的题目数量:")
    print(f"Qwen-0.6B: {len(qwen06b_matched_questions)}")
    print(f"Qwen-4B: {len(qwen4b_matched_questions)}")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表 - 使用子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 提取数据用于绘图，只包含有题目的分段
    valid_indices = [i for i, stat in enumerate(rosetta_stats) if stat['total_questions'] > 0]
    
    x_values = [rosetta_stats[i]['bin_center'] for i in valid_indices]
    rosetta_accuracies = [rosetta_stats[i]['accuracy'] for i in valid_indices]
    qwen06b_accuracies = [qwen06b_stats[i]['accuracy'] for i in valid_indices]
    qwen4b_accuracies = [qwen4b_stats[i]['accuracy'] for i in valid_indices]
    
    # 计算准确率提升百分比
    improvement_percentiles = []
    for i in range(len(valid_indices)):
        rosetta_acc = rosetta_accuracies[i]
        qwen06b_acc = qwen06b_accuracies[i]
        qwen4b_acc = qwen4b_accuracies[i]
        
        # 计算公式：(rosetta_acc - qwen06b_acc) / (qwen4b_acc - qwen06b_acc)
        # 避免除零错误
        if abs(qwen4b_acc - qwen06b_acc) < 1e-6:  # 分母接近0
            if abs(rosetta_acc - qwen06b_acc) < 1e-6:  # 分子也接近0
                percentile = 0.0  # 都没有提升
            else:
                percentile = 1.0 if rosetta_acc > qwen06b_acc else -1.0  # 只有rosetta有提升/下降
        else:
            percentile = (rosetta_acc - qwen06b_acc) / (qwen4b_acc - qwen06b_acc)
        
        improvement_percentiles.append(percentile * 100)  # 转换为百分比
    
    # 第一个子图：准确率对比
    ax1.plot(x_values, rosetta_accuracies, marker='o', linewidth=2.5, markersize=7, 
             label='Rosetta (Context CoT)', color='#2E86AB', alpha=0.8)
    ax1.plot(x_values, qwen06b_accuracies, marker='s', linewidth=2.5, markersize=7, 
             label='Qwen3-0.6B', color='#A23B72', alpha=0.8)
    ax1.plot(x_values, qwen4b_accuracies, marker='^', linewidth=2.5, markersize=7, 
             label='Qwen3-4B', color='#F18F01', alpha=0.8)
    
    # 设置第一个子图
    ax1.set_xlabel('Length Ratio (Generated Length / Input Length)', fontsize=13)
    ax1.set_ylabel('Accuracy', fontsize=13)
    ax1.set_title('Accuracy vs Length Ratio Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(lower_bound - 0.1, upper_bound + 0.1)
    
    # 第二个子图：准确率提升百分比
    ax2.plot(x_values, improvement_percentiles, marker='D', linewidth=3, markersize=8, 
             label='Improvement Percentile', color='#E74C3C', alpha=0.9)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline (0%)')
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Full Improvement (100%)')
    
    # 设置第二个子图
    ax2.set_xlabel('Length Ratio (Generated Length / Input Length)', fontsize=13)
    ax2.set_ylabel('Improvement Percentile (%)', fontsize=13)
    ax2.set_title('Rosetta Improvement Percentile: (Rosetta - Qwen-0.6B) / (Qwen-4B - Qwen-0.6B) × 100%', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(lower_bound - 0.1, upper_bound + 0.1)
    
    # 动态设置y轴范围，确保能看到所有数据
    y_min = min(improvement_percentiles) - 10
    y_max = max(improvement_percentiles) + 10
    ax2.set_ylim(y_min, y_max)
    
    # 添加颜色区域来标示不同的性能水平
    ax2.axhspan(0, 50, alpha=0.1, color='orange', label='Below Average')
    ax2.axhspan(50, 100, alpha=0.1, color='lightgreen', label='Above Average')
    if y_max > 100:
        ax2.axhspan(100, y_max, alpha=0.1, color='darkgreen', label='Exceeds Qwen-4B')
    if y_min < 0:
        ax2.axhspan(y_min, 0, alpha=0.1, color='lightcoral', label='Below Qwen-0.6B')
    
    # 添加详细信息文本
    total_rosetta_in_range = len([q for q in rosetta_questions if lower_bound <= q['length_ratio'] <= upper_bound])
    avg_improvement = np.mean(improvement_percentiles)
    info_text = f"""Data Statistics (in range {lower_bound:.1f}-{upper_bound:.1f}):
• Rosetta: {total_rosetta_in_range} questions
• Qwen-0.6B: {len(qwen06b_matched_questions)} matched questions  
• Qwen-4B: {len(qwen4b_matched_questions)} matched questions
• Bins: {len(valid_indices)} segments
• Avg Improvement: {avg_improvement:.1f}%"""
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = '/share/minzihan/unified_memory/length_ratio_analysis_with_improvement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 显示详细统计信息
    print("\n=== 详细统计信息 (包含提升百分比) ===")
    print(f"{'Range':<15} {'Rosetta':<10} {'Qwen-0.6B':<10} {'Qwen-4B':<10} {'Improvement%':<12} {'Questions':<10}")
    print("-" * 80)
    
    for i, idx in enumerate(valid_indices):
        r_stat, q06_stat, q4_stat = rosetta_stats[idx], qwen06b_stats[idx], qwen4b_stats[idx]
        range_str = f"{r_stat['bin_min']:.1f}-{r_stat['bin_max']:.1f}"
        improvement = improvement_percentiles[i]
        print(f"{range_str:<15} {r_stat['accuracy']:<10.3f} {q06_stat['accuracy']:<10.3f} {q4_stat['accuracy']:<10.3f} {improvement:<12.1f} {r_stat['total_questions']:<10}")
    
    print(f"\n平均提升百分比: {avg_improvement:.1f}%")
    print(f"提升百分比范围: {min(improvement_percentiles):.1f}% - {max(improvement_percentiles):.1f}%")
    
    # 解释提升百分比的含义
    print("\n=== 提升百分比解释 ===")
    print("• 0%: Rosetta性能等于Qwen-0.6B")
    print("• 50%: Rosetta性能处于Qwen-0.6B和Qwen-4B的中间")
    print("• 100%: Rosetta性能等于Qwen-4B")
    print("• >100%: Rosetta性能超过Qwen-4B")
    print("• <0%: Rosetta性能低于Qwen-0.6B")
    
    plt.show()

if __name__ == "__main__":
    plot_accuracy_vs_length_ratio()
