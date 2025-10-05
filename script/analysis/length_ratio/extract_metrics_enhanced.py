"""
简单提取模型结果数据
提取Rosetta和Qwen的子类准确率，以及长度比信息
"""

import json
import pandas as pd

def extract_data():
    """提取两个模型的数据并整理成表格"""
    
    # 文件路径
    rosetta_file = "/share/minzihan/unified_memory/cot_eval_results/Rosetta_context_cot_2_generate_20250818_173215_summary.json"
    qwen_file = "/share/minzihan/unified_memory/cot_eval_results/Qwen3-4B_generate_20250818_181807_summary.json"
    
    # 加载数据
    with open(rosetta_file, 'r', encoding='utf-8') as f:
        rosetta_data = json.load(f)
    
    with open(qwen_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    
    # 提取Rosetta数据
    rosetta_metrics = {}
    if 'length_statistics' in rosetta_data and 'subcategories' in rosetta_data['length_statistics']:
        for subcategory, stats in rosetta_data['length_statistics']['subcategories'].items():
            rosetta_metrics[subcategory] = {
                'accuracy': stats.get('accuracy', 0.0),
                'length_ratio': stats.get('avg_length_ratio', 0.0),
                'samples': stats.get('total_samples', 0)
            }
    
    # 提取Qwen数据
    qwen_metrics = {}
    if 'length_statistics' in qwen_data and 'subcategories' in qwen_data['length_statistics']:
        for subcategory, stats in qwen_data['length_statistics']['subcategories'].items():
            qwen_metrics[subcategory] = {
                'accuracy': stats.get('accuracy', 0.0),
                'length_ratio': stats.get('avg_length_ratio', 0.0),
                'samples': stats.get('total_samples', 0)
            }
    
    # 获取所有子类
    all_subcategories = sorted(set(rosetta_metrics.keys()) | set(qwen_metrics.keys()))
    
    # 整理数据
    data = []
    for subcategory in all_subcategories:
        row = {
            '子类': subcategory,
            'Rosetta准确率': rosetta_metrics.get(subcategory, {}).get('accuracy', None),
            'Qwen准确率': qwen_metrics.get(subcategory, {}).get('accuracy', None),
            'Rosetta长度比': rosetta_metrics.get(subcategory, {}).get('length_ratio', None),
            'Qwen长度比': qwen_metrics.get(subcategory, {}).get('length_ratio', None),
            'Rosetta样本数': rosetta_metrics.get(subcategory, {}).get('samples', None),
            'Qwen样本数': qwen_metrics.get(subcategory, {}).get('samples', None)
        }
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 格式化数值，保留4位小数
    for col in ['Rosetta准确率', 'Qwen准确率', 'Rosetta长度比', 'Qwen长度比']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    return df

def main():
    """主函数"""
    print("提取模型评估数据...")
    
    # 提取数据
    df = extract_data()
    
    # 显示结果
    print("\n模型评估数据汇总:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 显示一些基本统计
    print("\n基本统计:")
    print("-" * 40)
    
    # 计算有效数据的统计
    rosetta_acc = df[df['Rosetta准确率'] != "N/A"]['Rosetta准确率'].apply(lambda x: float(x))
    qwen_acc = df[df['Qwen准确率'] != "N/A"]['Qwen准确率'].apply(lambda x: float(x))
    
    if len(rosetta_acc) > 0:
        print(f"Rosetta平均准确率: {rosetta_acc.mean():.4f}")
    if len(qwen_acc) > 0:
        print(f"Qwen平均准确率: {qwen_acc.mean():.4f}")
    
    print(f"共有 {len(df)} 个子类")
    print(f"Rosetta有数据的子类: {len(rosetta_acc)} 个")
    print(f"Qwen有数据的子类: {len(qwen_acc)} 个")
    
    # 分析Rosetta表现与长度比的关系
    print("\nRosetta表现与长度比分析:")
    print("-" * 50)
    
    # 筛选同时有Rosetta和Qwen准确率数据的子类
    valid_comparison = df[(df['Rosetta准确率'] != "N/A") & 
                         (df['Qwen准确率'] != "N/A") & 
                         (df['Rosetta长度比'] != "N/A")]
    
    if len(valid_comparison) > 0:
        # 转换为数值
        valid_comparison = valid_comparison.copy()
        valid_comparison['Rosetta准确率_num'] = valid_comparison['Rosetta准确率'].apply(float)
        valid_comparison['Qwen准确率_num'] = valid_comparison['Qwen准确率'].apply(float)
        valid_comparison['Rosetta长度比_num'] = valid_comparison['Rosetta长度比'].apply(float)
        
        # 计算Rosetta相对Qwen的表现差异
        valid_comparison['准确率差异'] = valid_comparison['Rosetta准确率_num'] - valid_comparison['Qwen准确率_num']
        
        # 分为强势和弱势子类
        rosetta_strong = valid_comparison[valid_comparison['准确率差异'] > 0]  # Rosetta比Qwen强的子类
        rosetta_weak = valid_comparison[valid_comparison['准确率差异'] < 0]    # Rosetta比Qwen弱的子类
        rosetta_equal = valid_comparison[valid_comparison['准确率差异'] == 0]   # 表现相等的子类
        
        print(f"可比较的子类总数: {len(valid_comparison)}")
        print(f"Rosetta表现更好的子类: {len(rosetta_strong)} 个")
        print(f"Rosetta表现较差的子类: {len(rosetta_weak)} 个")
        print(f"表现相等的子类: {len(rosetta_equal)} 个")
        
        if len(rosetta_strong) > 0:
            strong_avg_length = rosetta_strong['Rosetta长度比_num'].mean()
            print(f"\nRosetta强势子类的平均长度比: {strong_avg_length:.4f}")
            print("强势子类详情:")
            for _, row in rosetta_strong.iterrows():
                print(f"  {row['子类']}: 准确率差异 +{row['准确率差异']:.4f}, 长度比 {row['Rosetta长度比_num']:.4f}")
        
        if len(rosetta_weak) > 0:
            weak_avg_length = rosetta_weak['Rosetta长度比_num'].mean()
            print(f"\nRosetta弱势子类的平均长度比: {weak_avg_length:.4f}")
            print("弱势子类详情:")
            for _, row in rosetta_weak.iterrows():
                print(f"  {row['子类']}: 准确率差异 {row['准确率差异']:.4f}, 长度比 {row['Rosetta长度比_num']:.4f}")
        
        if len(rosetta_strong) > 0 and len(rosetta_weak) > 0:
            length_diff = strong_avg_length - weak_avg_length
            print(f"\n长度比差异分析:")
            print(f"强势子类平均长度比 - 弱势子类平均长度比 = {length_diff:.4f}")
            if length_diff > 0:
                print("→ Rosetta在长度比较高的子类上表现更好")
            elif length_diff < 0:
                print("→ Rosetta在长度比较低的子类上表现更好")
            else:
                print("→ 长度比与Rosetta的相对表现无明显关系")
    else:
        print("没有足够的数据进行比较分析")

if __name__ == "__main__":
    main()
