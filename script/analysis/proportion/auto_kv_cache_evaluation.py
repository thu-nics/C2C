#!/usr/bin/env python3
"""
KV Cache 比例和顺序自动评测脚本
基于test_eval.yaml模板，自动测试不同的kv_cache_proportion和kv_cache_order_mode组合
"""

import os
import yaml
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_kv_cache_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KVCacheAutoEvaluator:
    """KV Cache自动评测类"""
    
    def __init__(self, base_config_path: str = "eval_recipe/test_eval.yaml"):
        self.base_config_path = base_config_path
        self.base_config = self.load_base_config()
        
        # 定义测试参数组合
        self.proportions = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.order_modes = ["front", "back"]
        
        # 生成所有组合
        self.test_combinations = []
        for proportion in self.proportions:
            for order_mode in self.order_modes:
                experiment_name = f"prop_{proportion:0.2f}_order_{order_mode}"
                self.test_combinations.append((proportion, order_mode, experiment_name))
        
        # 创建配置和结果目录
        self.config_dir = Path("eval_configs_kv_cache")
        self.config_dir.mkdir(exist_ok=True)
        
        logger.info(f"初始化完成，将测试{len(self.test_combinations)}种配置组合")
    
    def load_base_config(self) -> Dict:
        """加载基础配置文件"""
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load base config {self.base_config_path}: {e}")
            raise
    
    def create_config_for_combination(self, proportion: float, order_mode: str, experiment_name: str) -> str:
        """为特定组合创建配置文件"""
        # 深拷贝基础配置
        config = yaml.safe_load(yaml.dump(self.base_config))
        
        # 修改KV cache参数
        config["eval"]["kv_cache_proportion"] = proportion
        config["eval"]["kv_cache_order_mode"] = order_mode
        
        # 修改输出目录
        base_output_dir = config["output"]["output_dir"]
        config["output"]["output_dir"] = f"{base_output_dir}/{experiment_name}"
        
        # 保存配置文件
        config_path = self.config_dir / f"config_{experiment_name}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Created config: {config_path}")
        logger.info(f"  - Proportion: {proportion}")
        logger.info(f"  - Order mode: {order_mode}")
        logger.info(f"  - Output dir: {config['output']['output_dir']}")
        
        return str(config_path)
    
    def run_evaluation(self, config_path: str, experiment_name: str) -> bool:
        """运行单个评估"""
        logger.info(f"开始评估实验: {experiment_name}")
        
        try:
            # 构建评估命令
            cmd = [
                "python", "script/evaluation/unified_evaluator.py",
                "--config", config_path
            ]
            
            logger.info(f"运行命令: {' '.join(cmd)}")
            
            # 运行评估（实时输出）
            process = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # 实时输出日志
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[{experiment_name}] {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"评估成功完成: {experiment_name}")
                return True
            else:
                logger.error(f"评估失败: {experiment_name}, 返回码: {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"评估异常: {experiment_name}, 错误: {e}")
            return False
    
    def extract_results(self, output_dir: str) -> Dict:
        """从输出目录提取结果"""
        result_info = {
            "status": "unknown",
            "accuracy": None,
            "details": {}
        }
        
        output_path = Path(output_dir)
        if not output_path.exists():
            result_info["status"] = "missing_output"
            return result_info
        
        # 查找结果文件（可能的命名模式）
        possible_files = [
            "final_results.json",
            "evaluation_results.json", 
            "results.json",
            "summary.json"
        ]
        
        results_file = None
        for filename in possible_files:
            candidate = output_path / filename
            if candidate.exists():
                results_file = candidate
                break
        
        if results_file:
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    result_info["status"] = "success"
                    result_info["details"] = data
                    
                    # 尝试提取准确率
                    if "accuracy" in data:
                        result_info["accuracy"] = data["accuracy"]
                    elif "overall_accuracy" in data:
                        result_info["accuracy"] = data["overall_accuracy"]
                    elif "avg_accuracy" in data:
                        result_info["accuracy"] = data["avg_accuracy"]
                        
            except Exception as e:
                logger.warning(f"无法解析结果文件 {results_file}: {e}")
                result_info["status"] = "parse_error"
        else:
            result_info["status"] = "no_results_file"
            logger.warning(f"在 {output_dir} 中未找到结果文件")
        
        return result_info
    
    def run_all_experiments(self) -> Dict:
        """运行所有实验组合"""
        logger.info("="*80)
        logger.info("开始KV Cache自动评测实验")
        logger.info("="*80)
        
        logger.info(f"计划测试 {len(self.test_combinations)} 种配置组合:")
        for i, (proportion, order_mode, experiment_name) in enumerate(self.test_combinations, 1):
            logger.info(f"  {i:2d}. {experiment_name} (proportion={proportion}, order_mode={order_mode})")
        
        results = {}
        successful_experiments = 0
        
        for i, (proportion, order_mode, experiment_name) in enumerate(self.test_combinations, 1):
            logger.info(f"\n[{i}/{len(self.test_combinations)}] 处理组合: {experiment_name}")
            
            try:
                # 1. 创建配置文件
                config_path = self.create_config_for_combination(proportion, order_mode, experiment_name)
                
                # 2. 运行评估
                success = self.run_evaluation(config_path, experiment_name)
                
                # 3. 提取结果
                if success:
                    output_dir = f"{self.base_config['output']['output_dir']}/{experiment_name}"
                    result_info = self.extract_results(output_dir)
                    results[experiment_name] = {
                        "proportion": proportion,
                        "order_mode": order_mode,
                        "evaluation_success": True,
                        "result_info": result_info
                    }
                    successful_experiments += 1
                else:
                    results[experiment_name] = {
                        "proportion": proportion,
                        "order_mode": order_mode,
                        "evaluation_success": False,
                        "result_info": {"status": "evaluation_failed"}
                    }
                
            except Exception as e:
                logger.error(f"实验 {experiment_name} 发生异常: {e}")
                results[experiment_name] = {
                    "proportion": proportion,
                    "order_mode": order_mode,
                    "evaluation_success": False,
                    "result_info": {"status": "exception", "error": str(e)}
                }
        
        # 输出最终结果
        logger.info(f"\n" + "="*80)
        logger.info("最终结果汇总")
        logger.info("="*80)
        logger.info(f"总实验数: {len(self.test_combinations)}")
        logger.info(f"成功完成: {successful_experiments}")
        logger.info(f"失败数量: {len(self.test_combinations) - successful_experiments}")
        
        # 详细结果
        logger.info(f"\n详细结果:")
        for exp_name, result in results.items():
            status = "SUCCESS ✓" if result["evaluation_success"] else "FAILED ✗"
            accuracy = result["result_info"].get("accuracy", "N/A")
            logger.info(f"  {exp_name}: {status} (准确率: {accuracy})")
        
        return results
    
    def save_summary_report(self, results: Dict):
        """保存汇总报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_results_file = f"kv_cache_evaluation_results_{timestamp}.json"
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"详细结果已保存到: {detailed_results_file}")
        
        # 创建CSV汇总表
        csv_file = f"kv_cache_evaluation_summary_{timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("experiment_name,proportion,order_mode,evaluation_success,accuracy,status\n")
            for exp_name, result in results.items():
                proportion = result["proportion"]
                order_mode = result["order_mode"]
                eval_success = result["evaluation_success"]
                accuracy = result["result_info"].get("accuracy", "")
                status = result["result_info"].get("status", "")
                f.write(f"{exp_name},{proportion},{order_mode},{eval_success},{accuracy},{status}\n")
        logger.info(f"CSV汇总已保存到: {csv_file}")
        
        return detailed_results_file, csv_file


def main():
    """主函数"""
    evaluator = KVCacheAutoEvaluator()
    
    # 运行所有实验
    results = evaluator.run_all_experiments()
    
    # 保存汇总报告
    evaluator.save_summary_report(results)
    
    logger.info("\n" + "="*80)
    logger.info("KV Cache自动评测完成！")
    logger.info("="*80)


if __name__ == "__main__":
    main()
