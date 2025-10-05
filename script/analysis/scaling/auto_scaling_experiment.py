#!/usr/bin/env python3
"""
自动化模型训练和评估脚本
支持不同模型组合的批量实验
"""

import json
import yaml
import os
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_scaling_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelCombinationExperiment:
    """自动化模型组合实验类"""
    
    def __init__(self):
        # 定义两个模型家族

        # 配置文件路径
        self.training_config_template = "recipe/scaling.json"
        self.eval_config_template = "eval_recipe/scaling.yaml"
        
        # 基础路径
        self.base_checkpoint_dir = "local/checkpoints/scaling_MMLU_15k"
        self.base_eval_output_dir = "local/scaling_results"

        self.family1 = {
            "tiny": "/share/public/public_models/Qwen3-0.6B",
            "small": "/share/public/public_models/Qwen3-1.7B",
            "medium": "/share/public/public_models/Qwen3-4B", 
            "large": "/share/public/public_models/Qwen3-8B",
            "xlarge": "/share/public/public_models/Qwen3-14B"
        }
        
        self.family2 = {
            "tiny": "/share/public/public_models/Qwen2.5-0.5B-Instruct",
            "small": "/share/public/public_models/Qwen2.5-1.5B-Instruct",
            "medium": "/share/public/public_models/Qwen2.5-3B-Instruct",
            "large": "/share/public/public_models/Qwen2.5-7B-Instruct",
            "xlarge": "/share/public/public_models/Qwen2.5-14B-Instruct"
        }
        
        # 动态检查已完成的实验组合
        self.completed_experiments = self.check_completed_experiments()
        
        # 定义模型尺寸顺序（从小到大）
        size_order = ["tiny", "small", "medium", "large", "xlarge"]
        
        # 生成所有有效的实验组合
        # 约束：teacher模型尺寸 >= base模型尺寸
        all_combinations = []
        
        for base_idx, base_size in enumerate(size_order):
            for teacher_idx, teacher_size in enumerate(size_order):
                # 只有当teacher >= base时才添加组合
                if teacher_idx >= base_idx:
                    # 构造实验名称
                    base_name = self.get_model_size_name(base_size, "family1")
                    teacher_name = self.get_model_size_name(teacher_size, "family2")
                    experiment_name = f"{base_name}+{teacher_name}"
                    
                    all_combinations.append((base_size, teacher_size, experiment_name))
        
        # 去除重复组合并过滤掉已完成的实验
        unique_combinations = []
        seen_names = set()
        for base, teacher, name in all_combinations:
            if name not in seen_names and name not in self.completed_experiments:
                unique_combinations.append((base, teacher, name))
                seen_names.add(name)
        
        self.combinations = unique_combinations
    
    def get_model_size_name(self, size_key: str, family: str) -> str:
        """根据尺寸键获取模型大小标识"""
        size_mapping = {
            "tiny": "0.6B" if family == "family1" else "0.5B",
            "small": "1.7B" if family == "family1" else "1.5B", 
            "medium": "4B" if family == "family1" else "3B",
            "large": "8B" if family == "family1" else "7B",
            "xlarge": "14B"
        }
        return size_mapping[size_key]
    
    def check_completed_experiments(self) -> set:
        """动态检查已完成的实验组合"""
        completed = set()
        results_dir = Path(self.base_checkpoint_dir)
        
        if not results_dir.exists():
            logger.info(f"Results directory {results_dir} does not exist")
            return completed
        
        # 检查目录中的实验结果
        for item in results_dir.iterdir():
            if item.is_dir():
                dir_name = item.name
                if dir_name.startswith("scaling_"):
                    experiment_name = dir_name.replace("scaling_", "")
                    completed.add(experiment_name)
                    logger.info(f"Found completed experiment: {experiment_name}")
        
        return completed
    
    def load_json_config(self, config_path: str) -> Dict:
        """加载JSON配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON config {config_path}: {e}")
            raise
    
    def load_yaml_config(self, config_path: str) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML config {config_path}: {e}")
            raise
    
    def save_json_config(self, config: Dict, output_path: str):
        """保存JSON配置文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save JSON config {output_path}: {e}")
            raise
    
    def save_yaml_config(self, config: Dict, output_path: str):
        """保存YAML配置文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Failed to save YAML config {output_path}: {e}")
            raise
    
    def modify_training_config(self, base_size: str, teacher_size: str, experiment_name: str) -> str:
        """修改训练配置文件"""
        # 加载原始配置
        config = self.load_json_config(self.training_config_template)
        
        # 修改模型路径
        config["model"]["base_model"] = self.family1[base_size]
        config["model"]["teacher_model"] = self.family2[teacher_size]
        
        # 修改输出路径
        checkpoint_dir = f"{self.base_checkpoint_dir}/scaling_{experiment_name}"
        config["output"]["output_dir"] = checkpoint_dir
        config["output"]["wandb_config"]["run_name"] = f"scaling_{experiment_name}"
        
        # 保存修改后的配置
        output_config_path = f"recipe/scaling_{experiment_name}.json"
        self.save_json_config(config, output_config_path)
        
        logger.info(f"Training config saved to {output_config_path}")
        logger.info(f"Base model: {config['model']['base_model']}")
        logger.info(f"Teacher model: {config['model']['teacher_model']}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        
        return output_config_path, checkpoint_dir
    
    def modify_eval_config(self, base_size: str, teacher_size: str, checkpoint_dir: str, experiment_name: str) -> str:
        """修改评估配置文件"""
        # 加载原始配置
        config = self.load_yaml_config(self.eval_config_template)
        
        # 修改base model和teacher model路径
        config["model"]["rosetta_config"]["base_model"] = self.family1[base_size]
        config["model"]["rosetta_config"]["teacher_model"] = self.family2[teacher_size]
        
        # 修改检查点路径
        config["model"]["rosetta_config"]["checkpoints_dir"] = f"{checkpoint_dir}/final"
        
        # 修改输出路径
        eval_output_dir = f"{self.base_eval_output_dir}/{experiment_name}_mmlu-redux"
        config["output"]["output_dir"] = eval_output_dir
        
        # 保存修改后的配置
        output_config_path = f"eval_recipe/scaling_{experiment_name}.yaml"
        self.save_yaml_config(config, output_config_path)
        
        logger.info(f"Eval config saved to {output_config_path}")
        logger.info(f"Base model: {config['model']['rosetta_config']['base_model']}")
        logger.info(f"Teacher model: {config['model']['rosetta_config']['teacher_model']}")
        logger.info(f"Checkpoint dir: {config['model']['rosetta_config']['checkpoints_dir']}")
        logger.info(f"Eval output dir: {eval_output_dir}")
        
        return output_config_path
    
    def create_training_script(self, config_path: str, experiment_name: str) -> str:
        """为每个实验创建专用的训练脚本"""
        # 读取原始训练脚本
        with open("bash/train/sft_train_debug.sh", 'r') as f:
            original_script = f.read()
        
        # 替换配置文件路径
        modified_script = original_script.replace("recipe/scaling.json", config_path)
        
        # 创建新的脚本文件
        script_path = f"bash/train/sft_train_{experiment_name}.sh"
        with open(script_path, 'w') as f:
            f.write(modified_script)
        
        # 使脚本可执行
        os.chmod(script_path, 0o755)
        
        logger.info(f"Training script created: {script_path}")
        return script_path

    def run_training(self, config_path: str, experiment_name: str) -> bool:
        """运行模型训练"""
        logger.info(f"Starting training for experiment: {experiment_name}")
        
        try:
            # 创建专用的训练脚本
            script_path = self.create_training_script(config_path, experiment_name)
            
            # 构建训练命令
            cmd = ["bash", script_path]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # 运行训练（实时输出）
            process = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()  # 确保实时输出
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"Training completed successfully for {experiment_name}")
                # 清理临时脚本
                if os.path.exists(script_path):
                    os.remove(script_path)
                return True
            else:
                logger.error(f"Training failed for {experiment_name}")
                logger.error(f"Process returned code {process.returncode}")
                # 保留失败的脚本用于调试
                return False
                
        except Exception as e:
            logger.error(f"Exception during training {experiment_name}: {e}")
            return False
    
    def run_evaluation(self, config_path: str, experiment_name: str) -> bool:
        """运行模型评估"""
        logger.info(f"Starting evaluation for experiment: {experiment_name}")
        
        try:
            # 构建评估命令 (根据你的实际评估脚本调整)
            cmd = [
                "python", "script/evaluation/unified_evaluator.py",  # 假设你有一个eval.py脚本
                "--config", config_path
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # 运行评估（实时输出）
            process = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()  # 确保实时输出
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"Evaluation completed successfully for {experiment_name}")
                return True
            else:
                logger.error(f"Evaluation failed for {experiment_name}")
                logger.error(f"Process returned code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during evaluation {experiment_name}: {e}")
            return False
    
    def run_experiment(self, base_size: str, teacher_size: str, experiment_name: str) -> bool:
        """运行单个实验（训练+评估）"""
        logger.info(f"=" * 60)
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Base model size: {base_size}, Teacher model size: {teacher_size}")
        logger.info(f"=" * 60)
        
        try:
            # 1. 修改训练配置
            training_config_path, checkpoint_dir = self.modify_training_config(
                base_size, teacher_size, experiment_name
            )
            
            # 2. 运行训练
            training_success = self.run_training(training_config_path, experiment_name)
            
            if not training_success:
                logger.error(f"Training failed for {experiment_name}, skipping evaluation")
                return False
            
            # # 3. 修改评估配置
            # eval_config_path = self.modify_eval_config(base_size, teacher_size, checkpoint_dir, experiment_name)
            
            # # 4. 运行评估
            # eval_success = self.run_evaluation(eval_config_path, experiment_name)
            
            # if eval_success:
            #     logger.info(f"Experiment {experiment_name} completed successfully!")
            #     return True
            # else:
            #     logger.error(f"Evaluation failed for {experiment_name}")
            #     return False
                
        except Exception as e:
            logger.error(f"Exception in experiment {experiment_name}: {e}")
            return False
    
    def run_all_experiments(self, skip_failed: bool = True):
        """运行所有实验组合"""
        logger.info("=" * 80)
        logger.info("STARTING AUTO-SCALING EXPERIMENTS")
        logger.info("=" * 80)
        
        # 显示已完成的实验
        if self.completed_experiments:
            logger.info(f"Found {len(self.completed_experiments)} completed experiments:")
            for exp in sorted(self.completed_experiments):
                logger.info(f"  ✓ {exp}")
        else:
            logger.info("No previously completed experiments found.")
        
        # 显示待执行的实验
        logger.info(f"\nPlanning to run {len(self.combinations)} new experiments:")
        for i, (_, _, experiment_name) in enumerate(self.combinations, 1):
            logger.info(f"  {i:2d}. {experiment_name}")
        
        if len(self.combinations) == 0:
            logger.info("All experiments have been completed! Nothing to run.")
            return {}
        
        logger.info(f"\nStarting experiments...")
        
        results = {}
        successful_experiments = 0
        
        for i, (base_size, teacher_size, experiment_name) in enumerate(self.combinations, 1):
            logger.info(f"\n[{i}/{len(self.combinations)}] Processing combination: {experiment_name}")
            
            success = self.run_experiment(base_size, teacher_size, experiment_name)
            results[experiment_name] = success
            
            if success:
                successful_experiments += 1
            elif not skip_failed:
                logger.error(f"Stopping due to failed experiment: {experiment_name}")
                break
        
        # 输出最终结果
        logger.info(f"\n" + "=" * 80)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info(f"=" * 80)
        logger.info(f"Previously completed experiments: {len(self.completed_experiments)}")
        logger.info(f"New experiments attempted: {len(self.combinations)}")
        logger.info(f"New experiments successful: {successful_experiments}")
        logger.info(f"New experiments failed: {len(self.combinations) - successful_experiments}")
        
        logger.info(f"\nDetailed results:")
        for exp_name, success in results.items():
            status = "SUCCESS ✓" if success else "FAILED ✗"
            logger.info(f"  {exp_name}: {status}")
        
        return results

def main():
    """主函数"""
    experiment = ModelCombinationExperiment()
    
    # 运行所有实验
    results = experiment.run_all_experiments(skip_failed=True)
    
    # 可选：保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
