#!/usr/bin/env python3
"""
批量评测已训练的模型检查点脚本
自动遍历scaling_MMLU_15k目录下的所有checkpoint，生成评测配置并运行评测
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_evaluate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchCheckpointEvaluator:
    """批量checkpoint评测器"""
    
    def __init__(self):
        # 路径配置
        self.checkpoints_base_dir = Path("local/checkpoints/scaling_MMLU_15k")
        self.eval_template_path = "eval_recipe/test_eval.yaml"
        self.training_configs_dir = Path("recipe")
        self.evaluation_script = "script/evaluation/unified_evaluator.py"
        
        # 模型家族定义（与auto_scaling_experiment.py保持一致）
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
        
        # 尺寸映射
        self.size_name_to_key = {
            "0.6B": "tiny", "1.7B": "small", "4B": "medium", "8B": "large", "14B": "xlarge",
            "0.5B": "tiny", "1.5B": "small", "3B": "medium", "7B": "large"
        }
    
    def find_checkpoints(self) -> List[Tuple[str, Path]]:
        """
        找到所有可用的checkpoint目录
        
        Returns:
            List of (experiment_name, checkpoint_path) tuples
        """
        checkpoints = []
        
        if not self.checkpoints_base_dir.exists():
            logger.error(f"Checkpoints base directory does not exist: {self.checkpoints_base_dir}")
            return checkpoints
        
        for item in self.checkpoints_base_dir.iterdir():
            if item.is_dir() and item.name.startswith("scaling_"):
                # 检查是否存在final子目录
                final_dir = item / "final"
                if final_dir.exists() and final_dir.is_dir():
                    experiment_name = item.name.replace("scaling_", "")
                    checkpoints.append((experiment_name, final_dir))
                    logger.info(f"Found checkpoint: {experiment_name} -> {final_dir}")
                else:
                    logger.warning(f"No 'final' directory found in {item}")
        
        logger.info(f"Found {len(checkpoints)} valid checkpoints")
        return checkpoints
    
    def parse_experiment_name(self, experiment_name: str) -> Optional[Tuple[str, str]]:
        """
        解析实验名称，提取base模型和teacher模型信息
        
        Args:
            experiment_name: 实验名称，如 "0.6B+0.5B", "1.7B+3B"
            
        Returns:
            (base_size_key, teacher_size_key) 或 None
        """
        if "+" not in experiment_name:
            logger.error(f"Invalid experiment name format: {experiment_name}")
            return None
        
        try:
            base_size, teacher_size = experiment_name.split("+")
            
            # 查找对应的尺寸键
            base_key = self.size_name_to_key.get(base_size)
            teacher_key = self.size_name_to_key.get(teacher_size)
            
            if base_key is None or teacher_key is None:
                logger.error(f"Unknown model sizes in experiment: {experiment_name}")
                logger.error(f"  Base size '{base_size}' -> {base_key}")
                logger.error(f"  Teacher size '{teacher_size}' -> {teacher_key}")
                return None
            
            return base_key, teacher_key
            
        except ValueError:
            logger.error(f"Failed to parse experiment name: {experiment_name}")
            return None
    
    def load_eval_template(self) -> Dict:
        """加载评测模板配置"""
        try:
            with open(self.eval_template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load eval template {self.eval_template_path}: {e}")
            raise
    
    def create_eval_config(self, experiment_name: str, checkpoint_path: Path, 
                          base_model_path: str, teacher_model_path: str) -> str:
        """
        创建评测配置文件
        
        Args:
            experiment_name: 实验名称
            checkpoint_path: checkpoint路径
            base_model_path: base模型路径
            teacher_model_path: teacher模型路径
            
        Returns:
            生成的配置文件路径
        """
        # 加载模板
        config = self.load_eval_template()
        
        # 修改配置
        config["model"]["model_name"] = "Rosetta"
        config["model"]["rosetta_config"]["base_model"] = base_model_path
        config["model"]["rosetta_config"]["teacher_model"] = teacher_model_path
        config["model"]["rosetta_config"]["checkpoints_dir"] = str(checkpoint_path)
        
        # 设置输出目录
        output_dir = f"local/scaling_new_prompt_results/{experiment_name}_mmlu-redux"
        config["output"]["output_dir"] = output_dir
        
        # 保存配置文件
        config_filename = f"eval_recipe/scaling_new_prompt_{experiment_name}_eval.yaml"
        with open(config_filename, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Created eval config: {config_filename}")
        logger.info(f"  Base model: {base_model_path}")
        logger.info(f"  Teacher model: {teacher_model_path}")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Output dir: {output_dir}")
        
        return config_filename
    
    def run_evaluation(self, config_path: str, experiment_name: str) -> bool:
        """
        运行单个实验的评测
        
        Args:
            config_path: 评测配置文件路径
            experiment_name: 实验名称
            
        Returns:
            是否评测成功
        """
        logger.info(f"Starting evaluation for experiment: {experiment_name}")
        
        try:
            # 构建评测命令
            cmd = [
                "python", self.evaluation_script,
                "--config", config_path
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # 运行评测（实时输出）
            process = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # 实时输出日志
            if process.stdout:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            
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
    
    def cleanup_temp_configs(self, temp_configs: List[str]):
        """清理临时生成的配置文件"""
        for config_path in temp_configs:
            try:
                if os.path.exists(config_path):
                    os.remove(config_path)
                    logger.info(f"Removed temporary config: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary config {config_path}: {e}")
    
    def check_existing_results(self, experiment_name: str) -> bool:
        """检查实验结果是否已存在"""
        output_dir = Path(f"local/scaling_new_prompt_results/{experiment_name}_mmlu-redux")
        if output_dir.exists() and any(output_dir.iterdir()):
            logger.info(f"Results already exist for {experiment_name}, skipping")
            return True
        return False
    
    def run_batch_evaluation(self, skip_existing: bool = True, dry_run: bool = False):
        """
        运行批量评测
        
        Args:
            skip_existing: 是否跳过已有结果的实验
            dry_run: 是否为试运行（不实际执行评测）
        """
        logger.info("=" * 80)
        logger.info("STARTING BATCH CHECKPOINT EVALUATION")
        logger.info("=" * 80)
        
        # 查找所有checkpoint
        checkpoints = self.find_checkpoints()
        
        if not checkpoints:
            logger.error("No valid checkpoints found!")
            return
        
        logger.info(f"Found {len(checkpoints)} checkpoints to evaluate:")
        for exp_name, checkpoint_path in checkpoints:
            logger.info(f"  ✓ {exp_name} -> {checkpoint_path}")
        
        if dry_run:
            logger.info("DRY RUN MODE - No actual evaluations will be performed")
        
        successful_evaluations = 0
        failed_evaluations = 0
        skipped_evaluations = 0
        temp_configs = []
        
        for i, (experiment_name, checkpoint_path) in enumerate(checkpoints, 1):
            logger.info(f"\n[{i}/{len(checkpoints)}] Processing experiment: {experiment_name}")
            
            # 检查是否已有结果
            if skip_existing and self.check_existing_results(experiment_name):
                skipped_evaluations += 1
                continue
            
            # 解析实验名称
            parsed = self.parse_experiment_name(experiment_name)
            if parsed is None:
                logger.error(f"Failed to parse experiment name: {experiment_name}")
                failed_evaluations += 1
                continue
            
            base_key, teacher_key = parsed
            base_model_path = self.family1[base_key]
            teacher_model_path = self.family2[teacher_key]
            
            try:
                # 创建评测配置
                config_path = self.create_eval_config(
                    experiment_name, checkpoint_path, 
                    base_model_path, teacher_model_path
                )
                temp_configs.append(config_path)
                
                if dry_run:
                    logger.info(f"DRY RUN: Would evaluate {experiment_name}")
                    successful_evaluations += 1
                else:
                    # 运行评测
                    success = self.run_evaluation(config_path, experiment_name)
                    
                    if success:
                        successful_evaluations += 1
                    else:
                        failed_evaluations += 1
                        
            except Exception as e:
                logger.error(f"Failed to process {experiment_name}: {e}")
                failed_evaluations += 1
        
        # 清理临时配置文件
        if not dry_run:
            self.cleanup_temp_configs(temp_configs)
        
        # 输出最终结果
        logger.info(f"\n" + "=" * 80)
        logger.info("BATCH EVALUATION SUMMARY")
        logger.info(f"=" * 80)
        logger.info(f"Total checkpoints found: {len(checkpoints)}")
        logger.info(f"Successful evaluations: {successful_evaluations}")
        logger.info(f"Failed evaluations: {failed_evaluations}")
        logger.info(f"Skipped evaluations: {skipped_evaluations}")
        
        if dry_run:
            logger.info("\nThis was a DRY RUN. No actual evaluations were performed.")
            logger.info("Run with --no-dry-run to perform actual evaluations.")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch evaluate trained checkpoints')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip experiments that already have results (default: True)')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                       help='Evaluate all experiments, even if results exist')
    parser.add_argument('--dry-run', action='store_true', default=False,
                       help='Show what would be evaluated without running actual evaluations')
    parser.add_argument('--no-dry-run', action='store_false', dest='dry_run',
                       help='Perform actual evaluations (default)')
    
    args = parser.parse_args()
    
    # 创建评测器并运行
    evaluator = BatchCheckpointEvaluator()
    evaluator.run_batch_evaluation(
        skip_existing=args.skip_existing, 
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
