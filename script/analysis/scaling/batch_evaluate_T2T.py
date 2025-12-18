#!/usr/bin/env python3
"""
批量进行 T2T（Two-Stage）评测的脚本
- 直接从两个模型家族（family1: Qwen3，family2: Qwen2.5-Instruct）组合生成评测配置
- 基于 T2T_scaling.yaml 作为模板生成临时评测配置
- 组合规则：family1 尺寸 ≤ family2 尺寸（例如 tiny→tiny/small/medium/large/xlarge）
- 支持跳过已有结果
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_evaluate_T2T.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class T2TBatchEvaluator:
    """T2T 家族组合评测器"""

    def __init__(self,
                 template_path: str = "T2T_scaling.yaml",
                 evaluation_script: str = "script/evaluation/unified_evaluator.py",
                 output_base_dir: str = "local/auto_eval_results_scaling_T2T"):
        # 模板与脚本路径
        self.template_path = template_path
        self.evaluation_script = evaluation_script
        self.output_base_dir = output_base_dir

        # 模型家族定义（与 batch_evaluate_checkpoints.py 保持一致）
        self.family1: Dict[str, str] = {
            "tiny": "Qwen/Qwen3-0.6B",
            "small": "Qwen/Qwen3-1.7B",
            "medium": "Qwen/Qwen3-4B",
            "large": "Qwen/Qwen3-8B",
            "xlarge": "Qwen/Qwen3-14B",
        }
        self.family2: Dict[str, str] = {
            "tiny": "Qwen/Qwen2.5-0.5B-Instruct",
            "small": "Qwen/Qwen2.5-1.5B-Instruct",
            "medium": "Qwen/Qwen2.5-3B-Instruct",
            "large": "Qwen/Qwen2.5-7B-Instruct",
            "xlarge": "Qwen/Qwen2.5-14B-Instruct",
        }

        # 尺寸标签（用于生成易读实验名）
        self.family1_size_label: Dict[str, str] = {
            "tiny": "0.6B",
            "small": "1.7B",
            "medium": "4B",
            "large": "8B",
            "xlarge": "14B",
        }
        self.family2_size_label: Dict[str, str] = {
            "tiny": "0.5B",
            "small": "1.5B",
            "medium": "3B",
            "large": "7B",
            "xlarge": "14B",
        }

        # 尺寸顺序与等级
        self.size_order: List[str] = ["tiny", "small", "medium", "large", "xlarge"]
        self.size_rank: Dict[str, int] = {k: i for i, k in enumerate(self.size_order)}

    # ------------------------------ 模板与配置 ------------------------------
    def load_template(self) -> Dict:
        """加载 T2T 评测模板配置"""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load template {self.template_path}: {e}")
            raise

    def create_eval_config(self,
                           experiment_name: str,
                           answer_model_path: str,
                           context_model_path: str) -> str:
        """
        根据模板创建单次实验的评测配置文件
        返回生成的配置文件路径
        """
        cfg = self.load_template()

        # 覆盖模型路径
        cfg.setdefault("model", {})
        cfg["model"]["model_name"] = "two_stage"
        cfg["model"]["answer_model_path"] = answer_model_path
        cfg["model"]["context_model_path"] = context_model_path

        # 输出路径带上数据集名（如 mmlu-redux）
        dataset = cfg.get("eval", {}).get("dataset", "mmlu-redux")
        output_dir = f"{self.output_base_dir}/{experiment_name}_{dataset}"
        cfg.setdefault("output", {})
        cfg["output"]["output_dir"] = output_dir

        # 将生成的配置写入临时文件
        config_filename = f"eval_recipe/T2T_scaling_{experiment_name}_eval.yaml"
        os.makedirs(Path(config_filename).parent, exist_ok=True)
        with open(config_filename, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Created eval config: {config_filename}")
        logger.info(f"  Answer model: {answer_model_path}")
        logger.info(f"  Context model: {context_model_path}")
        logger.info(f"  Output dir: {output_dir}")
        return config_filename

    # ------------------------------ 组合生成 ------------------------------
    def _experiment_name(self, size_key_answer: str, size_key_context: str) -> str:
        ans = self.family1_size_label.get(size_key_answer, size_key_answer)
        ctx = self.family2_size_label.get(size_key_context, size_key_context)
        return f"{ans}+{ctx}"

    def generate_ge_pairs(self) -> List[Tuple[str, str, str]]:
        """
        生成满足 family1 尺寸 ≤ family2 尺寸 的组合
        返回 (experiment_name, answer_model_path, context_model_path)
        """
        pairs: List[Tuple[str, str, str]] = []
        f1_keys = [k for k in self.size_order if k in self.family1]
        f2_keys = [k for k in self.size_order if k in self.family2]
        for k1 in f1_keys:
            for k2 in f2_keys:
                if self.size_rank[k2] >= self.size_rank[k1]:
                    exp = self._experiment_name(k1, k2)
                    pairs.append((exp, self.family1[k1], self.family2[k2]))
        return pairs

    # ------------------------------ 运行评测 ------------------------------
    def check_existing_results(self, experiment_name: str, dataset: str) -> bool:
        out_dir = Path(f"{self.output_base_dir}/{experiment_name}_{dataset}")
        if out_dir.exists() and any(out_dir.iterdir()):
            logger.info(f"Results already exist for {experiment_name}, skipping")
            return True
        return False

    def run_evaluation(self, config_path: str, experiment_name: str) -> bool:
        logger.info(f"Starting evaluation for experiment: {experiment_name}")
        try:
            cmd = [
                "python", self.evaluation_script,
                "--config", config_path,
            ]
            logger.info(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
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
        for config_path in temp_configs:
            try:
                if os.path.exists(config_path):
                    os.remove(config_path)
                    logger.info(f"Removed temporary config: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary config {config_path}: {e}")

    # ------------------------------ 批量主流程 ------------------------------
    def run_batch(self, skip_existing: bool = True):
        logger.info("=" * 80)
        logger.info("STARTING BATCH T2T EVALUATION")
        logger.info("=" * 80)

        # 预读模板，拿到 dataset 用于 skip 判断
        template = self.load_template()
        dataset = template.get("eval", {}).get("dataset", "mmlu-redux")

        pairs = self.generate_ge_pairs()
        logger.info(f"Prepared {len(pairs)} experiment pairs (F1<=F2)")
        for exp, ans, ctx in pairs:
            logger.info(f"  ✓ {exp} -> answer={ans} | context={ctx}")

        success = 0
        failed = 0
        skipped = 0
        temp_configs: List[str] = []

        for i, (exp_name, ans_path, ctx_path) in enumerate(pairs, 1):
            logger.info(f"\n[{i}/{len(pairs)}] Processing experiment: {exp_name}")

            if skip_existing and self.check_existing_results(exp_name, dataset):
                skipped += 1
                continue

            try:
                cfg_path = self.create_eval_config(exp_name, ans_path, ctx_path)
                temp_configs.append(cfg_path)
                ok = self.run_evaluation(cfg_path, exp_name)
                if ok:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to process {exp_name}: {e}")
                failed += 1

        self.cleanup_temp_configs(temp_configs)

        logger.info("\n" + "=" * 80)
        logger.info("BATCH T2T EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total experiments: {len(pairs)}")
        logger.info(f"Successful evaluations: {success}")
        logger.info(f"Failed evaluations: {failed}")
        logger.info(f"Skipped evaluations: {skipped}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch T2T evaluation across model families')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip experiments that already have results (default: True)')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                        help='Evaluate all experiments, even if results exist')

    parser.add_argument('--template', type=str, default='T2T_scaling.yaml',
                        help='Path to the T2T template yaml (default: T2T_scaling.yaml)')
    parser.add_argument('--output-base-dir', type=str, default='local/auto_eval_results_scaling_T2T',
                        help='Base output directory for results')

    args = parser.parse_args()

    evaluator = T2TBatchEvaluator(template_path=args.template,
                                  output_base_dir=args.output_base_dir)
    evaluator.run_batch(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
