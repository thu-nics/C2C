#!/usr/bin/env python3
"""
Unified Ablation Study Script
Combines training and evaluation for ablation studies with different levels.
"""

import os
import sys
import json
import yaml
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    file_ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, "r") as f:
        if file_ext == ".json":
            return json.load(f)
        elif file_ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")

def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to file."""
    file_ext = os.path.splitext(output_path)[1].lower()
    
    with open(output_path, "w") as f:
        if file_ext == ".json":
            json.dump(config, f, indent=4)
        elif file_ext in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")

def create_ablation_config(base_config: Dict[str, Any], ablation_level: int, output_dir: str) -> Dict[str, Any]:
    """Create configuration for specific ablation level."""
    config = json.loads(json.dumps(base_config))  # Deep copy
    
    # Update ablation level
    config["model"]["projector"]["params"]["ablation_level"] = ablation_level
    
    # Update output directory
    config["output"]["output_dir"] = f"{output_dir}/level_{ablation_level}"
    
    # Update wandb run name
    config["output"]["wandb_config"]["run_name"] = f"ablation_level_{ablation_level}"
    
    return config

def create_eval_config(base_config: Dict[str, Any], ablation_level: int, 
                      checkpoint_dir: str, output_dir: str) -> Dict[str, Any]:
    """Create evaluation configuration for specific ablation level."""
    config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    
    # Update checkpoint directory
    config["model"]["rosetta_config"]["checkpoints_dir"] = f"{checkpoint_dir}/level_{ablation_level}/final"
    
    # Update output directory
    config["output"]["output_dir"] = f"{output_dir}/level_{ablation_level}_mmlu-redux"
    
    return config

def run_training(config_path: str, gpu_ids: list = None, 
                use_torchrun: bool = True, master_port: int = 29504) -> bool:
    """Run training with the given configuration using torchrun."""
    if use_torchrun:
        # Use torchrun for distributed training
        num_processes = len(gpu_ids) if gpu_ids else 8
        cmd = [
            "torchrun", 
            f"--nproc_per_node={num_processes}",
            f"--master_port={master_port}",
            "script/train/SFT_train.py",
            "--config", config_path
        ]
        
        # Set CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    else:
        # Use regular python for single GPU or dry run
        cmd = [
            "python", "script/train/SFT_train.py",
            "--config", config_path
        ]
        
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return False

def run_evaluation(config_path: str, gpu_ids: list = None) -> bool:
    """Run evaluation with the given configuration."""
    cmd = [
        "python", "script/evaluation/unified_evaluator.py",
        "--config", config_path
    ]
    
    if gpu_ids:
        # Set CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    else:
        env = None
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        return False

def check_checkpoint_exists(checkpoint_dir: str, level: int) -> bool:
    """Check if checkpoint directory exists for the given level."""
    level_checkpoint_dir = f"{checkpoint_dir}/level_{level}/final"
    return os.path.exists(level_checkpoint_dir) and os.path.isdir(level_checkpoint_dir)

def collect_results(output_dir: str, ablation_levels: List[int]) -> Dict[str, Any]:
    """Collect evaluation results from all ablation levels."""
    results = {}
    
    for level in ablation_levels:
        level_output_dir = f"{output_dir}/level_{level}_mmlu-redux"
        results_file = f"{level_output_dir}/multi_model_evaluation_summary.json"
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    level_results = json.load(f)
                results[f"level_{level}"] = level_results
                print(f"✅ Found results for level {level}")
            except Exception as e:
                print(f"⚠️  Failed to load results for level {level}: {e}")
                results[f"level_{level}"] = {"error": str(e)}
        else:
            print(f"⚠️  No results file found for level {level} at {results_file}")
            results[f"level_{level}"] = {"error": "No results file found"}
    
    return results

def print_results_summary(results: Dict[str, Any]):
    """Print a summary of evaluation results."""
    print(f"\n{'='*80}")
    print("📊 ABLATION EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Level':<8} {'Description':<40} {'MMLU Score':<12} {'Status':<10}")
    print("-" * 80)
    
    ablation_descriptions = {
        0: "Full C2C (baseline)",
        1: "No scalar weights",
        2: "No gates + No scalar weights", 
        3: "Source-only + No gates + No scalar weights"
    }
    
    for level_key, level_results in results.items():
        if level_key.startswith("level_"):
            level = int(level_key.split("_")[1])
            description = ablation_descriptions.get(level, "Unknown")
            
            if "error" in level_results:
                status = "Error"
                score = "N/A"
            else:
                # Try to extract MMLU score
                try:
                    if "mmlu-redux" in level_results:
                        mmlu_data = level_results["mmlu-redux"]
                        if "accuracy" in mmlu_data:
                            score = f"{mmlu_data['accuracy']:.3f}"
                        else:
                            score = "N/A"
                    else:
                        score = "N/A"
                    status = "Success"
                except:
                    score = "N/A"
                    status = "Partial"
            
            print(f"Level {level:<3} {description:<40} {score:<12} {status:<10}")
    
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Unified Ablation Study")
    
    # Configuration files
    parser.add_argument("--base_config", type=str, default="recipe/ablation_base.json",
                       help="Base training configuration file")
    parser.add_argument("--base_eval_config", type=str, default="eval_recipe/ablation_base.yaml",
                       help="Base evaluation configuration file")
    
    # Directories
    parser.add_argument("--output_dir", type=str, default="local/checkpoints/ablation_study",
                       help="Base output directory for training")
    parser.add_argument("--eval_output_dir", type=str, default="local/ablation_results",
                       help="Base output directory for evaluation results")
    
    # Training parameters
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
                       help="Comma-separated GPU IDs to use")
    parser.add_argument("--ablation_levels", type=str, default="0,1,2,3",
                       help="Comma-separated ablation levels to run")
    parser.add_argument("--master_port", type=int, default=29504,
                       help="Master port for distributed training")
    parser.add_argument("--use_torchrun", action="store_true", default=True,
                       help="Use torchrun for distributed training")
    
    # Control flags
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training phase")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation phase")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip training if output directory already exists")
    parser.add_argument("--collect_only", action="store_true",
                       help="Only collect and display existing results")
    
    args = parser.parse_args()
    
    # Parse arguments
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    ablation_levels = [int(x.strip()) for x in args.ablation_levels.split(",")]
    
    print("=" * 80)
    print("🔬 UNIFIED ABLATION STUDY")
    print("=" * 80)
    print(f"Training config: {args.base_config}")
    print(f"Evaluation config: {args.base_eval_config}")
    print(f"Training output: {args.output_dir}")
    print(f"Evaluation output: {args.eval_output_dir}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Ablation levels: {ablation_levels}")
    print(f"Use torchrun: {args.use_torchrun}")
    print(f"Master port: {args.master_port}")
    print(f"Skip training: {args.skip_training}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Collect only: {args.collect_only}")
    print("=" * 80)
    
    # Load base configurations
    try:
        base_config = load_config(args.base_config)
        print(f"✅ Loaded base training configuration from {args.base_config}")
    except Exception as e:
        print(f"❌ Failed to load base training configuration: {e}")
        return 1
    
    try:
        base_eval_config = load_config(args.base_eval_config)
        print(f"✅ Loaded base evaluation configuration from {args.base_eval_config}")
    except Exception as e:
        print(f"❌ Failed to load base evaluation configuration: {e}")
        return 1
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    
    # If collect_only, just collect and display results
    if args.collect_only:
        print("\n📋 Collecting existing results...")
        results = collect_results(args.eval_output_dir, ablation_levels)
        print_results_summary(results)
        return 0
    
    # Track results
    training_results = {}
    evaluation_results = {}
    
    # Phase 1: Training
    if not args.skip_training:
        print(f"\n{'='*60}")
        print("🚀 TRAINING PHASE")
        print(f"{'='*60}")
        
        for level in ablation_levels:
            print(f"\n{'='*40}")
            print(f"🎯 TRAINING ABLATION LEVEL {level}")
            print(f"{'='*40}")
            
            # Check if already exists
            level_output_dir = f"{args.output_dir}/level_{level}"
            if args.skip_existing and os.path.exists(level_output_dir):
                print(f"⏭️  Skipping level {level} (output directory exists)")
                training_results[level] = "skipped"
                continue
            
            # Create configuration for this level
            config = create_ablation_config(base_config, level, args.output_dir)
            
            # Save configuration
            config_path = f"{args.output_dir}/level_{level}_config.json"
            save_config(config, config_path)
            print(f"💾 Saved configuration to {config_path}")
            
            # Print ablation info
            ablation_info = {
                0: "Full C2C (baseline)",
                1: "No scalar weights (scalars=1.0)",
                2: "No gates (gates=1.0) + No scalar weights",
                3: "No target (source-only) + No gates + No scalar weights"
            }
            print(f"📋 Ablation description: {ablation_info.get(level, 'Unknown level')}")
            
            # Run training
            start_time = time.time()
            success = run_training(config_path, gpu_ids,
                                 args.use_torchrun, args.master_port)
            end_time = time.time()
            
            duration = end_time - start_time
            
            if success:
                print(f"✅ Level {level} training completed successfully in {duration:.2f}s")
                training_results[level] = "success"
            else:
                print(f"❌ Level {level} training failed after {duration:.2f}s")
                training_results[level] = "failed"
    else:
        print("⏭️  Skipping training phase")
    
    # Phase 2: Evaluation
    if not args.skip_evaluation:
        print(f"\n{'='*60}")
        print("📊 EVALUATION PHASE")
        print(f"{'='*60}")
        
        for level in ablation_levels:
            print(f"\n{'='*40}")
            print(f"🎯 EVALUATING ABLATION LEVEL {level}")
            print(f"{'='*40}")
            
            # Check if checkpoint exists
            if not check_checkpoint_exists(args.output_dir, level):
                print(f"❌ Checkpoint directory not found for level {level}")
                print(f"Expected: {args.output_dir}/level_{level}/final")
                evaluation_results[level] = "failed"
                continue
            
            # Create configuration for this level
            config = create_eval_config(base_eval_config, level, args.output_dir, args.eval_output_dir)
            
            # Save configuration
            config_path = f"{args.eval_output_dir}/level_{level}_eval_config.yaml"
            save_config(config, config_path)
            print(f"💾 Saved evaluation configuration to {config_path}")
            
            # Print ablation info
            ablation_info = {
                0: "Full C2C (baseline)",
                1: "No scalar weights (scalars=1.0)",
                2: "No gates (gates=1.0) + No scalar weights",
                3: "No target (source-only) + No gates + No scalar weights"
            }
            print(f"📋 Ablation description: {ablation_info.get(level, 'Unknown level')}")
            
            # Run evaluation
            start_time = time.time()
            success = run_evaluation(config_path, gpu_ids)
            end_time = time.time()
            
            duration = end_time - start_time
            
            if success:
                print(f"✅ Level {level} evaluation completed successfully in {duration:.2f}s")
                evaluation_results[level] = "success"
            else:
                print(f"❌ Level {level} evaluation failed after {duration:.2f}s")
                evaluation_results[level] = "failed"
    else:
        print("⏭️  Skipping evaluation phase")
    
    # Collect and display results
    if not args.skip_evaluation:
        print(f"\n{'='*60}")
        print("📋 COLLECTING RESULTS")
        print(f"{'='*60}")
        
        all_results = collect_results(args.eval_output_dir, ablation_levels)
        print_results_summary(all_results)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    
    if not args.skip_training:
        print("Training Results:")
        for level, status in training_results.items():
            status_emoji = {"success": "✅", "failed": "❌", "skipped": "⏭️"}
            print(f"  Level {level}: {status_emoji.get(status, '❓')} {status}")
    
    if not args.skip_evaluation:
        print("\nEvaluation Results:")
        for level, status in evaluation_results.items():
            status_emoji = {"success": "✅", "failed": "❌", "skipped": "⏭️"}
            print(f"  Level {level}: {status_emoji.get(status, '❓')} {status}")
    
    # Overall success
    training_success = sum(1 for status in training_results.values() if status == "success") if training_results else 0
    training_total = len(training_results) if training_results else 0
    eval_success = sum(1 for status in evaluation_results.values() if status == "success") if evaluation_results else 0
    eval_total = len(evaluation_results) if evaluation_results else 0
    
    print(f"\nOverall:")
    if training_total > 0:
        print(f"  Training: {training_success}/{training_total} levels completed successfully")
    if eval_total > 0:
        print(f"  Evaluation: {eval_success}/{eval_total} levels completed successfully")
    
    if (training_success == training_total or args.skip_training) and (eval_success == eval_total or args.skip_evaluation):
        print("🎉 Ablation study completed successfully!")
        return 0
    else:
        print("⚠️  Some phases failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())
