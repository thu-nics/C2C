#!/usr/bin/env python3
"""
Generate different configuration files for testing various projector types and freeze configurations.
"""

import json
import os
import itertools
from typing import Dict, Any, List

def load_base_config(config_path: str = "recipe/default_config.json") -> Dict[str, Any]:
    """Load the base configuration from the default config file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Default config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def generate_configs():
    """Generate all configuration combinations"""
    
    # Load base configuration from default config file
    base_config = load_base_config()
    
    # Define the parameter variations
    projector_configs = {
        "AdditiveProjector": {
            "type": "AdditiveProjector",
            "params": {
                "hidden_dim": 1024,
                "num_layers": 3,
                "dropout": 0.1,
                "activation": "gelu",
                "use_layer_norm": True,
                "init_weight": 0.1
            }
        },
        "MLPProjector": {
            "type": "MLPProjector",
            "params": {
                "hidden_dim": 1024,
                "num_layers": 3,
                "dropout": 0.1,
                "activation": "gelu",
                "use_layer_norm": True,
                "residual_connection": False
            }
        }
    }
    
    freeze_configs = {
        "freeze_teacher": ["teacher"],
        # "freeze_base": ["base"],
        # "freeze_projector": ["projector"],
        "freeze_base_teacher": ["base", "teacher"],
        # "freeze_base_projector": ["base", "projector"],
        # "freeze_teacher_projector": ["teacher", "projector"],
        # "freeze_none": []
    }
    
    # Create output directory
    output_dir = "recipe/experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all combinations
    config_files = []
    
    for projector_name, projector_config in projector_configs.items():
        for freeze_name, freeze_config in freeze_configs.items():
            # Create copy of base config
            config = json.loads(json.dumps(base_config))  # Deep copy
            
            # Update projector
            config["model"]["projector"] = projector_config
            
            # Update freeze configuration
            config["training"]["freeze"] = freeze_config
            
            # Create descriptive run name
            run_name = f"rosetta_{projector_name.lower()}_{freeze_name}"
            config["output"]["run_name"] = run_name
            
            # Create filename
            filename = f"{projector_name.lower()}_{freeze_name}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save config
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
            
            config_files.append(filepath)
            print(f"Generated: {filepath}")
    
    print(f"\nGenerated {len(config_files)} configuration files in {output_dir}/")
    return config_files

def generate_summary():
    """Generate a summary of all configurations"""
    output_dir = "recipe/experiments"
    summary_file = os.path.join(output_dir, "experiment_summary.txt")
    
    projector_types = ["AdditiveProjector", "MLPProjector"]
    freeze_options = [
        "freeze_teacher", "freeze_base", "freeze_projector", 
        "freeze_base_teacher", "freeze_base_projector", "freeze_teacher_projector",
        "freeze_none"
    ]
    
    with open(summary_file, 'w') as f:
        f.write("Experiment Configuration Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Projector Types:\n")
        for proj in projector_types:
            f.write(f"  - {proj}\n")
        f.write("\n")
        
        f.write("Freeze Configurations:\n")
        freeze_descriptions = {
            "freeze_teacher": "Only teacher model frozen",
            "freeze_base": "Only base model frozen",
            "freeze_projector": "Only projector frozen",
            "freeze_base_teacher": "Base and teacher models frozen",
            "freeze_base_projector": "Base model and projector frozen",
            "freeze_teacher_projector": "Teacher model and projector frozen",
            "freeze_none": "No components frozen"
        }
        
        for freeze, desc in freeze_descriptions.items():
            f.write(f"  - {freeze}: {desc}\n")
        f.write("\n")
        
        f.write("Total Experiments:\n")
        f.write(f"  {len(projector_types)} projectors × {len(freeze_options)} freeze configs = {len(projector_types) * len(freeze_options)} experiments\n\n")
        
        f.write("Generated Files:\n")
        for proj in projector_types:
            for freeze in freeze_options:
                filename = f"{proj.lower()}_{freeze}.json"
                f.write(f"  - {filename}\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    print("Generating experiment configurations...")
    config_files = generate_configs()
    generate_summary()
    print("\nDone!") 