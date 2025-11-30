"""
Live Chat Example with C2C Models

Demonstrates the key RosettaModel interface for cache-to-cache communication.

Usage:
    python live_chat_example.py --checkpoint_dir path/to/checkpoint
    python live_chat_example.py --checkpoint_dir path/to/ckpt1 path/to/ckpt2  # multi-source
"""

import argparse
import torch
import json, yaml, re, os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from rosetta.utils.evaluate import set_default_chat_template
from rosetta.utils.core import all_sharers_mask, format_sharer_mask
from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import load_projector

def load_model(checkpoint_dirs: list, subfolder: str = "final", device: str = "cuda:0"):
    """
    Load RosettaModel from checkpoint directories.
    Supports both single-source and multi-source modes.
    """
    
    device = torch.device(device)
    
    # Read config from each checkpoint
    configs = []
    for ckpt_dir in checkpoint_dirs:
        with open(Path(ckpt_dir) / "config.json") as f:
            configs.append(yaml.safe_load(f))
    
    base_model = configs[0]["model"]["base_model"] # assume all checkpoints have the same base model
    teacher_models = [cfg["model"]["teacher_model"] for cfg in configs]
    
    print(f"Base model: {base_model}")
    print(f"Teacher models: {teacher_models}")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    set_default_chat_template(tokenizer, base_model)
    
    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    
    # Load teacher models
    sharer_llms = [
        AutoModelForCausalLM.from_pretrained(
            tm, torch_dtype=torch.bfloat16, device_map={"": device}
        ).eval()
        for tm in teacher_models
    ]
    
    # Load projectors from each checkpoint
    projector_list = []
    projector_offsets = [0]
    
    for ckpt_dir in checkpoint_dirs:
        proj_dir = Path(ckpt_dir) / subfolder
        num_proj = len([f for f in os.listdir(proj_dir) if re.match(r"projector_\d+\.pt", f)])
        
        for i in range(num_proj):
            proj = load_projector(str(proj_dir / f"projector_{i}.json")).to(device)
            proj.load_state_dict(torch.load(proj_dir / f"projector_{i}.pt", map_location=device), strict=False)
            projector_list.append(proj)
        projector_offsets.append(len(projector_list))
    
    # Create RosettaModel
    rosetta_model = RosettaModel(
        model_list=[base_llm] + sharer_llms,
        base_model_idx=0,
        projector_list=projector_list,
    ).to(device).eval()
    
    # Load projector configs
    for llm_idx, ckpt_dir in enumerate(checkpoint_dirs):
        cfg_path = Path(ckpt_dir) / subfolder / "projector_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            # Adjust indices and merge into projector_dict
            for tgt_idx, sources in cfg.items():
                tgt_idx = int(tgt_idx)
                if tgt_idx not in rosetta_model.projector_dict:
                    rosetta_model.projector_dict[tgt_idx] = {}
                src_idx = llm_idx + 1  # actual source model index
                rosetta_model.projector_dict[tgt_idx][src_idx] = {
                    int(layer): [(sl, pi + projector_offsets[llm_idx]) for sl, pi in mappings]
                    for layer, mappings in list(sources.values())[0].items()
                }
    
    return rosetta_model, tokenizer, len(teacher_models)


def generate(model, tokenizer, prompt: str, num_sharers: int, device):
    """
    Generate response using RosettaModel.
    
    Key interface: kv_cache_index[i][0][0][0] controls sharer selection:
        - -1: no projection (receiver only)
        - >0: bitmask (1=sharer1, 2=sharer2, 3=both, 7=all three)
    """
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Create kv_cache_index with bitmask (use all sharers)
    sharer_mask = all_sharers_mask(num_sharers)
    seq_len = inputs.input_ids.shape[1]
    
    # instruction_index: apply C2C projection for prompt tokens
    # label_index: -1 means no projection for generation
    instruction_index = torch.tensor([sharer_mask, 0]).repeat(seq_len - 1, 1).unsqueeze(0).to(device)
    label_index = torch.tensor([[-1, 0]]).unsqueeze(0).to(device)
    
    print(f"  Using {format_sharer_mask(sharer_mask)} for prompt encoding")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            kv_cache_index=[instruction_index, label_index],
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            max_new_tokens=256,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description='Live Chat with C2C Models')
    parser.add_argument("--checkpoint_dir", type=str, nargs="+", required=True, help="Checkpoint directory(s)")
    parser.add_argument("--subfolder", type=str, default="final")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    checkpoint_dirs = args.checkpoint_dir
    
    # Load model
    device = torch.device(args.device)
    model, tokenizer, num_sharers = load_model(checkpoint_dirs, args.subfolder, args.device)
    print(f"Loaded {num_sharers} sharer(s). Type 'q' to quit.\n")
    
    # Chat loop
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        if not user_input:
            continue
        
        response = generate(model, tokenizer, user_input, num_sharers, device)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
