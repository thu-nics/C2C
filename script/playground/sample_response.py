import argparse
import re
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None 

from rosetta.model.projector import create_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.train.dataset_adapters import MMLUChatDataset, ChatDataset

def load_qwen_model(model_name):
    """加载Qwen模型"""
    print(f"Loading Qwen model: {model_name}")
    model_path = "Qwen/" + model_name
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side='left'
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
    )
    
    return model, tokenizer

def load_rosetta_model():
    """加载Rosetta模型"""

    slm_model_path = "Qwen/Qwen3-0.6B"
    llm_model_path = "Qwen/Qwen3-4B"
    
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map = 'cuda'
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_path),
        torch_dtype=torch.bfloat16,
        device_map = 'cuda'
    ).eval()
    
    # 创建投影器
    projector_config = {
            "type": "AdditiveProjector",
            "params": {
                "hidden_dim": 1024,
                "num_layers": 3,
                "dropout": 0.1,
                "activation": "gelu",
                "use_layer_norm": True,
                "init_weight": 0.1
            }
        }
    projector_params = projector_config["params"].copy()
    projector_params["dtype"] = torch.bfloat16
    projector = create_projector(
        projector_config["type"],
        source_dim=llm_model.config.head_dim,
        target_dim=slm_model.config.head_dim,
        **projector_params
    )
    
    # 初始化Rosetta模型
    rosetta_model = RosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=[projector]
    ).to('cuda').eval()
    
    # 导入projector权重
    projector_weight_path = "local/checkpoints/4b_21-27_cot_1e-4/final/projector_0.pt"
    rosetta_model.projector_list[0].load_state_dict(torch.load(projector_weight_path, map_location='cpu'))
    layer_offset = llm_model.config.num_hidden_layers - slm_model.config.num_hidden_layers
    # layer_offset = 0

    # 配置投影器映射
    for layer_idx in range(21, 28):
        rosetta_model.set_projector_config(
            source_model_idx=1,  # Teacher model
            source_model_layer_idx=layer_idx + layer_offset,
            target_model_idx=0,  # Base model
            target_model_layer_idx=layer_idx,
            projector_idx=0
        )
    
    return rosetta_model, slm_tokenizer

import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached, running...")

rosetta_model, rosetta_tokenizer = load_rosetta_model()
slm_model, slm_tokenizer = load_qwen_model("Qwen3-0.6B")
llm_model, llm_tokenizer = load_qwen_model("Qwen3-4B")

instruct_ds = MMLUChatDataset(split="validation", num_samples=None)

sampling_params = {
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 20,
    'min_p': 0.0,
    'repetition_penalty': 1.1,
    'max_new_tokens': 256
}
    
correct = 0
total = 0

rosetta_model.eval()
rosetta_model.cuda()  # 如果你在GPU上运行
slm_model.eval()
slm_model.cuda()  # 如果你在GPU上运行
llm_model.eval()
llm_model.cuda()  # 如果你在GPU上运行

with open("analysis/venn_regions.json", "r") as f:
    results = json.load(f)

idx = results["rosetta_llm"]

for i in idx:
    sample = instruct_ds[int(i)]
    sample[0]['content'] += "\nYou should first give a short explanation and then output the final answer in the format 'The correct answer is ...'. Don't output the answer directly. Don't give a very long explanation, just a few sentences is enough."
    # sample[0]['content'] += "Give your answer in the format: 'The correct answer is A/B/C/D'. You should only output 'The correct answer is ...', without any additional text."
    # 用三个模型分别构造并且输出回答
    instruction_rosetta = rosetta_tokenizer.apply_chat_template(
        sample[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_rosetta = rosetta_tokenizer(instruction_rosetta, add_special_tokens=False)
    with torch.no_grad():
        output_ids = rosetta_model.generate(
            input_ids=torch.tensor(input_rosetta["input_ids"]).unsqueeze(0).cuda(),
            attention_mask=torch.tensor(input_rosetta["attention_mask"]).unsqueeze(0).cuda(),
            **sampling_params
        )[0]
    full_output_rosetta = rosetta_tokenizer.decode(output_ids[len(input_rosetta['input_ids']):], skip_special_tokens=True)

    instruction_slm = slm_tokenizer.apply_chat_template(
        sample[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_slm = slm_tokenizer(instruction_slm, add_special_tokens=False)
    with torch.no_grad():
        output_ids_slm = slm_model.generate(
            input_ids=torch.tensor(input_slm["input_ids"]).unsqueeze(0).cuda(),
            attention_mask=torch.tensor(input_slm["attention_mask"]).unsqueeze(0).cuda(),
            **sampling_params
        )[0]
    full_output_slm = slm_tokenizer.decode(output_ids_slm[len(input_slm['input_ids']):], skip_special_tokens=True)

    instruction_llm = llm_tokenizer.apply_chat_template(
        sample[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_llm = llm_tokenizer(instruction_llm, add_special_tokens=False)
    with torch.no_grad():
        output_ids_llm = llm_model.generate(
            input_ids=torch.tensor(input_llm["input_ids"]).unsqueeze(0).cuda(),
            attention_mask=torch.tensor(input_llm["attention_mask"]).unsqueeze(0).cuda(),
            **sampling_params
        )[0]
    full_output_llm = llm_tokenizer.decode(output_ids_llm[len(input_llm['input_ids']):], skip_special_tokens=True)
