"""
Common evaluation utilities for benchmark tasks.

This module provides shared functions for model evaluation across different benchmarks
like MMLU-Redux and MMMLU.
"""

import re
import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.model.projector import load_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.model.oracle import OracleRosettaModel

def build_prompt(dataset: str, locale: str, question: str, choices: str, use_cot: bool, use_template: bool = True) -> str:
    """
    Build a localized prompt for a given dataset and locale.

    Currently supports:
    - dataset: "mmmlu"
      - locale: "SW_KE" (Swahili). Other locales fall back to English.

    Args:
        dataset: Dataset identifier (e.g., "mmmlu")
        locale: Locale/subject code (e.g., "SW_KE")
        question: Question text
        choices: Formatted choices string
        use_cot: Whether to include CoT instruction

    Returns:
        Localized prompt string
    """
    
        # Unified default English templates (shared by MMLU and MMMLU)
    if not use_cot:
        template = """Accurately answer the following question:

{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D".
- Do not include any explanations, additional text, or punctuation besides the answer.

The correct answer is"""

    else:
        template = """Accurately answer the following question:
                   
{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Let's think step by step and explain your reasoning briefly.
- Then give the final answer starting with The correct answer is"""

    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{choices}}", choices)

    if not use_template:
        prompt = question + "\n\nChoices:\n" + choices

    return prompt


def parse_answer(answer_str: str) -> List[str]:
    """
    Parse answer string to extract valid answer options.
    Converts digits 0/1/2/3 to letters A/B/C/D.
    
    Args:
        answer_str: String containing answer digits
        
    Returns:
        List of parsed answer letters
    """
    if not isinstance(answer_str, str):
        return []
    valid_digits = [c for c in answer_str if c in {'0','1','2','3'}]
    return sorted(list({
        chr(65 + int(d))  # 0->A, 1->B, 2->C, 3->D
        for d in valid_digits
    }))


def extract_answer_from_content(text: str) -> Optional[str]:
    """
    Extract answer from model output with robust multi-pattern matching.
    Supports multiple languages and response formats.
    
    Args:
        text: Model output text
        
    Returns:
        Extracted answer letter or None
    """
    text = text.strip()
    if not text:
        return None

    # Define multiple answer patterns for different languages and formats
    answer_patterns = [
        # English patterns
        r'Answer:\s*(.*)',
        r'answer:\s*(.*)',
        r'ANSWER:\s*(.*)',
        r'Your answer:\s*(.*)',
        r'your answer:\s*(.*)',
        r'YOUR ANSWER:\s*(.*)',
        r'The answer is\s*(.*)',
        r'the answer is\s*(.*)',
        r'THE ANSWER IS\s*(.*)',
        r'Correct answer is\s*(.*)',
        r'correct answer is\s*(.*)',
        r'Correct answer is:\s*(.*)',
        r'correct answer is:\s*(.*)',
        r'Correct answer:\s*(.*)',
        r'correct answer:\s*(.*)',
        r'CORRECT ANSWER:\s*(.*)',
        
        # Swahili patterns
        r'Jibu lako:\s*(.*)',
        r'jibu lako:\s*(.*)',
        r'JIBU LAKO:\s*(.*)',
        r'Jibu:\s*(.*)',
        r'jibu:\s*(.*)',
        r'JIBU:\s*(.*)',
        r'Jibu sahihi:\s*(.*)',
        r'jibu sahihi:\s*(.*)',
        r'JIBU SAHIHI:\s*(.*)',
        
        # Other common patterns
        r'Response:\s*(.*)',
        r'response:\s*(.*)',
        r'RESPONSE:\s*(.*)',
        r'Choice:\s*(.*)',
        r'choice:\s*(.*)',
        r'CHOICE:\s*(.*)',
        r'Option:\s*(.*)',
        r'option:\s*(.*)',
        r'OPTION:\s*(.*)',
    ]
    
    # 1. Try to match any of the answer patterns
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            # Search for first A-D letter in the matched part
            for char in answer_part:
                if char in {'A', 'B', 'C', 'D'}:
                    return char
    
    # 2. Look for standalone A-D letters that are likely answers
    # Prioritize letters at the end of text or with clear answer-like context
    standalone_patterns = [
        r'\b([A-D])(?:\s*[.,!?:)]?\s*$)',  # A-D at end of text with optional punctuation
        r'\b([A-D])(?:\s*[.,!?:)]\s)',     # A-D followed by punctuation and space
        r'(?:^|\s)([A-D])(?:\s*$)',        # A-D at start or with word boundary at end
    ]
    
    for pattern in standalone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Check if this looks like mathematical expressions rather than answers
            math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
            has_math = any(indicator in text for indicator in math_indicators)
            has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
            
            # If it has math indicators but no answer indicators, it's likely mathematical notation
            if has_math and not has_answer_indicators:
                continue  # Skip this match, try next pattern
            
            return matches[-1].upper()
    
    # 3. Fallback: find all A-D letters but be more selective
    all_letters = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    if all_letters:
        # Check if this looks like mathematical expressions rather than answers
        math_indicators = ['+', '-', '*', '/', '=', '^', 'x^', 'y^', 'z^', 'mod', 'sqrt', 'sin', 'cos', 'tan']
        has_math = any(indicator in text for indicator in math_indicators)
        has_answer_indicators = any(phrase in text.lower() for phrase in ['jibu', 'answer', 'choice', 'option', 'response', 'correct', 'sahihi'])
        
        # If it has math indicators but no answer indicators, it's likely mathematical notation
        if has_math and not has_answer_indicators:
            return None
        
        # Otherwise, return the last letter found
        return all_letters[-1].upper()
    
    # 3. Search backwards for any A-D letter as fallback
    for char in reversed(text):
        if char in {'A', 'B', 'C', 'D'}:
            return char

    return None


def apply_generation_config(model: Any, generation_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Apply generation configuration to a model and handle sampling parameters.
    
    This function applies the provided generation config to the model and removes
    sampling parameters (temperature, top_p, top_k, min_p) when do_sample=False
    to avoid warnings from the transformers library. If no config is provided,
    it defaults to greedy decoding with cleaned sampling parameters.
    
    Args:
        model: Model object with generation_config attribute
        generation_config: Optional generation configuration dictionary.
                          If None, defaults to greedy decoding (do_sample=False).
    """
    if not hasattr(model, 'generation_config'):
        return
    
    # If no config provided, default to greedy decoding
    if not generation_config:
        generation_config = {'do_sample': False}
    
    # Apply all configuration parameters
    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)
    
    # Disable sampling parameters if do_sample=False to avoid warnings
    # We set them to None instead of deleting, since some model code may
    # access these attributes unconditionally.
    if not generation_config.get('do_sample', True):
        sampling_params = ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty']
        for param in sampling_params:
            try:
                setattr(model.generation_config, param, None)
            except Exception:
                # If the backend does not allow setting, ignore silently
                pass


def set_default_chat_template(tokenizer, model_name: str):
    """
    Set default chat template for models without one.
    
    Args:
        tokenizer: Tokenizer object
        model_name: Name of the model
    """
    if tokenizer.chat_template is None:
        if "UlizaLlama3".lower() in model_name.lower():
            tokenizer.chat_template = (
                "{%- for message in messages %}"
                "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
                "{{- message['content'] }}"
                "{{- '<|eot_id|>' }}"
                "{%- endfor %}"
                "{%- if add_generation_prompt %}"
                "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{%- endif %}"
            )
        else:
            print(f"Model {model_name} has no chat template, setting default template...")
            default_template = """{% for message in messages %}{% if message['role'] == 'user' %}### Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}### Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}
    {% endif %}{% endfor %}{% if add_generation_prompt %}
    ### Assistant:{% endif %}"""
            tokenizer.chat_template = default_template
            print("Default chat template has been set.")
    else:
        print(f"Model {model_name} already has a chat template.")


def load_hf_model(model_name: str, device: torch.device, generation_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Load Hugging Face model and tokenizer.
    
    Args:
        model_name: Model name or path
        device: Device to load model on
        generation_config: Optional generation configuration to apply
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name),
        trust_remote_code=True,
        padding_side='left'
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check and set chat template
    set_default_chat_template(tokenizer, model_name)

    if model_name == "google/gemma-3-1b-it":
        torch._dynamo.config.cache_size_limit = 64
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            sliding_window=4096
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            torch_dtype=torch.bfloat16,
            device_map={"": device}
    ).eval()
    
    # Apply generation config
    apply_generation_config(model, generation_config)
    
    return model, tokenizer


def load_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
                      device: torch.device, generation_config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Load Rosetta model with projectors and aggregators.
    
    Args:
        model_config: Model configuration dict
        eval_config: Evaluation configuration dict
        device: Device to load model on
        generation_config: Optional generation configuration to apply
        
    Returns:
        Tuple of (rosetta_model, tokenizer)
    """
    # Prefer checkpoints_dir under model.rosetta_config; fall back to eval config for backward compatibility
    rosetta_config = model_config["rosetta_config"]
    slm_model_path = rosetta_config["base_model"]
    teacher_model_config = rosetta_config["teacher_model"]

    # Dict of models with list of checkpoints: {"model_name": "model_path", ...} + ckpt: ["ckpt1", "ckpt2"]
    
    llm_configs = []  # List of (model_path, checkpoint_dir) tuples
    
    if isinstance(teacher_model_config, str):
        # Single model - backward compatibility
        checkpoint_dir = rosetta_config.get("checkpoints_dir", eval_config.get("checkpoints_dir"))
        llm_configs.append((teacher_model_config, checkpoint_dir))
    
    elif isinstance(teacher_model_config, dict):
        # Format 4: Dict format with separate ckpt list
        # teacher_model: {"model1_name": "model1_path", "model2_name": "model2_path"}
        # ckpt: ["ckpt1_path", "ckpt2_path"]
        checkpoints_dir = rosetta_config.get("checkpoints_dir", [])
        model_items = list(teacher_model_config.items())
        
        if len(checkpoints_dir) != len(model_items):
            raise ValueError(f"Number of checkpoints ({len(checkpoints_dir)}) must match number of models ({len(model_items)})")
        
        for (model_name, model_path), ckpt_dir in zip(model_items, checkpoints_dir):
            llm_configs.append((model_path, ckpt_dir))

    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    set_default_chat_template(slm_tokenizer, slm_model_path)
    
    # Load SLM model
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # Apply generation config to SLM
    apply_generation_config(slm_model, generation_config)
    
    # Load LLM models
    llm_models = []
    for llm_model_path, _ in llm_configs:
        if llm_model_path == "google/gemma-3-1b-it":
            llm_model = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                sliding_window=4096
            ).eval()
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": device}
            ).eval()
        
        # Apply generation config to LLM
        apply_generation_config(llm_model, generation_config)
        llm_models.append(llm_model)
    
     # Load projectors and aggregators for each LLM from their respective checkpoint directories
    # Each checkpoint directory contains standard format: projector_{idx}.pt, aggregator_{idx}.pt
    projector_list = []
    num_llms = len(llm_models)
    
    # Track projector/aggregator offset for each LLM (for config index adjustment)
    projector_offsets = [0]
    aggregator_offsets = [0]
    
    for llm_idx, (_, checkpoint_dir) in enumerate(llm_configs):
        # Load projectors from this LLM's checkpoint directory
        # Standard naming: projector_{proj_idx}.pt / .json
        num_projectors = len([f for f in os.listdir(checkpoint_dir) 
                             if re.match(r"projector_\d+\.pt", f)])
        
        for proj_idx in range(num_projectors):
            json_cfg = os.path.join(checkpoint_dir, f"projector_{proj_idx}.json")
            proj = load_projector(json_cfg)
            proj = proj.to(device)
            pt_path = os.path.join(checkpoint_dir, f"projector_{proj_idx}.pt")
            if os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location=device)
                proj.load_state_dict(state_dict, strict=False)
            projector_list.append(proj)
        
        # Record offset for next LLM
        projector_offsets.append(len(projector_list))

    # Initialize Rosetta model
    # model_list: [slm_model, llm_model_1, llm_model_2, ...]
    model_list = [slm_model] + llm_models
    
    # Get multi-source fusion mode from config (default to "sequential" for backward compatibility)
    multi_source_fusion_mode = rosetta_config.get("multi_source_fusion_mode", "sequential")
    include_response = rosetta_config.get("include_response", False)
    
    rosetta_model = RosettaModel(
        model_list=model_list,
        base_model_idx=0,
        projector_list=projector_list,
        aggregator_list=[],
        include_response=include_response,
        multi_source_fusion_mode=multi_source_fusion_mode,
    ).to(device).eval()

    # Load projector/aggregator mapping configs from each LLM's checkpoint directory
    # Each directory has standard config files: projector_config.json, aggregator_config.json
    
    # Helper function to adjust config indices for flattened lists
    def adjust_config_indices(config_dict, proj_offset, agg_offset, actual_source_idx=None):
        """Adjust projector/aggregator indices in config dict by adding offsets.
        
        Args:
            config_dict: Original config dictionary
            proj_offset: Offset for projector indices (if not None)
            agg_offset: Offset for aggregator indices (if not None)
            actual_source_idx: If provided, remap all source_model_idx to this value
        """
        adjusted = {}
        for target_model_idx, sources in config_dict.items():
            adjusted[int(target_model_idx)] = {}
            for source_model_idx, layers in sources.items():
                # Use actual_source_idx if provided, otherwise keep original
                actual_src_idx = actual_source_idx if actual_source_idx is not None else int(source_model_idx)
                adjusted[int(target_model_idx)][actual_src_idx] = {}
                for target_layer_idx, mappings in layers.items():
                    adjusted_mappings = []
                    for source_layer_idx, idx in mappings:
                        # Adjust the projector/aggregator index
                        adjusted_idx = idx + proj_offset if proj_offset is not None else idx + agg_offset
                        adjusted_mappings.append((source_layer_idx, adjusted_idx))
                    adjusted[int(target_model_idx)][actual_src_idx][int(target_layer_idx)] = adjusted_mappings
        return adjusted
    
    # Load and merge configs from each LLM's checkpoint directory
    for llm_idx, (_, checkpoint_dir) in enumerate(llm_configs):
        proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
        agg_cfg_path = os.path.join(checkpoint_dir, "aggregator_config.json")
        
        # Actual source model index in model_list (llm_idx=0 -> model_list[1], llm_idx=1 -> model_list[2], etc.)
        actual_source_model_idx = llm_idx + 1
        
        # Load projector config
        if os.path.exists(proj_cfg_path):
            with open(proj_cfg_path, 'r') as f:
                config = json.load(f)
                # Adjust projector indices based on offset and set actual source_idx
                adjusted_config = adjust_config_indices(config, projector_offsets[llm_idx], None, actual_source_model_idx)
                # Merge into rosetta_model.projector_dict
                for target_idx, sources in adjusted_config.items():
                    if target_idx not in rosetta_model.projector_dict:
                        rosetta_model.projector_dict[target_idx] = {}
                    for source_idx, layers in sources.items():
                        if source_idx not in rosetta_model.projector_dict[target_idx]:
                            rosetta_model.projector_dict[target_idx][source_idx] = {}
                        rosetta_model.projector_dict[target_idx][source_idx].update(layers)

    return rosetta_model, slm_tokenizer


def load_oracle_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
                      device: torch.device) -> Tuple[Any, Any]:
    """
    Load Rosetta model with projectors and aggregators.
    
    Args:
        model_config: Model configuration dict
        eval_config: Evaluation configuration dict
        device: Device to load model on
        
    Returns:
        Tuple of (rosetta_model, tokenizer)
    """
    # Prefer checkpoints_dir under model.rosetta_config; fall back to eval config for backward compatibility
    rosetta_config = model_config["rosetta_config"]
    checkpoint_dir = rosetta_config.get("checkpoints_dir", eval_config.get("checkpoints_dir"))
    if checkpoint_dir is None:
        raise KeyError("checkpoints_dir must be provided under model.rosetta_config (preferred) or eval config (legacy)")
    slm_model_path = rosetta_config["base_model"]
    llm_model_path = rosetta_config["teacher_model"]

    # Load tokenizer
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    set_default_chat_template(slm_tokenizer, slm_model_path)
    
    # Load models
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # Load projectors
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    projector_list = []
    for t in range(num_projectors):
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg)
        proj = proj.to(device)
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)
    
    aggregator_list = []
    
    # Initialize Rosetta model
    rosetta_model = OracleRosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=projector_list,
        aggregator_list=aggregator_list,
    ).to(device).eval()

    # Load projector/aggregator mapping configs
    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    agg_cfg_path = os.path.join(checkpoint_dir, "aggregator_config.json")
    rosetta_model.load_projector_config(proj_cfg_path)
    rosetta_model.load_aggregator_config(agg_cfg_path)

    return rosetta_model, slm_tokenizer


def get_option_token_ids(tokenizer, num_options: int = 4) -> List[int]:
    """
    Get token IDs for options A, B, C, D (or more up to J).
    
    Args:
        tokenizer: Tokenizer object
        num_options: Number of options to get (default 4 for A-D, max 10 for A-J)
        
    Returns:
        List of token IDs for options
    """
    # Limit to maximum of 10 options (A-J)
    num_options = min(num_options, 10)
    option_ids = []
    for i in range(num_options):
        letter = chr(65 + i)  # A=65, B=66, etc.
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        option_ids.append(ids[0] if ids else tokenizer.eos_token_id)
    return option_ids

"""
Deprecated
"""

@torch.no_grad()
def generate_answer_with_logits(model, tokenizer, prompt: str, option_ids: List[int], 
                               device: torch.device, model_type: str = "hf") -> Tuple[str, np.ndarray]:
    """
    Generate answer using logits method.
    
    Args:
        model: Model object
        tokenizer: Tokenizer object
        prompt: Input prompt
        option_ids: Token IDs for options A, B, C, D
        device: Device to run on
        model_type: Type of model ("rosetta", "qwen", or "hf")
        
    Returns:
        Tuple of (predicted_answer, probabilities)
    """
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    # Try to apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False if model_type == "qwen" else None
        )
    except Exception as e:
        print(f"Failed to apply chat template for {model_type} model: {e}")
        text = f"### Human: {prompt}\n### Assistant:"
    
    text += "The correct answer is"
    input_ids = tokenizer(text, return_tensors="pt").to(device)['input_ids']
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    
    if model_type == "rosetta":
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(input_ids.shape[1]-1, 1).unsqueeze(0).to(device)
        response_index = torch.tensor([[-1, 0]], dtype=torch.long).unsqueeze(0)
        outputs = model.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            kv_cache_index=[instruction_index, response_index]
        )
    else:
        outputs = model(input_ids)
    
    logits = outputs.logits[0, -1]
    option_logits = torch.tensor([
        logits[option_ids[0]].item(),
        logits[option_ids[1]].item(),
        logits[option_ids[2]].item(),
        logits[option_ids[3]].item()
    ])
    
    probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
    pred = chr(65 + np.argmax(probs))
    return pred, probs


@torch.no_grad()
def generate_answer_with_generate(model, tokenizer, prompt: str, device: torch.device,
                                 model_type: str = "hf") -> Tuple[str, np.ndarray, int, int, str]:
    """
    Generate answer using text generation method.
    
    Args:
        model: Model object
        tokenizer: Tokenizer object
        prompt: Input prompt
        device: Device to run on
        model_type: Type of model ("rosetta" or "hf")
        
    Returns:
        Tuple of (predicted_answer, probabilities, input_length, generation_length, generated_text)
    """
    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception as e:
        print(f"Failed to apply chat template: {e}")
        text = f"### Human: {prompt}\n### Assistant:"

    # Prepare model input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generation parameters
    sampling_params = {
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'min_p': 0.0,
        'repetition_penalty': 1.2,
        'max_new_tokens': 1024
    }
    
    # Generate text
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        **sampling_params
    )
    
    # Parse output
    if isinstance(model, RosettaModel):
        generated_ids = outputs[0]
    else:
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
    
    # Extract answer
    pred = extract_answer_from_content(content)
    
    # Return uniform distribution for generate method
    probs = np.array([0.25, 0.25, 0.25, 0.25])

    input_length = inputs.input_ids.shape[1]
    gen_length = generated_ids.shape[0]

    return pred, probs, input_length, gen_length, content

