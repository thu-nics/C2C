"""
Model setup utilities for RosettaModel training/evaluation
"""

import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import create_projector

"""
Mapping strategies
"""
def k_nearest_sources(num_target_layers: int, num_source_layers: int, k: int) -> Dict[int, List[int]]:
    """
    Compute a per-target mapping to K nearest source layers.

    Returns: Dict[target_idx, List[source_idx]] only for targets we map.
    Distances are computed by placing target and source layers uniformly in [0, 1]
    and sorting by absolute distance.
    """
    if num_target_layers <= 1:
        target_positions = [0.0]
    else:
        target_positions = [i / (num_target_layers - 1) for i in range(num_target_layers)]
    if num_source_layers <= 1:
        source_positions = [0.0]
    else:
        source_positions = [j / (num_source_layers - 1) for j in range(num_source_layers)]

    mapping: Dict[int, List[int]] = {}
    for t_idx, t_pos in enumerate(target_positions):
        sorted_src = sorted(range(num_source_layers), key=lambda j: abs(source_positions[j] - t_pos))
        chosen = sorted_src[:max(0, k)]
        if len(chosen) > 0:
            mapping[t_idx] = chosen
    return mapping


def last_aligned_sources(num_target_layers: int, num_source_layers: int, k: int = 1) -> Dict[int, List[int]]:
    """
    Return a per-target mapping that aligns the last target layer to the last
    source layer and walks toward the front.

    Returns: Dict[target_idx, List[source_idx]] only for targets we map. For each
    target t, we choose up to K sources anchored at the aligned index, preferring
    backward indices first then forward to satisfy K.

    Example (T=11, S=33): target 10 -> [32, 31, ...], target 9 -> [31, 30, ...]
    """
    mapping: Dict[int, List[int]] = {}
    if num_target_layers <= 0 or num_source_layers <= 0:
        return mapping

    # Align ends; offset >= 0 means extra source layers at the front
    offset = num_source_layers - num_target_layers

    def take_k_from(s0: int) -> List[int]:
        result: List[int] = []
        # Prefer moving backward from the anchor (last-to-front)
        for back in range(k):
            idx = s0 - back
            if 0 <= idx < num_source_layers:
                result.append(idx)
        # If not enough due to boundary, extend forward
        next_idx = s0 + 1
        while len(result) < k and next_idx < num_source_layers:
            result.append(next_idx)
            next_idx += 1
        return result

    for t in range(num_target_layers):
        s0 = offset + t
        # Clamp to valid range for edge cases (e.g., fewer source layers)
        if s0 < 0:
            s0 = 0
        elif s0 > num_source_layers - 1:
            s0 = num_source_layers - 1
        chosen = take_k_from(s0)
        if len(chosen) > 0:
            mapping[t] = chosen

    return mapping


def setup_models(model_config: Dict[str, Any], device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Setup RosettaModel with base model, teacher model, and projectors"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=dtype,
        device_map=device
    )
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_config["teacher_model"],
        torch_dtype=dtype,
        device_map=device
    )
    
    # Create projector
    projector_config = model_config["projector"]
    projector_params = projector_config["params"].copy()
    projector_params["dtype"] = dtype
    
    projector = create_projector(
        projector_config["type"],
        source_dim=teacher_model.config.head_dim,
        target_dim=base_model.config.head_dim,
        **projector_params
    )

    # Setup RosettaModel
    rosetta_model = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,
        projector_list=[projector]
    ).to(device)
    
    # Configure projector mappings
    num_layers_to_map = min(
        base_model.config.num_hidden_layers, 
        teacher_model.config.num_hidden_layers
    )
    
    for layer_idx in range(num_layers_to_map):
        rosetta_model.set_projector_config(
            source_model_idx=1,  # Teacher
            source_model_layer_idx=layer_idx,
            target_model_idx=0,  # Base
            target_model_layer_idx=layer_idx,
            projector_idx=0
        )
    
    return rosetta_model, tokenizer 