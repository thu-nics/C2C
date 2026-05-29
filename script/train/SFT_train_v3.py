"""
SFT training entrypoint for C2C V3 hidden-state projectors.

This reuses the existing SFT trainer but swaps in:
  - rosetta.model.wrapper_v3.RosettaModel
  - rosetta.model.projector_v3 projectors

Only the model setup differs: V3 projectors receive the sharer hidden size as
their source dimension instead of the sharer KV head dimension.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import script.train.SFT_train as sft
from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.model.projector_v3 import AllInOneProjector, create_projector, save_projector
from rosetta.model.wrapper_v3 import RosettaModel
from rosetta.train.model_utils import k_nearest_sources, last_aligned_sources
from rosetta.utils.evaluate import set_default_chat_template


_ORIGINAL_SETUP_MODELS = sft.setup_models


HIDDEN_STATE_PROJECTOR_TYPES = {
    "c2chiddenstateprojector",
    "c2cprojectorv3",
    "hiddenstatec2cprojector",
}


def _uses_source_hidden_states(projector_type: str) -> bool:
    return projector_type.lower() in HIDDEN_STATE_PROJECTOR_TYPES


def setup_models(
    model_config: Dict[str, Any],
    training_mode: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    if training_mode == "baseline":
        return _ORIGINAL_SETUP_MODELS(model_config, training_mode, device, dtype)

    slm_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    if slm_tokenizer.pad_token is None:
        slm_tokenizer.pad_token = slm_tokenizer.eos_token
        slm_tokenizer.pad_token_id = slm_tokenizer.eos_token_id
    set_default_chat_template(slm_tokenizer, model_config["base_model"])

    llm_tokenizer = None
    if model_config.get("is_do_alignment", False):
        llm_tokenizer = AutoTokenizer.from_pretrained(model_config["teacher_model"])
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        set_default_chat_template(llm_tokenizer, model_config["teacher_model"])

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=dtype,
        attn_implementation=model_config.get("attn_implementation", None),
    )

    if model_config["teacher_model"] == "google/gemma-3-1b-it":
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_config["teacher_model"],
            torch_dtype=dtype,
            attn_implementation=model_config.get("attn_implementation", None),
            sliding_window=4096,
        )
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_config["teacher_model"],
            torch_dtype=dtype,
            attn_implementation=model_config.get("attn_implementation", None),
        )

    base_dim = int(
        base_model.model.layers[0].self_attn.k_proj.out_features
        / base_model.config.num_key_value_heads
    )
    teacher_dim = int(
        teacher_model.model.layers[0].self_attn.k_proj.out_features
        / teacher_model.config.num_key_value_heads
    )
    base_num_heads = base_model.config.num_key_value_heads
    teacher_num_heads = teacher_model.config.num_key_value_heads
    slm_num_layers = base_model.config.num_hidden_layers
    llm_num_layers = teacher_model.config.num_hidden_layers

    projector_config = model_config["projector"]
    projector_type = projector_config["type"]
    projector_params = projector_config["params"].copy()
    projector_params["dtype"] = dtype

    uses_hidden = _uses_source_hidden_states(projector_type)
    source_projector_dim = teacher_model.config.hidden_size if uses_hidden else teacher_dim
    if uses_hidden:
        configured_hidden_dim = projector_params.get("hidden_dim")
        if configured_hidden_dim is None:
            projector_params["hidden_dim"] = source_projector_dim
        elif int(configured_hidden_dim) != int(source_projector_dim):
            raise ValueError(
                "C2C V3 skips the input down-projection, so projector.params.hidden_dim "
                f"must equal teacher hidden_size ({source_projector_dim}); "
                f"got {configured_hidden_dim}."
            )

    projector_list = []
    for _ in range(slm_num_layers):
        projector = create_projector(
            projector_type,
            source_dim=source_projector_dim,
            target_dim=base_dim,
            source_num_heads=teacher_num_heads,
            target_num_heads=base_num_heads,
            **projector_params,
        )
        projector_list.append(projector.to(device))

    rosetta_model = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,
        projector_list=projector_list,
        include_response=model_config.get("include_response", False),
        multi_source_fusion_mode=model_config.get("multi_source_fusion_mode", "sequential"),
    ).to(device).eval()

    k = int(model_config.get("num_source_layers_per_target", 1))
    if model_config["mapping"] == "last_aligned":
        source_target_mapping = last_aligned_sources(slm_num_layers, llm_num_layers, k)
    elif model_config["mapping"] == "k_nearest":
        source_target_mapping = k_nearest_sources(slm_num_layers, llm_num_layers, k)
    else:
        raise ValueError(f"Invalid mapping strategy: {model_config['mapping']}")
    print(f"Using {model_config['mapping']} mapping strategy (target: [sources])")

    for target_layer_idx, src_list in source_target_mapping.items():
        for source_layer_idx in src_list:
            rosetta_model.set_projector_config(
                source_model_idx=1,
                source_model_layer_idx=source_layer_idx,
                target_model_idx=0,
                target_model_layer_idx=target_layer_idx,
                projector_idx=target_layer_idx,
            )

    aligner = None
    if model_config.get("is_do_alignment", False):
        strategy = model_config.get("alignment_strategy", "first")
        aligner = TokenAligner(
            slm_tokenizer=slm_tokenizer,
            llm_tokenizer=llm_tokenizer,
            strategy=AlignmentStrategy(strategy),
        )

    return rosetta_model, slm_tokenizer, aligner, llm_tokenizer


sft.RosettaModel = RosettaModel
sft.create_projector = create_projector
sft.save_projector = save_projector
sft.AllInOneProjector = AllInOneProjector
sft.setup_models = setup_models


if __name__ == "__main__":
    sft.main()
