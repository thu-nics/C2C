"""
Example inference using RosettaModel with Qwen3-0.6B and Qwen3-1.7B models and MLP projector
"""

import torch
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rosetta.model.projector import create_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.model.aligner import TokenAligner, AlignmentStrategy
from rosetta.model.projector import AllInOneProjector
from rosetta.train.dataset_adapters import generate_kv_cache_index
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.model.projector import load_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.model.aggregator import load_aggregator
from rosetta.utils.evaluate import set_default_chat_template
import re

def test_token_aligner(slm_tokenizer: AutoTokenizer, llm_tokenizer: AutoTokenizer):
    """Test the TokenAligner functionality
    Args:
        slm_tokenizer: SLM tokenizer
        llm_tokenizer: LLM tokenizer
    """
    print("\n" + "="*80)
    print("Testing TokenAligner")
    print("="*80)
    
    # Test with FIRST strategy
    aligner_first = TokenAligner(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        strategy=AlignmentStrategy.FIRST,
        verbose=True
    )
    
    # Test with LONGEST strategy
    aligner_longest = TokenAligner(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        strategy=AlignmentStrategy.LONGEST,
        verbose=True
    )
    
    # Test text samples
    test_texts = [
        "Hello world!",
        "The future of artificial intelligence is",
        "北京是中国的首都",  # Chinese text
        "🚀 Emojis and special characters!",
    ]
    
    for text in test_texts:
        print(f"\nTest text: '{text}'")
        print("-" * 40)
        
        # Test FIRST strategy
        print("\nFIRST Strategy:")
        aligner_first.visualize_alignment(text)
        
        # Test LONGEST strategy
        print("\nLONGEST Strategy:")
        aligner_longest.visualize_alignment(text)
    
    # Test alignment without visualization
    sample_text = "This is a test."
    slm_tokens, aligned_llm_tokens = aligner_first.align_sequence(sample_text)
    print(f"\nQuick alignment test for: '{sample_text}'")
    print(f"SLM tokens: {slm_tokens}")
    print(f"Aligned LLM tokens: {aligned_llm_tokens}")
    
    print("\n✅ TokenAligner test completed")
    

def run_inference_example(rosetta_model: RosettaModel, tokenizer: AutoTokenizer, prompt: str):
    """Run inference example with RosettaModel
    Args:
        rosetta_model: RosettaModel
        tokenizer: AutoTokenizer
        prompt: str
    """
    print("Running inference example...")

    device = rosetta_model.device
    
    # Prepare input


    input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"Input text: {input_text}")
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    print(f"Input tokens: {inputs['input_ids']}")
    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs['input_ids'].shape[1] - 1, 1).unsqueeze(0).to(device)
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
    kv_cache_index = [instruction_index, label_index]
    # slm_tokenizer = tokenizer
    # llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct") 
    # strategy = "first"
    # aligner = TokenAligner(slm_tokenizer=slm_tokenizer, llm_tokenizer=llm_tokenizer, strategy=AlignmentStrategy(strategy))
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # details = aligner.align_chat_messages(messages, add_generation_prompt=True, return_details=True)
    # slm_ids = torch.tensor(details['slm_ids_padded']).unsqueeze(0)
    # llm_ids = torch.tensor(details['llm_ids_padded']).unsqueeze(0)

    # slm_pad_mask = torch.tensor(details['slm_padding_mask']).unsqueeze(0)
    # llm_pad_mask = torch.tensor(details['llm_padding_mask']).unsqueeze(0)

    # slm_attention_mask = (~slm_pad_mask).float()
    # llm_attention_mask = (~llm_pad_mask).float()

    # message_mask = torch.tensor(details['message_mask'])
    # kv_cache_index = generate_kv_cache_index(slm_ids.shape[1], slm_ids.shape[1])
    # kv_cache_index[~message_mask] = torch.tensor([[-1,0]])

    # kv_idx = kv_cache_index
    # change_points = [0]
    # for i in range(1, kv_idx.size(0)):
    #     if not torch.equal(kv_idx[i], kv_idx[i - 1]):
    #         change_points.append(i)
    # change_points.append(kv_idx.size(0))

    # kv_cache_list = []

    # for i in range(len(change_points) - 1):
    #     start = change_points[i]
    #     end = change_points[i + 1]
    #     kv_cache_list.append(kv_idx[start:end, :].unsqueeze(0).to(device))
    # prefill_kv_cache_list = kv_cache_list[:-1]
    # print(f"Input prompt: '{prompt}'")
    # print(f"Input shape: {slm_ids.shape}")
    # print(f"Device: {device}")
    
    # slm_ids = slm_ids.to(device)
    # llm_ids = llm_ids.to(device)
    # slm_attention_mask = slm_attention_mask.to(device)
    # llm_attention_mask = llm_attention_mask.to(device)

    # Run inference
    # with torch.no_grad():
    #     # outputs = rosetta_model.forward(
    #     #     input_ids=[slm_ids, llm_ids],
    #     #     attention_mask=[slm_attention_mask, llm_attention_mask],
    #     #     kv_cache_index=kv_cache_list,
    #     #     position_ids=torch.arange(slm_ids.shape[1]).unsqueeze(0).to(device),
    #     #     use_cache=True,
    #     #     output_attentions=False,
    #     #     output_hidden_states=False,
    #     #     sample=False,
    #     # )
    #     outputs = rosetta_model(**inputs, kv_cache_index=kv_cache_index)
        
    #     # Get logits and generate next token
    #     logits = outputs.logits
    #     next_token_logits = logits[0, -1, :]
    #     next_token_id = torch.argmax(next_token_logits, dim=-1)
    #     next_token = tokenizer.decode(next_token_id)
        
    #     print(f"Output logits shape: {logits.shape}")
    #     print(f"Next predicted token: '{next_token}'")
    #     print("✅ Inference completed successfully")

    # Run generation
    with torch.no_grad():
        # outputs = rosetta_model.generate(
        #     prefill_kv_cache_index=prefill_kv_cache_list,
        #     input_ids=[slm_ids, llm_ids],
        #     attention_mask=[slm_attention_mask, llm_attention_mask],
        #     use_cache=True,
        #     output_attentions=False,
        #     output_hidden_states=False,
        #     max_new_tokens=256,
        #     do_sample=False,
        # )
        sampling_params = {
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.2,
            'max_new_tokens': 1024
        }
        outputs = rosetta_model.generate(**inputs, kv_cache_index=kv_cache_index, **sampling_params)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Rosetta output text: {output_text}")
    
    with torch.no_grad():
        sampling_params = {
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.2,
            'max_new_tokens': 1024
        }
        slm_model = rosetta_model.model_list[0]
        outputs = slm_model.generate(**inputs, **sampling_params)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"SLM output text: {output_text}")
    
    with torch.no_grad():
        sampling_params = {
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.2,
            'max_new_tokens': 1024
        }
        llm_model = rosetta_model.model_list[1]
        outputs = llm_model.generate(**inputs, **sampling_params)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"LLM output text: {output_text}")

def load_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
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
    
    # Load aggregators
    num_aggregators = len([f for f in os.listdir(checkpoint_dir) if re.match(r"aggregator_\d+\.pt", f)])
    aggregator_list = []
    for t in range(num_aggregators):
        json_cfg = os.path.join(checkpoint_dir, f"aggregator_{t}.json")
        agg_path = os.path.join(checkpoint_dir, f"aggregator_{t}.pt")
        agg = load_aggregator(json_cfg)
        if os.path.exists(agg_path):
            sd = torch.load(agg_path, map_location="cpu")
            agg.load_state_dict(sd, strict=False)
        agg = agg.to(device)
        aggregator_list.append(agg)
    
    # Initialize Rosetta model
    rosetta_model = RosettaModel(
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

def main():
    """Main function to run the inference example"""

    rosetta_model, slm_tokenizer = load_rosetta_model(
        model_config={
            "rosetta_config": {
                "base_model": "Qwen/Qwen3-0.6B",
                "teacher_model": "Qwen/Qwen3-4B",
                "checkpoints_dir": "local/checkpoints/0.6B_4B_general/final"
            }
        },
        eval_config={},
        device=torch.device("cuda")
    )
    
    # Test token aligner
    # test_token_aligner(slm_tokenizer, llm_tokenizer)
    
    # Run inference
    prompt = [{
        "role": "user",
        "content": "Accurately answer the following question:\n\nStatement 1 | If T: V -> W is a linear transformation and dim(V ) < dim(W) < 1, then T must be injective. Statement 2 | Let dim(V) = n and suppose that T: V -> V is linear. If T is injective, then it is a bijection.\n\nAre these statements correct? Let's think step by step and then answer the question starting with Answer:"
    }]
    run_inference_example(rosetta_model, slm_tokenizer, prompt)
    # run_inference_example(rosetta_model, slm_tokenizer, "从美国向北进入加拿大时，您会看到北星（北极星）越来越")


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, running...")
    main()
