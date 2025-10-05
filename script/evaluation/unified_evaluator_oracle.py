"""
Unified Evaluation Script for Multiple Benchmarks

This script provides a unified interface for evaluating models on various benchmarks
including MMLU-Redux and MMMLU. It supports multi-GPU parallel evaluation and 
multiple answer generation methods.

Usage:
    python unified_evaluator.py --config eval_recipe/config.yaml
"""

import argparse
import os
import json
import yaml
import csv
import torch
import torch.multiprocessing as mp
from collections import defaultdict
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from datasets import load_dataset
from datetime import datetime

from rosetta.utils.evaluate import (
    extract_answer_from_content,
    load_hf_model,
    load_oracle_rosetta_model,
    get_option_token_ids,
    build_prompt
)
from rosetta.model.wrapper import OracleRosettaModel
from rosetta.model.aligner import TokenAligner, AlignmentStrategy
from rosetta.train.dataset_adapters import generate_kv_cache_index
from transformers import AutoTokenizer
from rosetta.utils.evaluate import set_default_chat_template


# Dataset-specific configurations
DATASET_CONFIGS = {
    "mmlu-redux": {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "test_split": "test",
        "subjects": [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
            'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
            'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
        ],
        "subcategories": {
            "abstract_algebra": ["math"],
            "anatomy": ["health"],
            "astronomy": ["physics"],
            "business_ethics": ["business"],
            "clinical_knowledge": ["health"],
            "college_biology": ["biology"],
            "college_chemistry": ["chemistry"],
            "college_computer_science": ["computer science"],
            "college_mathematics": ["math"],
            "college_medicine": ["health"],
            "college_physics": ["physics"],
            "computer_security": ["computer science"],
            "conceptual_physics": ["physics"],
            "econometrics": ["economics"],
            "electrical_engineering": ["engineering"],
            "elementary_mathematics": ["math"],
            "formal_logic": ["philosophy"],
            "global_facts": ["other"],
            "high_school_biology": ["biology"],
            "high_school_chemistry": ["chemistry"],
            "high_school_computer_science": ["computer science"],
            "high_school_european_history": ["history"],
            "high_school_geography": ["geography"],
            "high_school_government_and_politics": ["politics"],
            "high_school_macroeconomics": ["economics"],
            "high_school_mathematics": ["math"],
            "high_school_microeconomics": ["economics"],
            "high_school_physics": ["physics"],
            "high_school_psychology": ["psychology"],
            "high_school_statistics": ["math"],
            "high_school_us_history": ["history"],
            "high_school_world_history": ["history"],
            "human_aging": ["health"],
            "human_sexuality": ["culture"],
            "international_law": ["law"],
            "jurisprudence": ["law"],
            "logical_fallacies": ["philosophy"],
            "machine_learning": ["computer science"],
            "management": ["business"],
            "marketing": ["business"],
            "medical_genetics": ["health"],
            "miscellaneous": ["other"],
            "moral_disputes": ["philosophy"],
            "moral_scenarios": ["philosophy"],
            "nutrition": ["health"],
            "philosophy": ["philosophy"],
            "prehistory": ["history"],
            "professional_accounting": ["other"],
            "professional_law": ["law"],
            "professional_medicine": ["health"],
            "professional_psychology": ["psychology"],
            "public_relations": ["politics"],
            "security_studies": ["politics"],
            "sociology": ["culture"],
            "us_foreign_policy": ["politics"],
            "virology": ["health"],
            "world_religions": ["philosophy"]
        },
        "categories": {
            "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
            "humanities": ["history", "philosophy", "law"],
            "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
            "other (business, health, misc.)": ["other", "business", "health"]
        }
    },
    "mmmlu": {
        "dataset_name": "openai/MMMLU",
        "test_split": "test",
        "subjects": [
            'AR_XY', 'BN_BD', 'DE_DE', 'ES_LA', 'FR_FR', 'HI_IN', 'ID_ID',
            'IT_IT', 'JA_JP', 'KO_KR', 'PT_BR', 'SW_KE', 'YO_NG', 'ZH_CN'
        ],
        "subcategories": {},  # MMMLU doesn't have subcategories
        "categories": {}  # MMMLU doesn't have categories
    }
}


class UnifiedEvaluator:
    """Unified evaluator for multiple benchmark datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.model_config = config["model"]
        self.output_config = config["output"]
        self.eval_config = config["eval"]
        self.dataset_name = self.eval_config.get("dataset", "mmlu-redux")
        
        # Get dataset-specific configuration
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        self.dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        # Setup output directory
        self.output_dir = Path(self.output_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating on dataset: {self.dataset_name}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Requested GPU IDs: {self.eval_config['gpu_ids']}")
        print(f"Answer method: {self.eval_config['answer_method']}")
    
    def format_example(self, example: Dict[str, Any], use_cot: bool = True) -> str:
        """
        Format an example into a prompt.
        
        Args:
            example: Example dictionary
            use_cot: Whether to use chain-of-thought prompting
            
        Returns:
            Formatted prompt string
        """
        if self.dataset_name == "mmmlu":
            return self._format_mmmlu_example(example, use_cot)
        elif self.dataset_name == "mmlu-redux":
            return self._format_mmlu_redux_example(example, use_cot)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _format_mmmlu_example(self, example: Dict[str, Any], use_cot: bool, subject: Optional[str] = None) -> str:
        """Format MMMLU example."""
        question_text = example['Question']
        choices = ""
        for i, choice_key in enumerate(['A', 'B', 'C', 'D']):
            if choice_key in example:
                choices += f"{choice_key}. {example[choice_key]}\n"

        # Localized prompt by subject (e.g., SW_KE uses Swahili). Fallback to English otherwise.
        prompt = build_prompt(
            dataset="mmmlu",
            locale=subject or "",
            question=question_text,
            choices=choices,
            use_cot=use_cot
        )
        return prompt
    
    def _format_mmlu_redux_example(self, example: Dict[str, Any], use_cot: bool) -> str:
        """Format MMLU-Redux example using unified prompt builder."""
        # Build choices string (A-D)
        choices = ""
        for i, choice in enumerate(example['choices']):
            choices += f"{chr(65+i)}. {choice}\n"

        # Use shared prompt builder for consistency with MMMLU
        prompt = build_prompt(
            dataset="mmlu-redux",
            locale="",
            question=example['question'],
            choices=choices,
            use_cot=use_cot
        )
        return prompt
    
    def parse_answer(self, example: Dict[str, Any]) -> Optional[str]:
        """
        Parse the correct answer from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            Correct answer letter or None
        """
        if self.dataset_name == "mmmlu":
            answer_key = example.get('Answer')
            if answer_key is None:
                return None
            
            # Convert various answer formats to letter
            if isinstance(answer_key, int):
                return chr(65 + answer_key)  # 0->A, 1->B, 2->C, 3->D
            elif isinstance(answer_key, str) and answer_key in ['0', '1', '2', '3']:
                return chr(65 + int(answer_key))
            elif isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None
        
        elif self.dataset_name == "mmlu-redux":  # mmlu-redux
            error_type = example.get('error_type', '')
            if error_type in ['no_correct_answer', 'expert']:
                return None
            
            if error_type == 'wrong_groundtruth':
                if example.get('correct_answer') is not None:
                    answer = example['correct_answer']
                    if answer >= '0' and answer <= '3':
                        answer_num = int(answer)
                    else:
                        answer_num = ord(answer) - ord('A')
                else:
                    answer_num = int(example['answer'])
            else:
                answer_num = int(example['answer'])
            
            return chr(65 + answer_num) if answer_num is not None else None
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def prepare_model_inputs(self, prompt: str, tokenizer, device: torch.device,
                              model_type: str, llm_tokenizer: Optional[Any],
                              answer_method: str):
        """
        Prepare model inputs (input_ids, attention_mask, position_ids, kv_cache_index) for
        both HF and Rosetta models, separated from the generation stage.

        Returns a dict with keys:
        - input_ids
        - attention_mask
        - position_ids
        - kv_cache_index
        - printable_text (str): chat-formatted input text for logging
        """
        messages = [{"role": "user", "content": prompt}]

        use_aligner = (model_type == "rosetta") and (llm_tokenizer is not None)

        # Build chat-formatted text
        if not use_aligner:
            if answer_method == 'logits':
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                text += "The correct answer is"
                response_length = tokenizer("The correct answer is", add_special_tokens=False).input_ids.__len__()
            else: # generate
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                response_length = 0
            # Default HF/Qwen path (and Rosetta generate path)
            tokenized = tokenizer(text, return_tensors="pt").to(device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            outputs = {
                "inputs": {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                },
                "printable_text": text
            }

            if model_type == "rosetta":
                # TODO: noqa for using 4 as fixed number
                full_length = input_ids.shape[1]
                response_length = 0
                instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(full_length - response_length, 1).unsqueeze(0).to(device)
                if response_length > 0:
                    response_index = torch.tensor([-1, 0], dtype=torch.long).repeat(response_length, 1).unsqueeze(0).to(device)
                    kv_cache_list = [instruction_index, response_index]
                else:
                    kv_cache_list = [instruction_index]

                if attention_mask is None:
                    outputs['inputs']["position_ids"] = torch.arange(input_ids.shape[-1], dtype=torch.long).unsqueeze(0).to("cuda")
                else:
                    outputs['inputs']["position_ids"] = attention_mask.long().cumsum(-1) - 1
                outputs['inputs']['kv_cache_index'] = kv_cache_list
            
        # Rosetta logits path with alignment (dual tokenizers)
        if use_aligner:
            alignment_strategy = self.model_config["rosetta_config"].get("alignment_strategy", "prefix")
            aligner = TokenAligner(
                slm_tokenizer=tokenizer,
                llm_tokenizer=llm_tokenizer,
                strategy=AlignmentStrategy(alignment_strategy)
            )

            if answer_method == 'logits':
                messages.append({"role": "assistant", "content": "The correct answer is"})
                remove_last_surfix = True
                add_generation_prompt = False
            else: # generate
                remove_last_surfix = False
                add_generation_prompt = True

            details = aligner.align_chat_messages(
                messages,
                add_generation_prompt=add_generation_prompt,
                return_details=True,
                enable_thinking=False,
                remove_last_surfix=remove_last_surfix
            )

            slm_ids = torch.tensor(details['slm_ids_padded']).unsqueeze(0).to(device)
            llm_ids = torch.tensor(details['llm_ids_padded']).unsqueeze(0).to(device)

            assert slm_ids.shape == llm_ids.shape, f"SLM and LLM input lengths do not match: {slm_ids.shape} vs {llm_ids.shape}"

            slm_pad_mask = torch.tensor(details['slm_padding_mask']).unsqueeze(0)
            llm_pad_mask = torch.tensor(details['llm_padding_mask']).unsqueeze(0)

            slm_attention_mask = (~slm_pad_mask).float()
            llm_attention_mask = (~llm_pad_mask).float()

            message_mask = torch.tensor(details['message_mask'])
            # TODO: support adding response
            
            # Create kv_cache_index and split by message_mask transitions in one pass
            kv_cache_list = []
            start = 0
            current_value = message_mask[0].item()
            
            for j in range(1, len(message_mask)):
                if message_mask[j] != message_mask[j - 1]:
                    # Found a change point, create segment for previous section
                    segment_length = j - start
                    if current_value:
                        segment = torch.tensor([1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
                    else:
                        segment = torch.tensor([-1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
                    kv_cache_list.append(segment)
                    
                    start = j
                    current_value = message_mask[j].item()
            
            # Handle the last segment
            segment_length = len(message_mask) - start
            if current_value:
                segment = torch.tensor([1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
            else:
                segment = torch.tensor([-1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
            kv_cache_list.append(segment)

            input_ids = [slm_ids, llm_ids]
            attention_mask = [slm_attention_mask.to(device), llm_attention_mask.to(device)]
            position_ids = torch.arange(slm_ids.shape[1]).unsqueeze(0).to(device)

            outputs = {
                "inputs": {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "kv_cache_index": kv_cache_list,
                },
                "printable_text": (details["slm_text"], details["llm_text"])
            }

        return outputs

    @torch.no_grad()
    def evaluate_subject(self, subject: str, model, tokenizer, device: torch.device, 
                        model_type: str = "hf", llm_tokenizer: Optional[Any] = None) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], List, List]:
        """
        Evaluate model on a specific subject.
        
        Args:
            subject: Subject name
            model: Model object
            tokenizer: Tokenizer object
            device: Device to run on
            model_type: Type of model
            
        Returns:
            Tuple of (correctness_array, accuracy, probabilities, length_stats, cot_logs)
        """
        # Load dataset
        dataset = load_dataset(self.dataset_config["dataset_name"], subject)
        test_data = dataset[self.dataset_config["test_split"]]
        
        # Get option token IDs
        option_ids = get_option_token_ids(tokenizer)
        
        cors = []
        all_probs = []
        length_stats = []
        cot_logs = []
        total_count = 0
        skip_count = 0
        printed_example = False
        
        # Sampling configuration
        sample_interval = self.eval_config.get("sample_interval", 1)
        sample_indices = list(range(0, len(test_data), sample_interval))
        limit = self.eval_config.get("limit", None)
        if isinstance(limit, int) and limit > 0:
            # Use first N indices
            sample_indices = sample_indices[:limit]
        elif isinstance(limit, (list, tuple)) and len(limit) == 2:
            # Treat as [start, end) range on original indices
            start, end = limit
            start = 0 if start is None else int(start)
            end = len(test_data) if end is None else int(end)
            sample_indices = [i for i in sample_indices if start <= i < end]
        
        for i in tqdm(sample_indices, desc=f"Evaluating {subject} ({self.eval_config['answer_method']})"):
            try:
                example = test_data[i]
                true_answer = self.parse_answer(example)
                
                if true_answer is None:
                    skip_count += 1
                    continue
                
                # Format prompt (pass subject for locale-aware templates)
                if self.dataset_name == "mmmlu":
                    prompt = self._format_mmmlu_example(example, use_cot=self.eval_config["use_cot"], subject=subject)
                elif self.dataset_name == "mmlu-redux":
                    prompt = self._format_mmlu_redux_example(example, use_cot=self.eval_config["use_cot"])
                else:
                    raise ValueError(f"Unknown dataset: {self.dataset_name}")
                
                # Prepare the inputs (separated from generation)
                prepared = self.prepare_model_inputs(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    device=device,
                    model_type=model_type,
                    llm_tokenizer=llm_tokenizer,
                    answer_method=self.eval_config["answer_method"]
                )

                # Generate answer
                if self.eval_config["answer_method"] == 'logits':
                    # Forward for logits
                    outputs = model.forward(**prepared['inputs'], identifier=i, subject=subject)

                    logits = outputs.logits[0, -1]
                    option_logits = torch.tensor([
                        logits[option_ids[0]].item(),
                        logits[option_ids[1]].item(),
                        logits[option_ids[2]].item(),
                        logits[option_ids[3]].item()
                    ])
                    probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
                    pred = chr(65 + np.argmax(probs))

                    # No CoT generation in logits mode
                    input_length, gen_length = None, None
                    cot_pred, cot_input_len, cot_gen_len, cot_text = None, None, None, None
                elif self.eval_config["answer_method"] == "generate":  # generate
                    sampling_params = {
                        'do_sample': True,
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 20,
                        'min_p': 0.0,
                        'repetition_penalty': 1.2,
                        'max_new_tokens': 1024
                    }

                    inputs = prepared['inputs']
                    inputs.update(sampling_params)
                    outputs = model.generate(**inputs)

                    if isinstance(model, OracleRosettaModel):
                        generated_ids = outputs[0]
                        if isinstance(prepared["inputs"]["input_ids"], list):
                            input_length = prepared["inputs"]["input_ids"][0].shape[1]
                        else:
                            input_length = prepared["inputs"]["input_ids"].shape[1]
                    else:
                        generated_ids = outputs[0][prepared["input_ids"].shape[1]:]
                        input_length = prepared["input_ids"].shape[1]
                    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")

                    pred = extract_answer_from_content(content)
                    probs = np.array([0.25, 0.25, 0.25, 0.25])
                    gen_length = generated_ids.shape[0]
                    cot_text = content
                    cot_pred = pred
                    cot_input_len, cot_gen_len = input_length, gen_length
                else:
                    raise ValueError(f"Unknown answer method: {self.eval_config['answer_method']}")
                    
                # Print one example of chat-formatted input (and output if generation) per subject
                if not printed_example:
                    try:
                        text = prepared.get("printable_text", "")
                        print("\n================ Example IO ({}) ================".format(subject))
                        if isinstance(text, (tuple, list)):
                            try:
                                slm_text, llm_text = text
                                print("[Input with chat template - SLM]:\n" + str(slm_text))
                                print("[Input with chat template - LLM]:\n" + str(llm_text))
                            except Exception:
                                print("[Input with chat template]:\n" + str(text))
                        else:
                            print("[Input with chat template]:\n" + str(text))
                        if self.eval_config["answer_method"] == 'generate' and cot_text is not None:
                            print("\n[Generated output]:\n" + str(cot_text))
                        print("================ End Example IO ================\n")
                    except Exception as e:
                        print(f"Failed to print example IO for {subject}: {e}")
                    finally:
                        printed_example = True
                
                # Check correctness
                is_correct = (pred == true_answer) if pred else False
                cors.append(is_correct)
                all_probs.append(probs)
                
                # Collect length statistics
                if self.eval_config["answer_method"] == 'generate' and input_length is not None and gen_length is not None:
                    length_ratio = gen_length / input_length if input_length > 0 else 0
                    length_stats.append({
                        'subject': subject,
                        'question_id': i,
                        'input_length': input_length,
                        'gen_length': gen_length,
                        'length_ratio': length_ratio,
                        'is_correct': is_correct,
                        'pred': pred,
                        'true_answer': true_answer
                    })
                
                # Collect CoT logs
                cot_log_entry = {
                    'subject': subject,
                    'question_id': i,
                    'true_answer': true_answer,
                    'pred': pred,
                    'is_correct': is_correct,
                    'answer_method': self.eval_config.get('answer_method', ''),
                    'cot_pred': cot_pred,
                    'cot_input_length': cot_input_len,
                    'cot_gen_length': cot_gen_len,
                    'cot_output': cot_text
                }
                
                # Add question and choices based on dataset format
                if self.dataset_name == "mmmlu":
                    cot_log_entry.update({
                        'question': example.get('Question', ''),
                        'A': example.get('A', ''),
                        'B': example.get('B', ''),
                        'C': example.get('C', ''),
                        'D': example.get('D', '')
                    })
                else:  # mmlu-redux
                    choices = example.get('choices', [])
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': choices[0] if len(choices) > 0 else '',
                        'B': choices[1] if len(choices) > 1 else '',
                        'C': choices[2] if len(choices) > 2 else '',
                        'D': choices[3] if len(choices) > 3 else ''
                    })
                
                cot_logs.append(cot_log_entry)
                total_count += 1
                
            except Exception as e:
                print(f"Error processing question {i} in subject {subject}: {e}")
                skip_count += 1
                continue
        
        if total_count > 0:
            acc = np.mean(cors)
            print(f"{subject} accuracy: {acc*100:.2f}% (evaluated on {total_count} samples, skipped {skip_count})")
        else:
            acc = 0
            print(f"{subject} skipped all samples ({skip_count} skipped)")
        
        return np.array(cors) if cors else None, acc, np.array(all_probs) if all_probs else None, length_stats, cot_logs
    
    def evaluate_on_gpu(self, rank: int, gpu_id: int, subjects: List[str], return_dict):
        """
        Evaluate on a single GPU.
        
        Args:
            rank: Process rank
            gpu_id: GPU ID
            subjects: List of subjects to evaluate
            return_dict: Shared dictionary for results
        """
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        # Load model
        if "Rosetta" in self.model_config["model_name"]:
            model, tokenizer = load_oracle_rosetta_model(self.model_config, self.eval_config, device=device)
            # Load LLM tokenizer only if alignment is enabled via eval or model config
            rosetta_cfg = self.model_config.get("rosetta_config", {})
            is_do_alignment = self.model_config.get("is_do_alignment", rosetta_cfg.get("is_do_alignment", False))
            llm_model_path = rosetta_cfg.get("teacher_model")
            llm_tokenizer = None
            if is_do_alignment and llm_model_path:
                try:
                    llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
                    if llm_tokenizer.pad_token is None:
                        llm_tokenizer.pad_token = llm_tokenizer.eos_token
                    set_default_chat_template(llm_tokenizer, llm_model_path)
                except Exception as e:
                    print(f"Failed to load LLM tokenizer '{llm_model_path}': {e}")
                    llm_tokenizer = None
            model_type = "rosetta"
        else:
            model, tokenizer = load_hf_model(self.model_config["model_name"], device=device)
            if "Qwen" in self.model_config["model_name"]:
                model_type = "qwen"
            else:
                model_type = "hf"
            llm_tokenizer = None
        
        all_cors = []
        subject_cors = {}
        subcat_cors = defaultdict(list)
        cat_cors = defaultdict(list)
        all_length_stats = []
        cot_logs_all = []
        
        for subject in subjects:
            cors, acc, _, length_stats, cot_logs = self.evaluate_subject(
                subject, model, tokenizer, device, model_type, llm_tokenizer
            )
            if cors is None:
                continue
            
            all_cors.append(cors)
            subject_cors[subject] = acc
            all_length_stats.extend(length_stats)
            cot_logs_all.extend(cot_logs)
            
            # Organize by subcategories and categories (if applicable)
            if self.dataset_name == "mmlu-redux":
                for subcat in self.dataset_config["subcategories"].get(subject, []):
                    subcat_cors[subcat].append(cors)
                    for cat, subcat_list in self.dataset_config["categories"].items():
                        if subcat in subcat_list:
                            cat_cors[cat].append(cors)
        
        return_dict[rank] = {
            "all_cors": all_cors,
            "subject_cors": subject_cors,
            "subcat_cors": dict(subcat_cors),
            "cat_cors": dict(cat_cors),
            "length_stats": all_length_stats,
            "cot_logs": cot_logs_all
        }
    
    def merge_results(self, results_by_rank: Dict) -> Tuple:
        """
        Merge results from multiple GPUs.
        
        Args:
            results_by_rank: Dictionary of results by rank
            
        Returns:
            Merged results tuple
        """
        all_cors = []
        subject_cors = {}
        subcat_cors = defaultdict(list)
        cat_cors = defaultdict(list)
        all_length_stats = []
        all_cot_logs = []
        
        for result in results_by_rank.values():
            all_cors.extend(result["all_cors"])
            subject_cors.update(result.get("subject_cors", {}))
            all_length_stats.extend(result.get("length_stats", []))
            all_cot_logs.extend(result.get("cot_logs", []))
            
            for k, v in result.get("subcat_cors", {}).items():
                subcat_cors[k].extend(v)
            for k, v in result.get("cat_cors", {}).items():
                cat_cors[k].extend(v)
        
        return all_cors, subject_cors, subcat_cors, cat_cors, all_length_stats, all_cot_logs
    
    def save_results(self, all_cors, subject_cors, subcat_cors, cat_cors, 
                    all_length_stats, all_cot_logs):
        """
        Save evaluation results.
        
        Args:
            Various result arrays and dictionaries
        """
        # Calculate overall accuracy
        overall_accuracy = np.mean(np.concatenate(all_cors)) if all_cors else 0
        
        # Prepare summary
        summary = {
            "model": self.model_config["model_name"],
            "dataset": self.dataset_name,
            "answer_method": self.eval_config["answer_method"],
            "overall_accuracy": overall_accuracy,
            "subjects": subject_cors
        }
        
        # Add categories and subcategories for MMLU-Redux
        if self.dataset_name == "mmlu-redux":
            summary["categories"] = {
                cat: np.mean(np.concatenate(cors)) if cors else 0
                for cat, cors in cat_cors.items()
            }
            summary["subcategories"] = {
                subcat: np.mean(np.concatenate(cors)) if cors else 0
                for subcat, cors in subcat_cors.items()
            }
        
        # Add length statistics
        if all_length_stats:
            length_summary = self._compute_length_statistics(all_length_stats)
            summary["length_statistics"] = length_summary
        
        # Generate filename
        model_name_for_file = self.model_config["model_name"].split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary JSON
        summary_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")
        
        # Save detailed length statistics
        if all_length_stats:
            detailed_length_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_length.json"
            with open(detailed_length_file, "w") as f:
                json.dump(all_length_stats, f, indent=2)
            print(f"Detailed length statistics saved to {detailed_length_file}")
        
        # Save CoT logs as CSV
        if all_cot_logs:
            cot_csv_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_cot.csv"
            fieldnames = [
                'subject', 'question_id', 'question', 'A', 'B', 'C', 'D',
                'true_answer', 'pred', 'is_correct', 'answer_method',
                'cot_pred', 'cot_input_length', 'cot_gen_length', 'cot_output'
            ]
            with open(cot_csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_cot_logs:
                    writer.writerow(row)
            print(f"CoT outputs saved to {cot_csv_file}")
        
        print(f"\nEvaluation complete!")
        print(f"Overall accuracy: {overall_accuracy*100:.2f}%")
    
    def _compute_length_statistics(self, length_stats: List[Dict]) -> Dict:
        """
        Compute length statistics summary.
        
        Args:
            length_stats: List of length statistics
            
        Returns:
            Summary dictionary
        """
        if self.dataset_name == "mmlu-redux":
            # Group by subcategory
            subcat_stats = defaultdict(list)
            for stat in length_stats:
                subject = stat['subject']
                for subcat in self.dataset_config["subcategories"].get(subject, []):
                    subcat_stats[subcat].append(stat)
            
            summary = {"subcategories": {}}
            for subcat, stats in subcat_stats.items():
                if stats:
                    summary["subcategories"][subcat] = {
                        "avg_input_length": np.mean([s['input_length'] for s in stats]),
                        "avg_gen_length": np.mean([s['gen_length'] for s in stats]),
                        "avg_length_ratio": np.mean([s['length_ratio'] for s in stats]),
                        "accuracy": np.mean([s['is_correct'] for s in stats]),
                        "total_samples": len(stats)
                    }
        else:
            # Group by subject for MMMLU
            subject_stats = defaultdict(list)
            for stat in length_stats:
                subject_stats[stat['subject']].append(stat)
            
            summary = {"subjects": {}}
            for subject, stats in subject_stats.items():
                if stats:
                    summary["subjects"][subject] = {
                        "avg_input_length": np.mean([s['input_length'] for s in stats]),
                        "avg_gen_length": np.mean([s['gen_length'] for s in stats]),
                        "avg_length_ratio": np.mean([s['length_ratio'] for s in stats]),
                        "accuracy": np.mean([s['is_correct'] for s in stats]),
                        "total_samples": len(stats)
                    }
        
        return summary
    
    def run(self):
        """Run the evaluation."""
        gpu_ids = self.eval_config["gpu_ids"]
        num_gpus = len(gpu_ids)
        print(f"Using {num_gpus} GPUs: {gpu_ids}")
        
        # Get subjects for this dataset
        subjects = self.dataset_config["subjects"]
        
        # Filter subjects if specified in config
        if "subjects" in self.eval_config:
            subjects = [s for s in subjects if s in self.eval_config["subjects"]]
        
        # Distribute subjects across GPUs
        subject_chunks = [subjects[i::num_gpus] for i in range(num_gpus)]
        
        # Launch multi-process evaluation
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        for rank, gpu_id in enumerate(gpu_ids):
            p = mp.Process(
                target=self.evaluate_on_gpu,
                args=(rank, gpu_id, subject_chunks[rank], return_dict)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        # Merge and save results
        results = self.merge_results(return_dict)
        self.save_results(*results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Unified Evaluation Script')
    parser.add_argument(
        "--config",
        type=str,
        default="eval_recipe/unified_eval.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Remove CUDA_VISIBLE_DEVICES to use all GPUs
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    # Create and run evaluator
    evaluator = UnifiedEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
