"""
Unified Evaluation Script for Multiple Benchmarks

This script provides a unified interface for evaluating models on various benchmarks
including MMLU-Redux, MMMLU, and LongBench. It supports multi-GPU parallel evaluation and 
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
from datasets import load_dataset, load_from_disk
from datetime import datetime
import hashlib
import random
import time
import re
import sys
import re
import hashlib

from rosetta.utils.evaluate import (
    extract_answer_from_content,
    load_hf_model,
    load_rosetta_model,
    get_option_token_ids,
    build_prompt,
    apply_generation_config
)
from rosetta.utils.matheval import GSM8KEvaluator, MATH500Evaluator
from rosetta.model.wrapper import RosettaModel
from rosetta.model.aligner import TokenAligner, AlignmentStrategy
from rosetta.train.dataset_adapters import generate_kv_cache_index
from transformers import AutoTokenizer
from rosetta.utils.evaluate import set_default_chat_template
from rosetta.baseline.multi_stage import TwoStageInference, TwoStageRosetta

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
    },

    "gpqa": {
        "dataset_name": "Idavidrein/gpqa",
        "test_split": "train",
        "subjects": [
            "gpqa_diamond",
        ],
        "subcategories": {},
        "categories": {}
    },
    "math-500": {
        "dataset_name": "HuggingFaceH4/MATH-500",
        "test_split": "test",
        "subjects": ["all"]
    },
    "longbench": {
        "dataset_name": "THUDM/LongBench",
        "test_split": "test",
        "subjects": [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ],
        "subjects_e": ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
        "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"],
        "subcategories": {},  # LongBench doesn't have subcategories
        "categories": {}  # LongBench doesn't have categories
    },
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "test_split": "test",
        "subjects": ["main"],
        "subcategories": {},
        "categories": {}
    },
    "openbookqa": {
        "dataset_name": "openbookqa",
        "test_split": "test",
        "subjects": ["main"],
        "subcategories": {},
        "categories": {}
    },
    "ai2-arc": {
        "dataset_name": "allenai/ai2_arc",
        "test_split": "test",
        "subjects": ["ARC-Challenge"],
        "subcategories": {},
        "categories": {}
    },
    "mmlu-pro": {
        "dataset_name": "TIGER-Lab/MMLU-Pro",
        "test_split": "test",
        "subjects": ["main"],
        "subcategories": {},
        "categories": {}
    },
    "ceval": {
        "dataset_name": "ceval/ceval-exam",
        "test_split": "test",
        "subjects": [
            "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
            "business_administration", "chinese_language_and_literature", "civil_servant",
            "clinical_medicine", "college_chemistry", "college_economics", "college_physics",
            "college_programming", "computer_architecture", "computer_network",
            "discrete_mathematics", "education_science", "electrical_engineer",
            "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology",
            "high_school_chemistry", "high_school_chinese", "high_school_geography",
            "high_school_history", "high_school_mathematics", "high_school_physics",
            "high_school_politics", "ideological_and_moral_cultivation", "law",
            "legal_professional", "logic", "mao_zedong_thought", "marxism",
            "metrology_engineer", "middle_school_biology", "middle_school_chemistry",
            "middle_school_geography", "middle_school_history", "middle_school_mathematics",
            "middle_school_physics", "middle_school_politics", "modern_chinese_history",
            "operating_system", "physician", "plant_protection", "probability_and_statistics",
            "professional_tour_guide", "sports_science", "tax_accountant",
            "teacher_qualification", "urban_and_rural_planner", "veterinary_medicine"
        ],
        "subcategories": {},
        "categories": {}
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
        
        # Extract generation config if provided
        self.generation_config = self.model_config.get("generation_config", {})
        
        # Get dataset-specific configuration
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        self.dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        # Load LongBench prompt formats if needed
        if self.dataset_name == "longbench":
            prompt_format_path = self.eval_config.get("longbench_prompt_format_path", 
                                                    "longbench/config/dataset2prompt.json")
            
            self.dataset_prompt_formats = json.load(open(prompt_format_path, "r"))
            maxlen_format_path = self.eval_config.get("longbench_maxlen_format_path", 
                                                "longbench/config/dataset2maxlen.json")  # 需与第二段代码的 config 路径一致
            self.dataset_maxlen = json.load(open(maxlen_format_path, "r"))
            
            # 3. 新增：记录当前是否为 LongBench-E 模式（用于任务类型后缀）
            self.is_longbench_e = self.eval_config.get("longbench_e", False)
            
            # 4. 新增：加载 tokenizer（用于后续截断，需提前初始化）
            # 注：此处复用 evaluator 后续加载的 tokenizer，需调整初始化顺序，或在 format 时传入 tokenizer
            self.tokenizer = None  # 后续在 evaluate_subject 中赋值        
            
        # Setup output directory
        self.output_dir = Path(self.output_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Debug options
        self.debug_dump_bad_samples = bool(self.eval_config.get("debug_dump_bad_samples", True))
        self.cuda_launch_blocking = bool(self.eval_config.get("cuda_launch_blocking", False))
        
        # Check if using two-stage based on model_name
        self.use_two_stage = self.model_config["model_name"].lower() in ["two_stage", "two_stage_rosetta"]
        if self.use_two_stage:
            self.context_model_path = self.model_config.get("context_model_path")
            self.background_prompt = self.model_config.get(
                "background_prompt", 
                "Briefly describe the most useful background to solve the problem:\n\n{question}"
            )
            
            if self.model_config["model_name"].lower() == "two_stage":
                self.answer_model_path = self.model_config.get("answer_model_path")
                print(f"Two-stage mode enabled:")
                print(f"  Context model: {self.context_model_path}")
                print(f"  Answer model: {self.answer_model_path}")
            elif self.model_config["model_name"].lower() == "two_stage_rosetta":
                self.rosetta_checkpoint_dir = self.model_config.get("rosetta_checkpoint_dir")
                self.rosetta_subfolder = self.model_config.get("rosetta_subfolder", "final")
                print(f"Two-stage Rosetta mode enabled:")
                print(f"  Context model: {self.context_model_path}")
                print(f"  Rosetta checkpoint: {self.rosetta_checkpoint_dir}")
                print(f"  Rosetta subfolder: {self.rosetta_subfolder}")
        
        print(f"Evaluating on dataset: {self.dataset_name}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Requested GPU IDs: {self.eval_config['gpu_ids']}")
        print(f"Answer method: {self.eval_config['answer_method']}")

    def _make_subject_splits(self, num_gpus: int) -> List[str]:
        """Create virtual subject splits for datasets without native subjects.

        For datasets like math-500 and gsm8k, return SPLIT_i_OF_N identifiers
        so we can distribute the workload across GPUs evenly.
        """
        return [f"SPLIT_{i}_OF_{num_gpus}" for i in range(num_gpus)]

    def _dump_bad_sample(self, subject: str, question_id: int, example: Dict[str, Any], error: Exception, prompt: Optional[str] = None):
        """Dump problematic sample for post-mortem analysis."""
        try:
            dump_dir = self.output_dir / "bad_samples"
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f"bad_{self.dataset_name}_{subject}_{question_id}.json"
            record = {
                "dataset": self.dataset_name,
                "subject": subject,
                "question_id": question_id,
                "error": str(error),
                "example": example,
            }
            if prompt is not None:
                record["prompt"] = prompt
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            print(f"Saved bad sample to {dump_path}")
        except Exception as e:
            print(f"Failed to dump bad sample for {subject} #{question_id}: {e}")
    
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
        elif self.dataset_name == "gpqa":
            return self._format_gpqa_example(example, use_cot)
        elif self.dataset_name in ["math-500", "gsm8k"]:
            return self._format_math_problem_example(example, use_cot)
        elif self.dataset_name == "openbookqa":
            return self._format_openbookqa_example(example, use_cot)
        elif self.dataset_name == "ai2-arc":
            return self._format_ai2_arc_example(example, use_cot)
        elif self.dataset_name == "mmlu-pro":
            return self._format_mmlu_pro_example(example, use_cot)
        elif self.dataset_name == "ceval":
            return self._format_ceval_example(example, use_cot)
        elif self.dataset_name == "longbench":
            return self._format_longbench_example(example)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _format_mmmlu_example(self, example: Dict[str, Any], use_cot: bool, subject: Optional[str] = None, use_template: bool = True) -> str:
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
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_mmlu_redux_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
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
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    


    def _format_longbench_example(self, example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:


        current_subject = self.current_evaluating_subject  

        subject = re.sub(r"_e$", "", current_subject) if self.is_longbench_e else current_subject
        prompt_format = self.dataset_prompt_formats[subject]
        
        raw_prompt = prompt_format.format(**example)
        
        max_length = self.model_config.get("max_length", 32768)
        tokenized_raw = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_raw) > max_length:
            half_len = int(max_length / 2)
            raw_prompt = tokenizer.decode(tokenized_raw[:half_len], skip_special_tokens=True) + \
                        tokenizer.decode(tokenized_raw[-half_len:], skip_special_tokens=True)
        
        no_chat_template_tasks = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
        if subject not in no_chat_template_tasks:
            messages = [{"role": "user", "content": raw_prompt}]
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            final_prompt = raw_prompt
        
        return final_prompt

    def _prepare_gpqa_item(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare GPQA example into unified fields with deterministic shuffling.

        GPQA columns:
          - Question
          - Correct Answer
          - Incorrect Answer 1/2/3
        Optional revised columns (use if present and not None/empty):
          - extra_revised_question
          - extra_revised_correct_answer
          - extra_revised_incorrect_answer_1/2/3
        """
        # Prefer revised fields if available and non-empty
        def pick(primary_key: str, revised_key: str) -> str:
            revised_val = example.get(revised_key)
            if revised_val is not None and str(revised_val).strip() != "":
                return str(revised_val)
            return str(example.get(primary_key, ""))

        question_text = pick("Question", "Extra Revised Question")
        correct = pick("Correct Answer", "Extra Revised Correct Answer")
        inc1 = pick("Incorrect Answer 1", "Extra Revised Incorrect Answer 1")
        inc2 = pick("Incorrect Answer 2", "Extra Revised Incorrect Answer 2")
        inc3 = pick("Incorrect Answer 3", "Extra Revised Incorrect Answer 3")

        all_choices = [correct, inc1, inc2, inc3]
        # Deterministic shuffle based on content to keep mapping stable across calls
        seed_source = "||".join([question_text] + all_choices)
        seed_int = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed_int)
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [all_choices[idx] for idx in indices]
        correct_new_index = shuffled.index(correct)

        return {
            "question": question_text,
            "choices": shuffled,
            "answer": correct_new_index,
        }

    def _format_gpqa_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format GPQA example using the same prompt template as MMLU-Redux."""
        prepared = self._prepare_gpqa_item(example)
        choices = ""
        for i, choice in enumerate(prepared['choices']):
            choices += f"{chr(65+i)}. {choice}\n"

        prompt = build_prompt(
            dataset="gpqa",  # reuse same template
            locale="",
            question=prepared['question'],
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt

    def _format_math_problem_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format math problem examples (MATH-500, GSM8K) with a shared prompt template."""
        if self.dataset_name == "math-500":
            question_text = example.get('problem', '')
        elif self.dataset_name == "gsm8k":
            question_text = example.get('question', '')
        else:
            question_text = ""
        
        template = (
                "Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n"
                "{question}\n\n"
                "Please think step by step and explain your reasoning. Remember to put your answer on its own line after \"Answer:\", and you do not need to use a \\boxed command."
            )
        return template.replace("{question}", question_text)
    
    def _format_openbookqa_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format OpenBookQA example using the same prompt template as MMLU-Redux."""
        question_text = example.get('question_stem', '')
        # OpenBookQA 'choices' can be either
        # 1) a dict: {'text': [...], 'label': [...]} (HF common form), or
        # 2) a list of dicts: [{'text': str, 'label': 'A'|'B'|...}, ...]
        choices_texts: List[str] = []
        raw_choices = example.get('choices')
        if isinstance(raw_choices, dict):
            choices_texts = list(raw_choices.get('text', []))
        elif isinstance(raw_choices, list):
            for item in raw_choices:
                if isinstance(item, dict):
                    choices_texts.append(str(item.get('text', '')))
                else:
                    choices_texts.append(str(item))
        choices = ""
        for i, text in enumerate(choices_texts):
            choices += f"{chr(65+i)}. {text}\n"

        # Use shared prompt builder for consistency with MMLU
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_ai2_arc_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format AI2-ARC example using the same prompt template as MMLU-Redux."""
        question_text = example.get('question', '')
        # AI2-ARC 'choices' can be either
        # 1) a dict: {'text': [...], 'label': [...]} (HF common form), or
        # 2) a list of dicts: [{'text': str, 'label': 'A'|'B'|...}, ...]
        choices_texts: List[str] = []
        raw_choices = example.get('choices')
        if isinstance(raw_choices, dict):
            choices_texts = list(raw_choices.get('text', []))
        elif isinstance(raw_choices, list):
            for item in raw_choices:
                if isinstance(item, dict):
                    choices_texts.append(str(item.get('text', '')))
                else:
                    choices_texts.append(str(item))
        choices = ""
        for i, text in enumerate(choices_texts):
            choices += f"{chr(65+i)}. {text}\n"

        # Use shared prompt builder for consistency with MMLU
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_mmlu_pro_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format MMLU-Pro example with up to 10 options (A-J)."""
        question_text = example.get('question', '')
        options = example.get('options', [])
        
        # Build choices string (A-J for up to 10 options)
        choices = ""
        for i, option in enumerate(options):
            if i < 10:  # Support up to 10 options (A-J)
                choices += f"{chr(65+i)}. {option}\n"
        
        # Use shared prompt builder for consistency
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_ceval_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format C-EVAL example using the same prompt template as MMLU-Redux."""
        question_text = example.get('question', '')
        
        # Build choices string from A, B, C, D fields
        choices = ""
        for letter in ['A', 'B', 'C', 'D']:
            choice_text = example.get(letter, '')
            if choice_text:
                choices += f"{letter}. {choice_text}\n"
        
        # Use shared prompt builder for consistency
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
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
        elif self.dataset_name == "longbench":
            # For LongBench, we don't parse answers as we're generating text
            return None
        elif self.dataset_name == "gpqa":
            # Build deterministic shuffled mapping and return the correct letter
            prepared = self._prepare_gpqa_item(example)
            return chr(65 + int(prepared['answer']))
        elif self.dataset_name == "math-500":
            gt = example.get('answer')
            return None if gt is None else str(gt).strip()
        elif self.dataset_name == "gsm8k":
            full = example.get('answer', '')
            if not isinstance(full, str):
                full = str(full)
            if '####' in full:
                tail = full.split('####')[-1].strip()
                m = re.search(r"[-+]?\d+(?:\.\d+)?", tail)
                return m.group(0) if m else tail
            return None
        elif self.dataset_name == "openbookqa":
            answer_key = example.get('answerKey')
            if answer_key is None:
                return None
            # answerKey should be A, B, C, or D
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None
        elif self.dataset_name == "ai2-arc":
            answer_key = example.get('answerKey')
            if answer_key is None:
                return None
            # answerKey should be A, B, C, or D
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None
        elif self.dataset_name == "mmlu-pro":
            answer_key = example.get('answer')
            if answer_key is None:
                return None
            # answer should be A, B, C, D, E, F, G, H, I, or J
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                return answer_key
            else:
                return None
        elif self.dataset_name == "ceval":
            answer_key = example.get('answer')
            if answer_key is None:
                return None
            # answer should be A, B, C, or D
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def extract_predicted_answer(self, content: str) -> Optional[str]:
        """Extract model's predicted answer according to the active dataset.

        - math-500: use regex to capture the line after 'Answer:'
        - others: fallback to shared extract_answer_from_content
        """
        if self.dataset_name == "math-500":
            match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", content)
            return match.group(1).strip() if match else None
        return extract_answer_from_content(content)

    def _measure_latency_ms(self, run_fn, device: torch.device) -> Tuple[Any, float]:
        """Measure latency in milliseconds using CUDA events if available, fallback to CPU timer.

        Args:
            run_fn: Callable that performs the inference and returns outputs
            device: Torch device used for inference

        Returns:
            (result, latency_ms)
        """
        use_cuda_events = isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # Ensure previous work on this device is done
            torch.cuda.synchronize()
            start_event.record()
            result = run_fn()
            end_event.record()
            # Wait for kernels to finish and measure time
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            return result, float(elapsed_ms)
        else:
            t0 = time.perf_counter()
            result = run_fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return result, float(elapsed_ms)

    def create_segmented_kv_cache_index(self, instruction_length: int, response_length: int, 
                                       proportion: float, order_mode: str, device: torch.device) -> List[torch.Tensor]:
        """
        Create segmented KV cache index for both instruction and response parts, 
        with automatic segmentation based on index value changes.
        
        Args:
            instruction_length: Total number of instruction tokens
            response_length: Total number of response tokens  
            proportion: Proportion of instruction tokens that should use [1, 0] (rest use [-1, 0])
            order_mode: "front" ([1,0] first) or "back" ([1,0] last)
            device: Device to create tensors on
            
        Returns:
            List of tensors, each representing a segment with consistent kv_cache values
        """
        if proportion < 0.0 or proportion > 1.0:
            raise ValueError(f"proportion must be between 0.0 and 1.0, got {proportion}")
        
        if order_mode not in ["front", "back"]:
            raise ValueError(f"order_mode must be 'front' or 'back', got '{order_mode}'")
        
        # Calculate split sizes for instruction
        instruction_positive_length = int(instruction_length * proportion)
        instruction_negative_length = instruction_length - instruction_positive_length
        
        # Create the complete sequence (instruction + response)
        complete_sequence = []
        
        # Add instruction part according to order_mode
        if order_mode == "front":
            # [1, 0] tokens first, then [-1, 0] tokens in instruction
            complete_sequence.extend([[1, 0]] * instruction_positive_length)
            complete_sequence.extend([[-1, 0]] * instruction_negative_length)
        else:  # order_mode == "back"
            # [-1, 0] tokens first, then [1, 0] tokens in instruction
            complete_sequence.extend([[-1, 0]] * instruction_negative_length)
            complete_sequence.extend([[1, 0]] * instruction_positive_length)
        
        # Add response part (always [-1, 0])
        complete_sequence.extend([[-1, 0]] * response_length)
        
        # Now segment the complete sequence based on value changes
        if len(complete_sequence) == 0:
            return []
        
        segments = []
        current_segment_start = 0
        current_value = complete_sequence[0]
        
        for i in range(1, len(complete_sequence)):
            if complete_sequence[i] != current_value:
                # Found a change, create segment for previous section
                segment_length = i - current_segment_start
                segment = torch.tensor(current_value, dtype=torch.long).repeat(segment_length, 1).unsqueeze(0).to(device)
                segments.append(segment)
                
                # Update for next segment
                current_segment_start = i
                current_value = complete_sequence[i]
        
        # Handle the last segment
        segment_length = len(complete_sequence) - current_segment_start
        segment = torch.tensor(current_value, dtype=torch.long).repeat(segment_length, 1).unsqueeze(0).to(device)
        segments.append(segment)
        
        return segments

    def prepare_model_inputs(self, prompt: str, tokenizer, device: torch.device,
                              model_type: str, llm_tokenizer: Optional[Any],
                              answer_method: str, proportion: float = 1.0, 
                              order_mode: str = "front"):
        """
        Prepare model inputs (input_ids, attention_mask, position_ids, kv_cache_index) for
        both HF and Rosetta models, separated from the generation stage.

        Args:
            proportion: Float between 0.0 and 1.0 controlling the proportion of instruction 
                       tokens that should use [1, 0] vs [-1, 0] (default: 1.0, all [1, 0])
            order_mode: String specifying order of mixed instruction indices:
                       "front" - [1, 0] tokens first, then [-1, 0] tokens
                       "back" - [-1, 0] tokens first, then [1, 0] tokens

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
                # Use custom response text if provided, otherwise default
                response_text = self.eval_config.get("response_text", "The correct answer is")
                text += response_text
                response_length = tokenizer(response_text, add_special_tokens=False).input_ids.__len__()
            else: # generate
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                response_length = 1
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
                full_length = input_ids.shape[1]
                instruction_length = full_length - response_length
                
                # Create segmented KV cache index for the complete sequence
                kv_cache_list = self.create_segmented_kv_cache_index(
                    instruction_length=instruction_length,
                    response_length=response_length,
                    proportion=proportion,
                    order_mode=order_mode,
                    device=device
                )

                if attention_mask is None:
                    outputs['inputs']["position_ids"] = torch.arange(input_ids.shape[-1], dtype=torch.long).unsqueeze(0).to("cuda")
                else:
                    outputs['inputs']["position_ids"] = attention_mask.long().cumsum(-1) - 1
                outputs['inputs']['kv_cache_index'] = kv_cache_list
            
        # Rosetta logits path with alignment (dual tokenizers)
        # TODO: add rosetta proportion for aligner
        if use_aligner:
            alignment_strategy = self.model_config["rosetta_config"].get("alignment_strategy", "prefix")
            aligner = TokenAligner(
                slm_tokenizer=tokenizer,
                llm_tokenizer=llm_tokenizer,
                strategy=AlignmentStrategy(alignment_strategy)
            )

            if answer_method == 'logits':
                # Use custom response text if provided, otherwise default
                response_text = self.eval_config.get("response_text", "The correct answer is")
                messages.append({"role": "assistant", "content": response_text})
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
        is_virtual_split = False
        split_index = 0
        total_splits = 1
        if self.dataset_name in ["math-500", "gsm8k", "openbookqa", "gpqa", "ai2-arc", "mmlu-pro"]:
            # Detect virtual split subject: SPLIT_i_OF_N
            m = re.match(r"^SPLIT_(\d+)_OF_(\d+)$", str(subject))
            if m:
                is_virtual_split = True
                split_index = int(m.group(1))
                total_splits = max(1, int(m.group(2)))

        if self.dataset_name == "math-500":
            dataset = load_dataset(self.dataset_config["dataset_name"])
        elif self.dataset_name == "gsm8k":
            base_subset = "main" if is_virtual_split else subject
            dataset = load_dataset(self.dataset_config["dataset_name"], base_subset)
        elif self.dataset_name == "openbookqa":
            dataset = load_dataset(self.dataset_config["dataset_name"])
        elif self.dataset_name == "gpqa":
            base_subset = "gpqa_diamond" if is_virtual_split else subject
            dataset = load_dataset(self.dataset_config["dataset_name"], base_subset)
        elif self.dataset_name == "ai2-arc":
            base_subset = "ARC-Challenge" if is_virtual_split else subject
            dataset = load_dataset(self.dataset_config["dataset_name"], base_subset)
        elif self.dataset_name == "mmlu-pro":
            dataset = load_dataset(self.dataset_config["dataset_name"])
        elif self.dataset_name == "ceval":
            dataset = load_dataset(self.dataset_config["dataset_name"], subject)
        else:
            dataset = load_dataset(self.dataset_config["dataset_name"], subject)
        # dataset = load_from_disk("local/teacher_datasets/MMMLU")
        test_data = dataset[self.dataset_config["test_split"]]
        
        self.current_evaluating_subject = subject
        # 新增：将 tokenizer 赋值给 evaluator 实例（供截断使用）
        self.tokenizer = tokenizer
 
        # For LongBench, we don't use option token IDs
        num_options = 10 if self.dataset_name == "mmlu-pro" else 4
        if self.dataset_name != "longbench":
            option_ids = get_option_token_ids(tokenizer, num_options)
        
        # Prepare rule-based math evaluators if needed
        rule_evaluator = None
        if self.dataset_name == "gsm8k":
            rule_evaluator = GSM8KEvaluator()
        elif self.dataset_name == "math-500":
            rule_evaluator = MATH500Evaluator()

        cors = []
        all_probs = []
        length_stats = []
        cot_logs = []
        total_count = 0
        skip_count = 0
        printed_example = False

        if self.dataset_name == "longbench":
            # 检查学科名称是否以-e结尾
            is_longbench_e = subject.endswith("_e")
            
            # 根据学科类型确定输出基础目录
            if is_longbench_e:
                # 对于以-e结尾的学科，使用pred_e目录，并移除学科名中的-e后缀
                subject_name = subject[:-2]  # 移除末尾的-e
                output_base_dir = self.output_dir / "pred_e" / self.model_config["model_name"].split("/")[-1]
            else:
                # 普通学科使用pred目录
                subject_name = subject
                output_base_dir = self.output_dir / "pred" / self.model_config["model_name"].split("/")[-1]
            
            output_base_dir.mkdir(parents=True, exist_ok=True)
            # 使用处理后的学科名作为文件名（移除了-e后缀）
            output_file = output_base_dir / f"{subject_name}.jsonl"
            
            # 如果文件已存在，先删除（确保每次运行都是新的）
            if output_file.exists():
                output_file.unlink()

               
        # Sampling configuration
        sample_interval = self.eval_config.get("sample_interval", 1)
        sample_indices = list(range(0, len(test_data), sample_interval))

        # Apply virtual split window for datasets without native subjects
        if is_virtual_split and total_splits > 1:
            n = len(test_data)
            start = (split_index * n) // total_splits
            end = ((split_index + 1) * n) // total_splits
            sample_indices = [i for i in sample_indices if start <= i < end]
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
                
                if self.dataset_name != "longbench":
                    true_answer = self.parse_answer(example)
                    if true_answer is None:
                        skip_count += 1
                        continue
                else:
                    
                    id_hash=int(hashlib.sha256(str(example["_id"]).encode('utf-8')).hexdigest(), 16)
                    
                    if id_hash%4!=1:
                        skip_count += 1
                        continue

                # Format prompt (pass subject for locale-aware templates)
                if self.dataset_name == "mmmlu":
                    prompt = self._format_mmmlu_example(example, use_cot=self.eval_config["use_cot"], subject=subject, use_template=self.eval_config["use_template"])
                elif self.dataset_name == "mmlu-redux":
                    prompt = self._format_mmlu_redux_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "gpqa":
                    prompt = self._format_gpqa_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name in ["math-500", "gsm8k"]:
                    prompt = self._format_math_problem_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "openbookqa":
                    prompt = self._format_openbookqa_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "ai2-arc":
                    prompt = self._format_ai2_arc_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "mmlu-pro":
                    prompt = self._format_mmlu_pro_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "ceval":
                    prompt = self._format_ceval_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "longbench":
                    prompt = self._format_longbench_example(example, tokenizer) 
                else:
                    raise ValueError(f"Unknown dataset: {self.dataset_name}")
                
                # Generate answer
                if model_type in ["two_stage", "two_stage_rosetta"]:
                    # Two-stage inference mode (both regular and Rosetta)
                    # Extract question without options
                    if self.dataset_name == "mmmlu":
                        question_text = example.get('Question', '')
                    elif self.dataset_name == "mmlu-redux":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "gpqa":
                        question_text = self._prepare_gpqa_item(example)['question']
                    elif self.dataset_name == "gsm8k":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "math-500":
                        question_text = example.get('problem', '')
                    elif self.dataset_name == "openbookqa":
                        question_text = example.get('question_stem', '')
                    elif self.dataset_name == "ai2-arc":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "mmlu-pro":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "ceval":
                        question_text = example.get('question', '')
                    else:
                        question_text = ""

                    prompt_with_options = prompt

                    if self.eval_config["answer_method"] == 'logits':
                        # Forward logits path
                        response_text = self.eval_config.get("response_text", "The correct answer is")

                        def _two_stage_forward_call():
                            return model.logits_with_context(
                                question_without_options=question_text,
                                question_with_options=prompt_with_options,
                                response_text=response_text
                            )
                        (outputs, bg_context), latency_ms = self._measure_latency_ms(_two_stage_forward_call, device)

                        # Get option token IDs from the tokenizer used for second stage
                        num_options = 10 if self.dataset_name == "mmlu-pro" else 4
                        option_ids = get_option_token_ids(tokenizer, num_options)
                        logits = outputs.logits[0, -1]
                        option_logits = torch.tensor([
                            logits[option_ids[i]].item() for i in range(num_options)
                        ])
                        probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
                        pred = chr(65 + np.argmax(probs))

                        # Record background context as CoT in logits+two-stage
                        cot_text = f"[Background Context]:\n{bg_context}"
                        cot_pred = None
                        input_length, gen_length = None, None
                        cot_input_len, cot_gen_len = None, None
                    else:
                        # Generate using two-stage model
                        def _two_stage_call():
                            return model.generate(
                                question_without_options=question_text,
                                question_with_options=prompt_with_options,
                                communication_max_new_tokens=self.eval_config.get("communication_max_new_tokens", 1024),
                                response_max_new_tokens=self.eval_config.get("response_max_new_tokens", 1024)
                            )
                        content, latency_ms = self._measure_latency_ms(_two_stage_call, device)

                        # Extract and grade answer
                        pred = None
                        math_eval = None
                        if self.dataset_name in ["math-500", "gsm8k"] and rule_evaluator is not None:
                            is_corr, extracted = rule_evaluator.rule_judge(content, true_answer, finish_generation=True)
                            math_eval = {"is_correct": bool(is_corr), "extracted_answer": str(extracted)}
                            pred = str(extracted)
                        else:
                            pred = extract_answer_from_content(content)
                        probs = np.array([0.25, 0.25, 0.25, 0.25])

                        # Get context for logging using process method
                        result = model.process(question_text, prompt_with_options)
                        cot_text = f"[Background Context]:\n{result['context']}\n\n[Answer]:\n{content}"
                        cot_pred = pred
                        # Tokenize to get accurate token counts
                        input_length = len(tokenizer.encode(prompt_with_options, add_special_tokens=False))
                        gen_length = len(tokenizer.encode(content, add_special_tokens=False)) + len(tokenizer.encode(result['context'], add_special_tokens=False))
                        cot_input_len, cot_gen_len = input_length, gen_length
                        
                else:
                    # Regular single-model inference
                    # Prepare the inputs (separated from generation)
                    # Get proportion and order_mode from config, with defaults
                    proportion = self.eval_config.get("kv_cache_proportion", 1.0)
                    order_mode = self.eval_config.get("kv_cache_order_mode", "front")
                    
                    prepared = self.prepare_model_inputs(
                        prompt=prompt,
                        tokenizer=tokenizer,
                        device=device,
                        model_type=model_type,
                        llm_tokenizer=llm_tokenizer,
                        answer_method=self.eval_config["answer_method"],
                        proportion=proportion,
                        order_mode=order_mode
                    )
                    
                    if self.eval_config["answer_method"] == 'logits':
                        # Forward for logits
                        def _forward_call():
                            return model.forward(**prepared['inputs'])
                        outputs, latency_ms = self._measure_latency_ms(_forward_call, device)

                        logits = outputs.logits[0, -1]
                        option_logits = torch.tensor([
                            logits[option_ids[i]].item() for i in range(num_options)
                        ])
                        probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
                        pred = chr(65 + np.argmax(probs))

                        # No CoT generation in logits mode
                        input_length, gen_length = None, None
                        cot_pred, cot_input_len, cot_gen_len, cot_text = None, None, None, None
                    elif self.eval_config["answer_method"] == "generate":  # generate
                        # Ensure model has uniform generation config applied
                        #apply_generation_config(model, self.generation_config)

                        inputs = prepared['inputs']
                        def _generate_call():
                            return model.generate(**inputs, **self.generation_config)
                        outputs, latency_ms = self._measure_latency_ms(_generate_call, device)
                        
                        if isinstance(model, RosettaModel):
                            generated_ids = outputs[0]
                            if isinstance(prepared["inputs"]["input_ids"], list):
                                input_length = prepared["inputs"]["input_ids"][0].shape[1]
                            else:
                                input_length = prepared["inputs"]["input_ids"].shape[1]
                            generated_ids = generated_ids[input_length:]

                        else:
                            generated_ids = outputs[0][prepared['inputs']["input_ids"].shape[1]:]
                            input_length = prepared['inputs']["input_ids"].shape[1]
                        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
                        # Default values for non-MATH datasets
                        pred = None
                        math_eval = None
                        if self.dataset_name in ["math-500", "gsm8k"] and rule_evaluator is not None:
                            is_corr, extracted = rule_evaluator.rule_judge(content, true_answer, finish_generation=True)
                            math_eval = {"is_correct": bool(is_corr), "extracted_answer": str(extracted)}
                            pred = str(extracted)
                        else:
                            pred = self.extract_predicted_answer(content)
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
                        if model_type in ["two_stage", "two_stage_rosetta"]:
                            text = prompt  # Just show the formatted prompt for two-stage
                        else:
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
                if self.dataset_name != "longbench":
                    if self.dataset_name in ["math-500", "gsm8k"]:
                        # Use evaluate_answer result if available (generate path). If not, fallback to simple match
                        if 'math_eval' in locals() and math_eval is not None:
                            is_correct = bool(math_eval.get('is_correct', False))
                        else:
                            is_correct = (pred == true_answer) if pred else False
                    else:
                        is_correct = (pred == true_answer) if pred else False
                    cors.append(is_correct)
                    all_probs.append(probs)
                else:
                    is_correct = None
                    
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
                        'true_answer': true_answer if self.dataset_name != "longbench" else None
                    })
            # 对于LongBench，立即保存结果
                if self.dataset_name == "longbench":
                    output_entry = {
                        #"input":prepared.get("printable_text", ""),
                        #"context": example["context"],
                        #"question": example["input"],
                        "pred": content, 
                        "answers": example["answers"],
                        "all_classes": example["all_classes"],
                        "length": example["length"],
                        "_id": example["_id"],
                        
                    }
                
                # 追加写入文件
                    with open(output_file, "a", encoding='utf-8') as f:
                        json.dump(output_entry, f, ensure_ascii=False)
                        f.write('\n')
                        
                # Collect CoT logs
                cot_log_entry = {
                    'subject': subject,
                    'question_id': i,
                    'true_answer': true_answer if self.dataset_name != "longbench" else None,
                    'pred': pred,
                    'is_correct': is_correct,
                    'answer_method': self.eval_config.get('answer_method', ''),
                    'cot_pred': cot_pred,
                    'cot_input_length': cot_input_len,
                    'cot_gen_length': cot_gen_len,
                    'cot_output': cot_text,
                    'answer_latency_ms': float(latency_ms) if 'latency_ms' in locals() and latency_ms is not None else None
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
                elif self.dataset_name == "gpqa":
                    prepared_gpqa = self._prepare_gpqa_item(example)
                    choices = prepared_gpqa.get('choices', [])
                    cot_log_entry.update({
                        'question': prepared_gpqa.get('question', ''),
                        'A': choices[0] if len(choices) > 0 else '',
                        'B': choices[1] if len(choices) > 1 else '',
                        'C': choices[2] if len(choices) > 2 else '',
                        'D': choices[3] if len(choices) > 3 else ''
                    })
                elif self.dataset_name == "math-500":
                    cot_log_entry.update({
                        'question': example.get('problem', ''),
                        'A': '', 'B': '', 'C': '', 'D': ''
                    })
                    # Add extraction diagnostics from math evaluator if available
                    if 'math_eval' in locals() and math_eval is not None:
                        cot_log_entry.update({
                            'extraction_method_used': math_eval.get('extraction_method_used', ''),
                            'ground_truth_normalized': math_eval.get('ground_truth_normalized', ''),
                            'extracted_normalized': math_eval.get('extracted_normalized', '')
                        })
                elif self.dataset_name == "gsm8k":
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': '', 'B': '', 'C': '', 'D': ''
                    })
                    if 'math_eval' in locals() and math_eval is not None:
                        cot_log_entry.update({
                            'extraction_method_used': math_eval.get('extraction_method_used', ''),
                            'ground_truth_normalized': math_eval.get('ground_truth_normalized', ''),
                            'extracted_normalized': math_eval.get('extracted_normalized', '')
                        })
                elif self.dataset_name == "openbookqa":
                    # Normalize OpenBookQA choices to texts list
                    choices_texts: List[str] = []
                    raw_choices = example.get('choices')
                    if isinstance(raw_choices, dict):
                        choices_texts = list(raw_choices.get('text', []))
                    elif isinstance(raw_choices, list):
                        for item in raw_choices:
                            if isinstance(item, dict):
                                choices_texts.append(str(item.get('text', '')))
                            else:
                                choices_texts.append(str(item))
                    cot_log_entry.update({
                        'question': example.get('question_stem', ''),
                        'A': choices_texts[0] if len(choices_texts) > 0 else '',
                        'B': choices_texts[1] if len(choices_texts) > 1 else '',
                        'C': choices_texts[2] if len(choices_texts) > 2 else '',
                        'D': choices_texts[3] if len(choices_texts) > 3 else ''
                    })
                elif self.dataset_name == "ai2-arc":
                    # Normalize AI2-ARC choices to texts list
                    choices_texts: List[str] = []
                    raw_choices = example.get('choices')
                    if isinstance(raw_choices, dict):
                        choices_texts = list(raw_choices.get('text', []))
                    elif isinstance(raw_choices, list):
                        for item in raw_choices:
                            if isinstance(item, dict):
                                choices_texts.append(str(item.get('text', '')))
                            else:
                                choices_texts.append(str(item))
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': choices_texts[0] if len(choices_texts) > 0 else '',
                        'B': choices_texts[1] if len(choices_texts) > 1 else '',
                        'C': choices_texts[2] if len(choices_texts) > 2 else '',
                        'D': choices_texts[3] if len(choices_texts) > 3 else ''
                    })
                elif self.dataset_name == "mmlu-pro":
                    # MMLU-Pro supports up to 10 options (A-J)
                    options = example.get('options', [])
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': options[0] if len(options) > 0 else '',
                        'B': options[1] if len(options) > 1 else '',
                        'C': options[2] if len(options) > 2 else '',
                        'D': options[3] if len(options) > 3 else '',
                        'E': options[4] if len(options) > 4 else '',
                        'F': options[5] if len(options) > 5 else '',
                        'G': options[6] if len(options) > 6 else '',
                        'H': options[7] if len(options) > 7 else '',
                        'I': options[8] if len(options) > 8 else '',
                        'J': options[9] if len(options) > 9 else ''
                    })
                elif self.dataset_name == "ceval":
                    # C-EVAL uses A, B, C, D fields directly
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': example.get('A', ''),
                        'B': example.get('B', ''),
                        'C': example.get('C', ''),
                        'D': example.get('D', ''),
                    })
                elif self.dataset_name == "mmlu-redux":  # mmlu-redux
                    choices = example.get('choices', [])
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': choices[0] if len(choices) > 0 else '',
                        'B': choices[1] if len(choices) > 1 else '',
                        'C': choices[2] if len(choices) > 2 else '',
                        'D': choices[3] if len(choices) > 3 else ''
                    })
                elif self.dataset_name == "longbench":
                    cot_log_entry.update({
                        'context': example.get('context', ''),
                        'question': example.get('question', ''),
                        'input': example.get('input', ''),
                        'answers': example.get('answers', []),
                        'all_classes': example.get('all_classes', []),
                        'length': example.get('length', 0),
                        '_id': example.get('_id', f"{subject}_{i}")
                    })
                
                cot_logs.append(cot_log_entry)
                total_count += 1
                
            except Exception as e:
                print(f"Error processing question {i} in subject {subject}: {e}")
                if self.debug_dump_bad_samples:
                    try:
                        # Attempt to include the last built prompt if available
                        maybe_prompt = locals().get('prompt', None)
                        self._dump_bad_sample(subject, i, example, e, maybe_prompt)
                    except Exception as ee:
                        print(f"Failed to record bad sample for {subject} #{i}: {ee}")
                # If CUDA device-side assert, force sync to get accurate site and re-raise
                if "device-side assert" in str(e).lower() and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                skip_count += 1
                continue
        
        if total_count > 0 and self.dataset_name != "longbench":
            acc = np.mean(cors)
            print(f"{subject} accuracy: {acc*100:.2f}% (evaluated on {total_count} samples, skipped {skip_count})")
        else:
            acc = 0
            print(f"{subject} processed {total_count} samples, skipped {skip_count}")
        
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
        if "two_stage_rosetta" == self.model_config["model_name"].lower():
            model = TwoStageRosetta(
                context_model_path=self.context_model_path,
                rosetta_checkpoint_dir=self.rosetta_checkpoint_dir,
                rosetta_subfolder=self.rosetta_subfolder,
                device=device,
                max_new_tokens=self.generation_config.get("max_new_tokens", self.eval_config.get("max_new_tokens", 1024)),
                background_prompt=self.background_prompt,
                generation_config=self.generation_config
            )
            # Use the Rosetta tokenizer for consistency
            tokenizer = model.rosetta_tokenizer
            model_type = "two_stage_rosetta"
            llm_tokenizer = model.llm_tokenizer
            print(f"Initialized TwoStageRosetta pipeline on GPU {gpu_id}")
        elif "two_stage" == self.model_config["model_name"].lower():
            model = TwoStageInference(
                context_model_path=self.context_model_path,
                answer_model_path=self.answer_model_path,
                device=device,
                max_new_tokens=self.generation_config.get("max_new_tokens", self.eval_config.get("max_new_tokens", 1024)),
                background_prompt=self.background_prompt,
                generation_config=self.generation_config
            )
            # Use the answer model's tokenizer for consistency
            tokenizer = AutoTokenizer.from_pretrained(self.answer_model_path)
            model_type = "two_stage"
            llm_tokenizer = None
            print(f"Initialized two-stage pipeline on GPU {gpu_id}")
        elif "rosetta" in self.model_config["model_name"].lower():
            model, tokenizer = load_rosetta_model(self.model_config, self.eval_config, device=device, generation_config=self.generation_config)
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
            model, tokenizer = load_hf_model(self.model_config["model_name"], device=device, generation_config=self.generation_config)
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
            if cors is None and self.dataset_name != "longbench":
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
        # Calculate overall accuracy (skip for LongBench)
        if self.dataset_name != "longbench":
            overall_accuracy = np.mean(np.concatenate(all_cors)) if all_cors else 0
        else:
            overall_accuracy = 0
        
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
        
        # Save CoT logs as CSV or JSONL based on dataset
        if all_cot_logs:
            if self.dataset_name != "longbench":
                cot_csv_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_cot.csv"
                fieldnames = [
                    'subject', 'question_id', 'question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'true_answer', 'pred', 'is_correct', 'answer_method',
                    'cot_pred', 'cot_input_length', 'cot_gen_length', 'cot_output',
                    'answer_latency_ms',
                    # Extraction diagnostics (mainly for MATH-500)
                    'extraction_method_used', 'ground_truth_normalized', 'extracted_normalized'
                ]
                with open(cot_csv_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in all_cot_logs:
                        writer.writerow(row)
                print(f"CoT outputs saved to {cot_csv_file}")
        
        print(f"\nEvaluation complete!")
        if self.dataset_name != "longbench":
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
            # Group by subject for MMMLU and LongBench
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
                        "accuracy": np.mean([s['is_correct'] for s in stats]) if 'is_correct' in stats[0] else None,
                        "total_samples": len(stats)
                    }
        
        return summary
    
    def run(self):
        """Run the evaluation."""
        gpu_ids = self.eval_config["gpu_ids"]
        num_gpus = len(gpu_ids)
        print(f"Using {num_gpus} GPUs: {gpu_ids}")
        # Enable CUDA synchronous errors if requested
        if self.cuda_launch_blocking:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Get subjects for this dataset
        subjects = self.dataset_config["subjects"]
        if self.dataset_name in ["math-500", "gsm8k", "openbookqa", "gpqa", "ai2-arc", "mmlu-pro"]:
            # Create virtual subject splits to distribute across GPUs
            subjects = self._make_subject_splits(num_gpus)
        
        # Filter subjects if specified in config
        if "subjects" in self.eval_config and self.eval_config["subjects"] is not None:
            subjects = [s for s in subjects if s in self.eval_config["subjects"]]
        
        # For LongBench, check if we're evaluating on LongBench-E
            if self.dataset_name == "longbench" and self.eval_config.get("longbench_e", False):
                subjects = [f"{s}_e" for s in self.eval_config["subjects"]]
        
        if self.dataset_name == "longbench" and self.eval_config.get("longbench_e", False):
            subjects = [f"{s}_e" for s in self.dataset_config["subjects_e"]]
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
        if(self.dataset_name == "longbench"):
            print("LongBench evaluation completed. Predictions are saved in respective files.")
            return
        else:
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
    
    print("Using config: ", args.config)

    # Remove CUDA_VISIBLE_DEVICES to use all GPUs
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    # Create and run evaluator
    evaluator = UnifiedEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64 # you can expand this as needed
    mp.set_start_method("spawn", force=True)
    main()
