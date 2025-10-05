"""
Multi-stage evaluation utilities for VLM+LLM and LLM+LLM pipelines.

This module provides utilities for multi-stage evaluation where:
1. VLM describes/analyzes images + LLM performs reasoning
2. LLM provides background context + LLM performs reasoning
"""

from typing import Dict, Optional, Any
import torch
from transformers import (
    # Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from rosetta.utils.evaluate import set_default_chat_template, apply_generation_config

try:
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    print("Please install qwen-vl-utils to use VLM models")

class TwoStageInference:
    """Two-stage LLM+LLM inference pipeline for question answering."""
    
    def __init__(
        self,
        context_model_path: str,
        answer_model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        background_prompt: str = "Briefly describe the most useful background to solve the problem:\n\n{question}",
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize two-stage LLM pipeline.
        
        Args:
            context_model_path: Path to context-providing LLM
            answer_model_path: Path to answer-generating LLM
            device: Device to use
            max_new_tokens: Maximum number of new tokens to generate
            background_prompt: Prompt template for background generation
            generation_config: Optional generation configuration to apply to models
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.background_prompt = background_prompt
        self.generation_config = generation_config or {}
        self._load_models(context_model_path, answer_model_path)
    
    def _load_models(self, context_path: str, answer_path: str):
        """Load both LLM models."""
        # Load context LLM
        self.context_tokenizer = AutoTokenizer.from_pretrained(context_path)
        # for gemma, set sliding_window=4096
        if context_path == "google/gemma-3-1b-it":
            torch._dynamo.config.cache_size_limit = 64
            self.context_model = AutoModelForCausalLM.from_pretrained(
                context_path, torch_dtype=torch.bfloat16, device_map={"": self.device}, sliding_window=4096
            )
        else:
            self.context_model = AutoModelForCausalLM.from_pretrained(
                context_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
            )
        # Apply generation config to context model
        apply_generation_config(self.context_model, self.generation_config)
        
        # Load answer LLM
        self.answer_tokenizer = AutoTokenizer.from_pretrained(answer_path)
        self.answer_model = AutoModelForCausalLM.from_pretrained(
            answer_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
        )
        # Apply generation config to answer model
        apply_generation_config(self.answer_model, self.generation_config)
    
    def get_background_context(
        self,
        question: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Get background context from the first LLM.
        
        Args:
            question: Question text (without options)
            max_new_tokens: Max tokens to generate (uses instance default if None)
            
        Returns:
            Background context
        """
        prompt = self.background_prompt.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        template_kwargs = {'enable_thinking': False}
        
        inputs = self.context_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs
        )
        inputs = inputs.to(self.device)
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        with torch.inference_mode():
            outputs = self.context_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        generated_ids = outputs[:, inputs.shape[-1]:]
        context = self.context_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return context
    
    def answer_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: Optional[int] = None,
        original_question: Optional[str] = None
    ) -> str:
        """
        Answer question using the second LLM with context.
        
        Args:
            question: Full question with options and proper template
            context: Background context from first LLM
            max_new_tokens: Max tokens to generate (uses instance default if None)
            original_question: Original question asked to first LLM (for conversation format)
            
        Returns:
            Generated answer
        """
        # Use conversation format: user asks for background, assistant provides it, user asks main question
        if original_question:
            messages = [
                {"role": "user", "content": self.background_prompt.format(question=original_question)},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # Fallback to simple format
            messages = [{"role": "user", "content": f"Background context: {context}\n\n{question}"}]

        template_kwargs = {'enable_thinking': False}
        
        inputs = self.answer_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs
        )
        inputs = inputs.to(self.device)
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        with torch.inference_mode():
            outputs = self.answer_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        generated_ids = outputs[:, inputs.shape[-1]:]
        answer = self.answer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return answer
    
    def forward_with_context(
        self,
        question: str,
        context: str,
        original_question: Optional[str] = None,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Run a forward pass on the answer model using provided context (logits mode).

        Args:
            question: Full question with options and proper template
            context: Background context from first LLM
            original_question: Original question asked to first LLM (for conversation format)
            response_text: Optional text to append after the chat template to steer next-token logits
            **forward_kwargs: Extra kwargs forwarded to the model's forward

        Returns:
            Model outputs from the forward pass (e.g., logits)
        """
        # Use conversation format: user asks for background, assistant provides it, user asks main question
        if original_question:
            messages = [
                {"role": "user", "content": self.background_prompt.format(question=original_question)},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # Fallback to simple format
            messages = [{"role": "user", "content": f"Background context: {context}\n\n{question}"}]

        template_kwargs = {'enable_thinking': False}

        # Build model inputs; if response_text is provided, append it to steer next-token prediction
        if response_text is not None:
            # Build raw text then append response_text
            text = self.answer_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs
            )
            text = text + response_text
            tokenized = self.answer_tokenizer(text, return_tensors="pt")
        else:
            # Directly build tensors with generation prompt (predict next assistant token)
            tokenized = self.answer_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs
            )

        inputs = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.inference_mode():
            outputs = self.answer_model(**inputs, **forward_kwargs)

        return outputs

    def forward(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage forward pass (logits mode):
        1) Generate background context with the context model
        2) Run a forward pass on the answer model conditioned on that context

        Args:
            question_without_options: Question text without multiple choice options
            question_with_options: Full question with options and proper template
            response_text: Optional text appended after the chat template to steer next-token logits
            **forward_kwargs: Extra kwargs forwarded to the model's forward

        Returns:
            Model outputs from the forward pass (e.g., logits)
        """
        context = self.get_background_context(question_without_options)
        return self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )

    def logits_with_context(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage logits helper that also returns the generated background context
        for logging as CoT.

        Returns:
            (outputs, context)
        """
        context = self.get_background_context(question_without_options)
        outputs = self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )
        return outputs, context

    def generate(
        self,
        question_without_options: str,
        question_with_options: str,
        communication_max_new_tokens: Optional[int] = None,
        response_max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate answer using two-stage processing.
        
        Args:
            question_without_options: Question text without multiple choice options
            question_with_options: Full question with options and proper template
            communication_max_new_tokens: Maximum tokens to generate for the background context
            response_max_new_tokens: Maximum tokens to generate for the answer
            **kwargs: Additional generation parameters (ignored for compatibility)
            
        Returns:
            Generated answer string
        """
        # Stage 1: Get background context
        context = self.get_background_context(question_without_options, communication_max_new_tokens)
        
        # Stage 2: Answer question with context
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            max_new_tokens=response_max_new_tokens
        )
        
        return answer
    
    def process(
        self,
        question_without_options: str,
        question_with_options: str
    ) -> Dict[str, str]:
        """
        Full two-stage processing (legacy method for backward compatibility).
        
        Args:
            question_without_options: Question text without multiple choice options
            question_with_options: Full question with options and proper template
            
        Returns:
            Dictionary with context and answer
        """
        # Stage 1: Get background context
        context = self.get_background_context(question_without_options)
        
        # Stage 2: Answer question with context
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options
        )
        
        return {
            "context": context,
            "answer": answer
        }


class TwoStageRosetta(TwoStageInference):
    """Two-stage LLM+Rosetta inference pipeline for question answering."""
    
    def __init__(
        self,
        context_model_path: str,
        rosetta_checkpoint_dir: str,
        rosetta_subfolder: str = "final",
        device: str = "cuda",
        max_new_tokens: int = 1024,
        background_prompt: str = "Briefly describe the most useful background to solve the problem:\n\n{question}",
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize two-stage pipeline with Rosetta as second model.
        
        Args:
            context_model_path: Path to context-providing LLM
            rosetta_checkpoint_dir: Path to Rosetta checkpoint directory
            rosetta_subfolder: Subfolder name in checkpoint directory (e.g., 'final', 'checkpoint-1000')
            device: Device to use
            max_new_tokens: Maximum number of new tokens to generate
            background_prompt: Prompt template for background generation
            generation_config: Optional generation configuration to apply to models
        """
        # Initialize parent class with dummy answer model path
        # We'll override the answer model loading
        super().__init__(
            context_model_path=context_model_path,
            answer_model_path=None,  # Will be overridden
            device=device,
            max_new_tokens=max_new_tokens,
            background_prompt=background_prompt,
            generation_config=generation_config
        )
        
        self.rosetta_checkpoint_dir = rosetta_checkpoint_dir
        self.rosetta_subfolder = rosetta_subfolder
        self._load_rosetta_model()
    
    def _load_models(self, context_path: str, answer_path: str):
        """
        Override parent class _load_models to prevent loading dummy answer model.
        We only load the context model here, and the Rosetta model is loaded separately.
        """
        # Only load context LLM (answer model is replaced by Rosetta)
        self.context_tokenizer = AutoTokenizer.from_pretrained(context_path)
        self.context_model = AutoModelForCausalLM.from_pretrained(
            context_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
        )
        # Apply generation config to context model
        apply_generation_config(self.context_model, self.generation_config)
        
        # Skip loading answer model - we use Rosetta instead
        print(f"Loaded context model from {context_path}")
        print("Skipping answer model loading - using Rosetta model instead")
    
    def _load_rosetta_model(self):
        """Load Rosetta model and related components following load_model_from_checkpoint pattern."""
        import json
        from pathlib import Path
        from rosetta.utils.evaluate import load_rosetta_model
        
        checkpoint_path = Path(self.rosetta_checkpoint_dir)
        
        # Load config
        config_path = checkpoint_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if this is a Rosetta model (has projectors)
        subfolder_dir = checkpoint_path / self.rosetta_subfolder
        has_projectors = subfolder_dir.exists() and any(
            f.name.startswith("projector_") and f.name.endswith(".pt") 
            for f in subfolder_dir.iterdir()
        )
        
        if not has_projectors:
            raise ValueError(f"No projectors found in {subfolder_dir}. This doesn't appear to be a Rosetta checkpoint.")
        
        # Load Rosetta model (following load_model_from_checkpoint pattern)
        print(f"Loading Rosetta model from {self.rosetta_checkpoint_dir}")
        
        # Create model config for Rosetta loading
        model_config = {
            "model_name": "Rosetta",
            "rosetta_config": {
                "checkpoints_dir": str(subfolder_dir),
                "base_model": config["model"]["base_model"],
                "teacher_model": config["model"]["teacher_model"],
                "is_do_alignment": config["model"].get("is_do_alignment", False),
                "alignment_strategy": config["model"].get("alignment_strategy", "first")
            }
        }

        print(f"Model config: {model_config}")
        
        eval_config = {
            "checkpoints_dir": str(subfolder_dir)
        }
        
        # Load Rosetta model using the existing utility
        self.rosetta_model, self.rosetta_tokenizer = load_rosetta_model(
            model_config, 
            eval_config, 
            device=self.device
        )
        
        # Load LLM tokenizer for alignment if needed
        is_do_alignment = config["model"].get("is_do_alignment", False)
        llm_model_path = config["model"].get("teacher_model")
        self.llm_tokenizer = None
        
        if is_do_alignment and llm_model_path:
            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                set_default_chat_template(self.llm_tokenizer, llm_model_path)
            except Exception as e:
                print(f"Failed to load LLM tokenizer '{llm_model_path}': {e}")
                self.llm_tokenizer = None
        
        print(f"Initialized TwoStageRosetta with Rosetta model on {self.device}")
    
    def _prepare_rosetta_inputs(
        self,
        question: str,
        context: str,
        original_question: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        answer_method: str = "generate",
        response_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare inputs for Rosetta model using the simpler approach from live_chat_example.py.
        
        Args:
            question: Question to answer
            context: Background context from first LLM
            original_question: Original question asked to first LLM (for conversation format)
            max_new_tokens: Max tokens to generate (uses instance default if None)
            
        Returns:
            Dictionary with prepared inputs for Rosetta model
        """
        # Use conversation format: user asks for background, assistant provides it, user asks main question
        if original_question:
            messages = [
                {"role": "user", "content": self.background_prompt.format(question=original_question)},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # Fallback to simple format
            messages = [{"role": "user", "content": f"Background context: {context}\n\n{question}"}]

        # Apply chat template (following live_chat_example.py pattern)
        base_text = None
        if hasattr(self.rosetta_tokenizer, 'apply_chat_template'):
            base_text = self.rosetta_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            base_text = f"### Human: {question}\n### Assistant:"

        # Optionally append response_text for logits mode to steer next-token logits
        if answer_method == 'logits' and response_text is not None:
            text = base_text + response_text
        else:
            text = base_text

        # Tokenize input
        inputs = self.rosetta_tokenizer(text, return_tensors="pt").to(self.device)
        
        # Create kv_cache_index for Rosetta model
        full_length = inputs.input_ids.shape[1]
        if answer_method == 'logits':
            # Compute response length as the extra tokens appended by response_text
            if response_text is not None:
                base_tok = self.rosetta_tokenizer(base_text, return_tensors="pt")
                response_length = int(inputs.input_ids.shape[1] - base_tok.input_ids.shape[1])
                response_length = max(response_length, 0)
            else:
                response_length = 0
            instr_len = max(full_length - response_length, 0)
            instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(instr_len, 1).unsqueeze(0).to(self.device)
            if response_length > 0:
                response_index = torch.tensor([-1, 0], dtype=torch.long).repeat(response_length, 1).unsqueeze(0).to(self.device)
                kv_cache_list = [instruction_index, response_index]
            else:
                kv_cache_list = [instruction_index]
        else:
            # Generate: treat the last position as response (length 1)
            instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(full_length - 1, 1).unsqueeze(0).to(self.device)
            label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(self.device)
            kv_cache_list = [instruction_index, label_index]
        
        # Add position_ids if needed
        if inputs.attention_mask is None:
            position_ids = torch.arange(inputs.input_ids.shape[-1], dtype=torch.long).unsqueeze(0).to(self.device)
        else:
            position_ids = inputs.attention_mask.long().cumsum(-1) - 1
        
        outputs = {
            "inputs": {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "position_ids": position_ids,
                "kv_cache_index": kv_cache_list
            },
            "printable_text": text
        }

        return outputs
    
    def answer_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: Optional[int] = None,
        original_question: Optional[str] = None
    ) -> str:
        """
        Answer question using Rosetta model with context.
        Overrides parent class method to use Rosetta instead of regular LLM.
        
        Args:
            question: Question to answer
            context: Background context from first LLM
            max_new_tokens: Max tokens to generate (uses instance default if None)
            original_question: Original question asked to first LLM (for conversation format)
            
        Returns:
            Generated answer
        """
        # Prepare inputs for Rosetta model
        prepared = self._prepare_rosetta_inputs(
            question=question,
            context=context,
            original_question=original_question,
            max_new_tokens=max_new_tokens
        )
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        # Generation parameters (following live_chat_example.py pattern)
        sampling_params = {
            'do_sample': False,
            'max_new_tokens': max_new_tokens
        }
        
        # Generate using Rosetta model (following live_chat_example.py pattern)
        input_length = prepared['inputs']['input_ids'].shape[1]

        with torch.inference_mode():
            outputs = self.rosetta_model.generate(
                kv_cache_index=prepared['inputs']['kv_cache_index'],
                input_ids=prepared['inputs']['input_ids'],
                attention_mask=prepared['inputs']['attention_mask'],
                position_ids=prepared['inputs']['position_ids'],
                **sampling_params
            )
            generated_ids = outputs[0]
        
        # Decode response
        answer = self.rosetta_tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        
        return answer

    def forward_with_context(
        self,
        question: str,
        context: str,
        original_question: Optional[str] = None,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Run a forward pass on the Rosetta model using provided context (logits mode).

        Args:
            question: Full question with options and proper template
            context: Background context from first LLM
            original_question: Original question asked to first LLM (for conversation format)
            response_text: Optional text appended after the chat template to steer next-token logits
            **forward_kwargs: Extra kwargs forwarded to the model's forward

        Returns:
            Model outputs from the forward pass (e.g., logits)
        """
        prepared = self._prepare_rosetta_inputs(
            question=question,
            context=context,
            original_question=original_question,
            answer_method='logits',
            response_text=response_text
        )

        inputs = prepared['inputs']
        with torch.inference_mode():
            outputs = self.rosetta_model.forward(
                kv_cache_index=inputs['kv_cache_index'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                position_ids=inputs['position_ids'],
                **forward_kwargs
            )
        return outputs

    def forward(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage forward pass (logits mode) for Rosetta:
        1) Generate background context with the context model
        2) Run a forward pass on the Rosetta model conditioned on that context

        Args:
            question_without_options: Question text without multiple choice options
            question_with_options: Full question with options and proper template
            response_text: Optional text appended after the chat template to steer next-token logits
            **forward_kwargs: Extra kwargs forwarded to the model's forward

        Returns:
            Model outputs from the forward pass (e.g., logits)
        """
        # Work in progress
        raise NotImplementedError
        context = self.get_background_context(question_without_options)
        return self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )

    def logits_with_context(
        self,
        question_without_options: str,
        question_with_options: str,
        response_text: Optional[str] = None,
        **forward_kwargs
    ) -> Any:
        """
        Two-stage logits helper that also returns the generated background context
        for logging as CoT (Rosetta backend).

        Returns:
            (outputs, context)
        """
        context = self.get_background_context(question_without_options)
        outputs = self.forward_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            response_text=response_text,
            **forward_kwargs
        )
        return outputs, context
    
    def generate(
        self,
        question_without_options: str,
        question_with_options: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate answer using two-stage processing with Rosetta.
        
        Args:
            question_without_options: Question text without multiple choice options
            question_with_options: Full question with options and proper template
            max_new_tokens: Maximum tokens to generate (passed to both stages)
            **kwargs: Additional generation parameters (ignored for compatibility)
            
        Returns:
            Generated answer string
        """
        # Stage 1: Get background context (uses parent class method)
        context = self.get_background_context(question_without_options, max_new_tokens)
        
        # Stage 2: Answer question with context using Rosetta
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options,
            max_new_tokens=max_new_tokens
        )
        
        return answer
    
    def process(
        self,
        question_without_options: str,
        question_with_options: str
    ) -> Dict[str, str]:
        """
        Full two-stage processing with Rosetta (legacy method for backward compatibility).
        
        Args:
            question_without_options: Question text without multiple choice options
            question_with_options: Full question with options and proper template
            
        Returns:
            Dictionary with context and answer
        """
        # Stage 1: Get background context (uses parent class method)
        context = self.get_background_context(question_without_options)
        
        # Stage 2: Answer question with context using Rosetta
        answer = self.answer_with_context(
            question=question_with_options,
            context=context,
            original_question=question_without_options
        )
        
        return {
            "context": context,
            "answer": answer
        }



class MultiModalInference:
    """Multi-modal VLM+LLM inference pipeline."""
    
    def __init__(
        self,
        vlm_model_path: str,
        llm_model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-modal pipeline.
        
        Args:
            vlm_model_path: Path to VLM model
            llm_model_path: Path to LLM model  
            device: Device to use
            max_new_tokens: Maximum number of new tokens to generate
            generation_config: Optional generation configuration to apply to models
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config or {}
        self._load_models(vlm_model_path, llm_model_path)
    
    def _load_models(self, vlm_path: str, llm_path: str):
        """Load VLM and LLM models."""
        # Load VLM
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )
        # Apply generation config to VLM model
        apply_generation_config(self.vlm_model, self.generation_config)
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_path)
        
        # Load LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16, device_map={"": self.device}
        )
        # Apply generation config to LLM model
        apply_generation_config(self.llm_model, self.generation_config)
    
    def get_image_description(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Get image description from VLM.
        
        Args:
            image_path: Path to image
            prompt: Description prompt
            max_new_tokens: Max tokens to generate (uses instance default if None)
            
        Returns:
            Image description
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
            
        with torch.inference_mode():
            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
        description = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return description
    
    def answer_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: Optional[int] = None,
        original_question: Optional[str] = None
    ) -> str:
        """
        Answer question using LLM with context.
        
        Args:
            question: Question to answer
            context: Context (e.g., image description from VLM)
            max_new_tokens: Max tokens to generate (uses instance default if None)
            original_question: Original question asked to VLM (for conversation format)
            
        Returns:
            Generated answer
        """
        # Use conversation format: user asks about image, assistant describes, user asks follow-up
        if original_question:
            messages = [
                {"role": "user", "content": original_question},
                {"role": "assistant", "content": context},
                {"role": "user", "content": question}
            ]
        else:
            # Fallback to simple format
            messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]

        template_kwargs = {'enable_thinking': False}
        
        # Some tokenizers may not support enable_thinking parameter
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs
        )

        inputs = inputs.to(self.llm_model.device)
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        with torch.inference_mode():
            outputs = self.llm_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        generated_ids = outputs[:, inputs.shape[-1]:]
        answer = self.llm_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return answer
    
    def process(
        self,
        image_path: str,
        question: str,
        description_prompt: str = "Briefly describe this image."
    ) -> Dict[str, str]:
        """
        Full multi-stage processing.
        
        Args:
            image_path: Path to image
            question: Question to answer
            description_prompt: Prompt for image description
            
        Returns:
            Dictionary with description and answer
        """
        # Stage 1: Get image description
        description = self.get_image_description(image_path, description_prompt)
        
        # Stage 2: Answer question with context (pass original prompt for conversation format)
        answer = self.answer_with_context(
            question=question, 
            context=description,
            original_question=description_prompt
        )
        
        return {
            "description": description,
            "answer": answer
        }