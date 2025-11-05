"""
Gradio Side-by-Side Model Comparison Demo

This creates a web interface to compare three inference modes simultaneously:
1. Single: Regular HuggingFace model
2. T2T: Two-stage inference (shows context + answer)
3. C2C: Rosetta model with projectors

ZeroGPU Support:
- Models are loaded to CPU at startup
- @spaces.GPU decorator moves models to GPU on-demand for each inference
- Works seamlessly on both ZeroGPU and regular GPU environments
"""

import os
import sys
import torch
import argparse
import gradio as gr
from pathlib import Path
from typing import Optional, Generator
from queue import Queue
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import spaces 
ZEROGPU_AVAILABLE = os.getenv("ZERO_GPU", "").lower() == "true" # ZeroGPU support - HuggingFace Spaces sets ZERO_GPU=true when ZeroGPU is available

from rosetta.utils.evaluate import load_rosetta_model, load_hf_model, set_default_chat_template
from rosetta.model.wrapper import RosettaModel
from rosetta.baseline.multi_stage import TwoStageInference


class ModelManager:
    """Manages loading and inference for all three model types."""
    
    def __init__(
        self,
        single_model_name: str = "Qwen/Qwen3-0.6B",
        t2t_context_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        t2t_answer_model: str = "Qwen/Qwen3-0.6B",
        c2c_checkpoint_path: str = "local/checkpoints/qwen3_0.6b+qwen2.5_0.5b_Fuser",
        device: str = "auto"
    ):
        """
        Initialize ModelManager with model configurations.
        
        Args:
            single_model_name: HuggingFace model name for single mode
            t2t_context_model: Context model for T2T mode
            t2t_answer_model: Answer model for T2T mode
            c2c_checkpoint_path: Path to C2C checkpoint directory
            device: Device to use (cuda, cpu, or auto)
        """
        # For ZeroGPU, load models to CPU and move to GPU in decorated functions
        if device == "auto":
            if ZEROGPU_AVAILABLE:
                self.device = torch.device("cpu")
                print("ZeroGPU detected: Loading models to CPU (will move to GPU on-demand)")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Model configurations
        self.single_model_name = single_model_name
        self.t2t_context_model = t2t_context_model
        self.t2t_answer_model = t2t_answer_model
        self.c2c_checkpoint_path = c2c_checkpoint_path
        
        # T2T prompt configurations
        self.t2t_background_prompt = "Briefly describe the most useful background to answer the question:\n\n{question}"
        self.t2t_answer_prompt = "Based on the background, answer the question:\n\n{question}"  # Format for second round question
        self.t2t_context_max_tokens = 512
        self.t2t_answer_max_tokens = 512
        
        # Generation configuration (shared across all models)
        # To enable sampling: set use_sampling=True and adjust temperature/top_p/top_k
        # Current mode: Greedy decoding (do_sample=False)
        self.use_sampling = False  # Set to True to enable sampling
        self.temperature = 0.7     # Used when use_sampling=True
        self.top_p = 0.8          # Used when use_sampling=True
        self.top_k = 20           # Used when use_sampling=True
        
        # Initialize models
        self.single_model = None
        self.single_tokenizer = None
        self.t2t_model = None
        self.c2c_model = None
        self.c2c_tokenizer = None
        
        # C2C model names (will be loaded from config)
        self.c2c_base_model = None
        self.c2c_teacher_model = None
        
        print("=" * 60)
        print("Initializing models... This may take a few minutes.")
        print("=" * 60)
        
        self._load_all_models()
    
    def _load_single_model(self):
        """Load single HuggingFace model."""
        print(f"\n[Single] Loading {self.single_model_name}...")
        self.single_model, self.single_tokenizer = load_hf_model(
            self.single_model_name, self.device
        )
        set_default_chat_template(self.single_tokenizer, self.single_model_name)
        print("[Single] ✓ Model loaded")
    
    def _load_t2t_model(self):
        """Load two-stage model."""
        print(f"\n[T2T] Loading two-stage model...")
        print(f"  Context: {self.t2t_context_model}")
        print(f"  Answer: {self.t2t_answer_model}")
        print(f"  Background prompt: {self.t2t_background_prompt}")
        print(f"  Answer prompt: {self.t2t_answer_prompt}")
        
        self.t2t_model = TwoStageInference(
            context_model_path=self.t2t_context_model,
            answer_model_path=self.t2t_answer_model,
            device=str(self.device),
            background_prompt=self.t2t_background_prompt
        )
        print("[T2T] ✓ Model loaded")
    
    def _load_c2c_model(self):
        """Load Rosetta (C2C) model."""
        print(f"\n[C2C] Loading from {self.c2c_checkpoint_path}...")
        
        # Auto-download if checkpoint doesn't exist
        if not Path(self.c2c_checkpoint_path).exists():
            print("[C2C] Downloading checkpoint from HuggingFace (may take a few minutes)...")
            try:
                from huggingface_hub import snapshot_download
                checkpoint_name = Path(self.c2c_checkpoint_path).name
                snapshot_download(
                    repo_id='nics-efc/C2C_Fuser',
                    allow_patterns=[f'{checkpoint_name}/*'],
                    local_dir=str(Path(self.c2c_checkpoint_path).parent)
                )
                print("[C2C] ✓ Download complete")
            except ImportError:
                raise ImportError("Install huggingface_hub: pip install huggingface_hub")
            except Exception as e:
                raise RuntimeError(f"Download failed: {e}\nManual download: https://huggingface.co/nics-efc/C2C_Fuser")
        
        # Load config
        import yaml
        config_path = Path(self.c2c_checkpoint_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Store model names from config
        self.c2c_base_model = config["model"]["base_model"]
        self.c2c_teacher_model = config["model"]["teacher_model"]
        
        # Load Rosetta model
        subfolder_dir = Path(self.c2c_checkpoint_path) / "final"
        if not subfolder_dir.exists():
            raise FileNotFoundError(f"Final checkpoint directory not found: {subfolder_dir}")
        
        model_config = {
            "model_name": "Rosetta",
            "rosetta_config": {
                "checkpoints_dir": str(subfolder_dir),
                "base_model": self.c2c_base_model,
                "teacher_model": self.c2c_teacher_model,
                "is_do_alignment": config["model"].get("is_do_alignment", False),
                "alignment_strategy": config["model"].get("alignment_strategy", "first")
            }
        }
        
        eval_config = {"checkpoints_dir": str(subfolder_dir)}
        
        self.c2c_model, self.c2c_tokenizer = load_rosetta_model(
            model_config, eval_config, self.device
        )
        print("[C2C] ✓ Model loaded")
    
    def _load_all_models(self):
        """Load all models sequentially."""
        try:
            self._load_single_model()
            self._load_t2t_model()
            self._load_c2c_model()
            print("\n" + "=" * 60)
            print("✓ All models loaded successfully!")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"\n✗ Error loading models: {e}")
            raise
    
    def _get_generation_kwargs(self, max_new_tokens: int) -> dict:
        """
        Get generation kwargs with consistent settings across all models.
        
        Args:
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Dictionary of generation parameters
        """
        kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': self.use_sampling
        }
        
        if self.use_sampling:
            kwargs.update({
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k
            })
        
        return kwargs
    
    @spaces.GPU(duration=60)
    def generate_single(self, user_input: str) -> Generator[str, None, None]:
        """Generate response from single model with streaming."""
        # Move model to GPU for ZeroGPU
        device = torch.device("cuda" if ZEROGPU_AVAILABLE else self.device)
        if ZEROGPU_AVAILABLE and self.single_model.device.type != "cuda":
            self.single_model.to(device)
        
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": user_input}]
        text = self.single_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.single_tokenizer(text, return_tensors="pt").to(device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.single_tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Generation parameters
        generation_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'streamer': streamer,
            **self._get_generation_kwargs(max_new_tokens=2048)
        }
        
        # Run generation in separate thread
        thread = Thread(target=self.single_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        generated_text = ""
        for token in streamer:
            generated_text += token
            yield generated_text
    
    @spaces.GPU(duration=90)
    def generate_t2t(self, user_input: str) -> Generator[tuple[str, str], None, None]:
        """Generate response from T2T model with streaming (returns context, answer)."""
        # Move models to GPU for ZeroGPU
        device = torch.device("cuda" if ZEROGPU_AVAILABLE else self.device)
        if ZEROGPU_AVAILABLE:
            if self.t2t_model.context_model.device.type != "cuda":
                self.t2t_model.context_model.to(device)
            if self.t2t_model.answer_model.device.type != "cuda":
                self.t2t_model.answer_model.to(device)
        
        # Stage 1: Context generation
        context_streamer = TextIteratorStreamer(
            self.t2t_model.context_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        prompt = self.t2t_background_prompt.format(question=user_input)
        inputs = self.t2t_model.context_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(device)
        
        generation_kwargs = {
            'input_ids': inputs,
            'streamer': context_streamer,
            **self._get_generation_kwargs(max_new_tokens=self.t2t_context_max_tokens)
        }
        
        # Generate context in thread
        thread = Thread(target=self.t2t_model.context_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream context tokens
        context_text = ""
        for token in context_streamer:
            context_text += token
            yield context_text, ""
        
        thread.join()
        
        # Decode full context
        with torch.inference_mode():
            outputs = self.t2t_model.context_model.generate(
                inputs, **self._get_generation_kwargs(max_new_tokens=self.t2t_context_max_tokens)
            )
        context = self.t2t_model.context_tokenizer.batch_decode(
            outputs[:, inputs.shape[-1]:], skip_special_tokens=True
        )[0]
        
        # Stage 2: Answer generation
        answer_streamer = TextIteratorStreamer(
            self.t2t_model.answer_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Format the second round question
        answer_question = self.t2t_answer_prompt.format(question=user_input)

        inputs = self.t2t_model.answer_tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": context},
                {"role": "user", "content": answer_question}
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(device)
        
        generation_kwargs = {
            'input_ids': inputs,
            'streamer': answer_streamer,
            **self._get_generation_kwargs(max_new_tokens=self.t2t_answer_max_tokens)
        }
        
        # Generate answer in thread
        thread = Thread(target=self.t2t_model.answer_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream answer tokens
        answer_text = ""
        for token in answer_streamer:
            answer_text += token
            yield context_text, answer_text
    
    @spaces.GPU(duration=60)
    def generate_c2c(self, user_input: str) -> Generator[str, None, None]:
        """Generate response from C2C model with streaming."""
        # Move model to GPU for ZeroGPU
        device = torch.device("cuda" if ZEROGPU_AVAILABLE else self.device)
        if ZEROGPU_AVAILABLE and self.c2c_model.device.type != "cuda":
            self.c2c_model.to(device)
        
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": user_input}]
        text = self.c2c_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.c2c_tokenizer(text, return_tensors="pt").to(device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.c2c_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Prepare C2C-specific inputs
        full_length = inputs.input_ids.shape[1]
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(
            full_length - 1, 1
        ).unsqueeze(0).to(device)
        label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(
            1, 1
        ).unsqueeze(0).to(device)
        position_ids = inputs.attention_mask.long().cumsum(-1) - 1 if inputs.attention_mask is not None else \
                      torch.arange(full_length, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generation parameters
        generation_kwargs = {
            'kv_cache_index': [instruction_index, label_index],
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'position_ids': position_ids,
            'streamer': streamer,
            **self._get_generation_kwargs(max_new_tokens=2048)
        }
        
        # Run generation in separate thread
        thread = Thread(target=self.c2c_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens
        generated_text = ""
        for token in streamer:
            generated_text += token
            yield generated_text


def create_demo(model_manager: ModelManager):
    """Create Gradio interface."""
    
    # Preset example questions
    EXAMPLE_QUESTIONS = {
        "example1": """Instead of asking why the act of destroying the environment might be immoral, Hill wants to ask ...

A. Why the act of destroying nature might be immoral.
B. Why people who destroy the environment might be bad people.
C. How the decision to preserve the environment benefits the environment.
D. Whether plants have interests.""",
        "example2": "Which company launched the Gemini 1.5 Pro model in early 2024?"
    }
    
    def respond(user_input: str):
        """Main response function that yields updates for all three models."""
        if not user_input.strip():
            yield "", "", "", ""
        
        # Generators for each model
        single_gen = model_manager.generate_single(user_input)
        t2t_gen = model_manager.generate_t2t(user_input)
        c2c_gen = model_manager.generate_c2c(user_input)
        
        single_done = False
        t2t_done = False
        c2c_done = False
        
        single_text = ""
        t2t_context = ""
        t2t_answer = ""
        c2c_text = ""
        
        # Stream from all three models
        while not (single_done and t2t_done and c2c_done):
            # Update single
            if not single_done:
                try:
                    single_text = next(single_gen)
                except StopIteration:
                    single_done = True
            
            # Update T2T
            if not t2t_done:
                try:
                    t2t_context, t2t_answer = next(t2t_gen)
                except StopIteration:
                    t2t_done = True
            
            # Update C2C
            if not c2c_done:
                try:
                    c2c_text = next(c2c_gen)
                except StopIteration:
                    c2c_done = True
            
            # Yield current state
            yield single_text, t2t_context, t2t_answer, c2c_text
    
    # Create Gradio interface
    with gr.Blocks(title="C2C Demo", theme=gr.themes.Base()) as demo:
        # Header with logo
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                gr.Image("https://raw.githubusercontent.com/thu-nics/C2C/main/resource/logo.png", show_label=False, show_download_button=False, container=False, height=80)
            with gr.Column(scale=5):
                gr.Markdown("# Cache-to-Cache Communication Demo")
                gr.Markdown("Compare three inference modes side-by-side: **Single** | **Text-to-Text Communication** | **Cache-to-Cache Communication**")
        
        gr.Markdown("---")
        
        # Input section
        gr.Markdown("## Question")

        # Preset question examples
        gr.Markdown("Example Questions:")
        with gr.Row():
            example1_btn = gr.Button("📝 Example 1: Philosophy", size="sm")
            example2_btn = gr.Button("📝 Example 2: Knowledge Cutoff", size="sm")


        with gr.Row():
            user_input = gr.Textbox(
                label="",
                placeholder="Type your question here...",
                lines=2,
                scale=4,
                show_label=False
            )
        
        with gr.Row():
            submit_btn = gr.Button("🚀 Submit", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear", scale=1)
        
        gr.Markdown("---")
        
        # Output section - three columns
        gr.Markdown("## Responses")
        with gr.Row():
            # Single column
            with gr.Column():
                gr.Markdown("### Single Model")
                gr.Markdown(f"*{model_manager.single_model_name}*")
                single_output = gr.Textbox(
                    label="",
                    lines=18,
                    max_lines=30,
                    interactive=False,
                    show_label=False
                )
            
            # T2T column (with two sub-boxes)
            with gr.Column():
                gr.Markdown("### Text-to-Text Communication")
                gr.Markdown(f"*{model_manager.t2t_context_model} → {model_manager.t2t_answer_model}*")
                t2t_context_output = gr.Textbox(
                    label="📝 Context",
                    lines=6,
                    max_lines=12,
                    interactive=False
                )
                t2t_answer_output = gr.Textbox(
                    label="💬 Answer",
                    lines=7,
                    max_lines=14,
                    interactive=False
                )
            
            # C2C column
            with gr.Column():
                gr.Markdown("### Cache-to-Cache Communication")
                gr.Markdown(f"*{model_manager.c2c_teacher_model} → {model_manager.c2c_base_model}*")
                c2c_output = gr.Textbox(
                    label="",
                    lines=18,
                    max_lines=30,
                    interactive=False,
                    show_label=False
                )
        
        # Event handlers
        submit_btn.click(
            fn=respond,
            inputs=[user_input],
            outputs=[single_output, t2t_context_output, t2t_answer_output, c2c_output]
        )
        
        user_input.submit(
            fn=respond,
            inputs=[user_input],
            outputs=[single_output, t2t_context_output, t2t_answer_output, c2c_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", "", ""),
            inputs=None,
            outputs=[user_input, single_output, t2t_context_output, t2t_answer_output, c2c_output]
        )
        
        # Example question handlers
        example1_btn.click(
            fn=lambda: EXAMPLE_QUESTIONS["example1"],
            inputs=None,
            outputs=[user_input]
        )
        
        example2_btn.click(
            fn=lambda: EXAMPLE_QUESTIONS["example2"],
            inputs=None,
            outputs=[user_input]
        )
    
    return demo


def main():
    """Main entry point."""
    print("=" * 60)
    print("Model Comparison Demo - Gradio Interface")
    print("=" * 60)
    
    # Initialize models
    # C2C-S: qwen3_0.6b+qwen2.5_0.5b_Fuser
    context_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    c2c_checkpoint_path = "local/checkpoints/qwen3_0.6b+qwen2.5_0.5b_Fuser"

    # C2C-L: qwen3_0.6b+qwen2.5_0.5b_Fuser_large
    # context_model_name = "Qwen/Qwen3-4B-Base"
    # c2c_checkpoint_path = "local/checkpoints/qwen3_0.6b+qwen3_4b_base_Fuser"

    answer_model_name = "Qwen/Qwen3-0.6B"
    model_manager = ModelManager(
        single_model_name=answer_model_name,
        t2t_context_model=context_model_name,
        t2t_answer_model=answer_model_name,
        c2c_checkpoint_path=c2c_checkpoint_path
    )
    
    # Create and launch demo
    demo = create_demo(model_manager)
    
    print("\n" + "=" * 60)
    print("🚀 Launching Gradio interface...")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

