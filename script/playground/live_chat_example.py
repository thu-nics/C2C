"""
Live Chat Example with Trained Models

This script provides an interactive chat interface for trained models.
It supports both Rosetta models (with projectors) and regular HuggingFace models.

Usage:
    python live_chat_example.py --checkpoint_dir local/checkpoints/20250904_210200
    python live_chat_example.py --checkpoint_dir local/checkpoints/20250904_210200 --subfolder checkpoint-1000
    python live_chat_example.py --model_name Qwen/Qwen3-0.6B  # For HF models
"""

import argparse
import os
import sys
import torch
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM
from rosetta.utils.evaluate import load_rosetta_model, load_hf_model, set_default_chat_template
from rosetta.model.wrapper import RosettaModel


class LiveChatBot:
    """Interactive chat bot for trained models."""
    
    def __init__(self, model, tokenizer, model_name: str, model_type: str = "hf"):
        """
        Initialize the chat bot.
        
        Args:
            model: Loaded model (RosettaModel or HF model)
            tokenizer: Tokenizer
            model_type: Type of model ("rosetta" or "hf")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda')
        set_default_chat_template(self.tokenizer, model_name)

        print(f"Chat bot initialized with {model_type} model on {self.device}")
        print("Type 'q' or 'quit' to exit the chat.\n")
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's input text
            
        Returns:
            Generated response text
        """
        # Prepare messages
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": user_input}]
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generation parameters
        if self.model_type == "rosetta":
            sampling_params = {
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 20,
                # 'min_p': 0.0,
                # 'presence_penalty': 1.5,
                'max_new_tokens': 2048
            }
        else:
            sampling_params = {
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'min_p': 0.0,
                'max_new_tokens': 2048
            }

        # Generate response
        with torch.no_grad():
            if self.model_type == "rosetta":
                # For Rosetta models, we need to handle the special input format
                # Create kv_cache_index for Rosetta model
                full_length = inputs.input_ids.shape[1]
                instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(full_length - 1, 1).unsqueeze(0).to(self.device) # shape: (seq_len-1, 2)
                label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(self.device) # shape: (1, 2)
                kv_cache_list = [instruction_index, label_index]
                
                # Add position_ids if needed
                if inputs.attention_mask is None:
                    position_ids = torch.arange(inputs.input_ids.shape[-1], dtype=torch.long).unsqueeze(0).to(self.device)
                else:
                    position_ids = inputs.attention_mask.long().cumsum(-1) - 1
                
                outputs = self.model.generate(
                    kv_cache_index=kv_cache_list,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    position_ids=position_ids,
                    **sampling_params
                )
                generated_ids = outputs[0]
            else:
                # For regular HF models
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **sampling_params
                )
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        
        # Decode response
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response
    
    def chat_loop(self):
        """Main chat loop."""
        print("=" * 50)
        print("🤖 Live Chat Bot Started")
        print("=" * 50)
        
        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for quit commands
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate and display response
            print("🤖 Bot: ", end="", flush=True)
            response = self.generate_response(user_input)
            print(response)


def load_model_from_checkpoint(checkpoint_dir: str, subfolder: str = "final") -> tuple:
    """
    Load a trained model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        subfolder: Subfolder name in checkpoint directory (e.g., 'final', 'checkpoint-1000')
        
    Returns:
        Tuple of (model, tokenizer, model_type)
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Load config
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check if this is a Rosetta model (has rosetta_config or projectors)
    subfolder_dir = checkpoint_path / subfolder
    has_projectors = subfolder_dir.exists() and any(
        f.name.startswith("projector_") and f.name.endswith(".pt") 
        for f in subfolder_dir.iterdir()
    )
    
    if has_projectors:
        # Load Rosetta model
        print(f"Loading Rosetta model from {checkpoint_dir}")
        
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
        
        eval_config = {
            "checkpoints_dir": str(subfolder_dir)
        }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_rosetta_model(model_config, eval_config, device)
        
        return model, tokenizer, "rosetta"
    
    else:
        # Load regular HF model
        model_name = config["model"].get("base_model", config["model"].get("model_name"))
        if not model_name:
            raise ValueError("No model name found in config")
        
        print(f"Loading HuggingFace model: {model_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_hf_model(model_name, device)
        
        return model, tokenizer, "hf"


def load_model_from_name(model_name: str) -> tuple:
    """
    Load a model directly by name.
    
    Args:
        model_name: HuggingFace model name or path
        
    Returns:
        Tuple of (model, tokenizer, model_type)
    """
    print(f"Loading HuggingFace model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_hf_model(model_name, device)
    
    return model, tokenizer, "hf"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Live Chat with Trained Models')
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        # default="local/checkpoints/example",
        default=None,
        help="Path to checkpoint directory (for trained models). Expecting config.json and final/ in the directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model name (for pre-trained models)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda, cpu, auto)"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="final",
        help="Subfolder name in checkpoint directory (e.g., 'final', 'checkpoint-1000', etc.)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.checkpoint_dir and not args.model_name:
        print("Error: Must provide either --checkpoint_dir or --model_name")
        return
    
    if args.checkpoint_dir and args.model_name:
        print("Error: Cannot provide both --checkpoint_dir and --model_name")
        return
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    if args.checkpoint_dir:
        model, tokenizer, model_type = load_model_from_checkpoint(args.checkpoint_dir, args.subfolder)
    else:
        model, tokenizer, model_type = load_model_from_name(args.model_name)
    
    # Create and start chat bot
    model_name = args.model_name if args.model_name else "Unknown"
    bot = LiveChatBot(model, tokenizer, model_name, model_type)
    bot.chat_loop()


if __name__ == "__main__":
    main()
