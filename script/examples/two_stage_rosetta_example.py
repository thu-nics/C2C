"""
Example usage of the TwoStageRosetta pipeline for LLM+Rosetta evaluation.
"""

from rosetta.baseline.multi_stage import TwoStageRosetta


def example_standalone():
    """Example of standalone usage."""
    # Initialize the two-stage pipeline with Rosetta (following load_model_from_checkpoint pattern)
    pipeline = TwoStageRosetta(
        context_model_path="Qwen/Qwen3-4B",  # First LLM for context generation
        rosetta_checkpoint_dir="local/checkpoints/Qwen3_0.6B_4B_general_LLM_data",  # Path to checkpoint directory
        rosetta_subfolder="checkpoint-2000",  # Subfolder name (e.g., 'final', 'checkpoint-1000')
        device="cuda",
        max_new_tokens=512,
        background_prompt="In one clear sentence, describe the most essential background knowledge needed to answer the question: {question}"
    )
    
    # Example MMLU question
    question_without_options = "What is the primary function of mitochondria in cells?"
    
    question_with_options = """What is the primary function of mitochondria in cells?

A. Protein synthesis
B. Energy production through ATP synthesis
C. DNA replication
D. Waste removal

Answer: The correct answer is"""
    
    # Generate answer using the new generate method (model-like interface)
    answer = pipeline.generate(
        question_without_options=question_without_options,
        question_with_options=question_with_options,
        max_new_tokens=512
    )
    
    print("Final Answer from Two-Stage Rosetta Pipeline:")
    print(answer)
    
    # For detailed context, use process method
    result = pipeline.process(
        question_without_options=question_without_options,
        question_with_options=question_with_options
    )
    
    print("\n" + "="*50)
    print("Detailed Breakdown:")
    print("Background Context from First LLM:")
    print(result["context"])
    print("\nFinal Answer from Rosetta Model:")
    print(result["answer"])


def example_with_evaluator_integration():
    """
    Example showing how to integrate with unified_evaluator.py
    
    The integration would be similar to TwoStageInference but with Rosetta-specific config.
    """
    
    # This would be in the evaluator's evaluate_subject method
    def modified_evaluate_subject_snippet(example, use_two_stage_rosetta=True):
        """
        Pseudo-code showing integration points for TwoStageRosetta.
        """
        if use_two_stage_rosetta:
            # Extract question without options
            question_text = example.get('question', '')
            
            # Build question with options (existing code)
            choices = ""
            for i, choice in enumerate(example.get('choices', [])):
                choices += f"{chr(65+i)}. {choice}\n"
            
            question_with_options = f"""{question_text}

{choices}
Answer: The correct answer is"""
            
            # Use TwoStageRosetta (following load_model_from_checkpoint pattern)
            pipeline = TwoStageRosetta(
                context_model_path="model1_path",
                rosetta_checkpoint_dir="/path/to/rosetta/checkpoints",
                rosetta_subfolder="final"
            )
            
            # Use generate method (model-like interface)
            answer = pipeline.generate(
                question_without_options=question_text,
                question_with_options=question_with_options,
                max_new_tokens=1024
            )
            
            return answer
        else:
            # Use existing single-model approach
            pass


def format_question_for_stages(example, dataset_name="mmlu-redux"):
    """
    Helper function to format questions for two-stage processing.
    
    Args:
        example: Dataset example
        dataset_name: Name of the dataset
    
    Returns:
        Tuple of (question_without_options, question_with_options)
    """
    if dataset_name == "mmlu-redux":
        question_text = example['question']
        
        # Build choices
        choices = ""
        for i, choice in enumerate(example['choices']):
            choices += f"{chr(65+i)}. {choice}\n"
        
        # Question with full template for answering
        question_with_options = f"""{question_text}

{choices}
Answer: The correct answer is"""
        
        return question_text, question_with_options
    
    elif dataset_name == "mmmlu":
        question_text = example['Question']
        
        # Build choices
        choices = ""
        for i, choice_key in enumerate(['A', 'B', 'C', 'D']):
            if choice_key in example:
                choices += f"{choice_key}. {example[choice_key]}\n"
        
        question_with_options = f"""{question_text}

{choices}
Answer: The correct answer is"""
        
        return question_text, question_with_options
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def example_different_prompts():
    """Example showing different background prompts."""
    print("\n" + "="*60)
    print("Example with Different Background Prompts")
    print("="*60)
    
    # Different prompt styles
    prompts = {
        "concise": "Briefly describe the most useful background to solve the problem:\n\n{question}",
        "detailed": "Analyze the key concepts and provide relevant background information needed to solve this problem:\n\n{question}",
        "step_by_step": "Break down the problem and explain the key concepts step by step:\n\n{question}",
        "domain_focused": "Provide domain-specific knowledge and context relevant to this question:\n\n{question}"
    }
    
    question = "What is the primary function of mitochondria in cells?"
    
    for style, prompt in prompts.items():
        print(f"\n--- {style.upper()} STYLE ---")
        print(f"Prompt: {prompt}")
        print(f"Full prompt: {prompt.format(question=question)}")
        print("-" * 40)


def example_configuration_options():
    """Example showing different configuration options for Rosetta."""
    print("\n" + "="*60)
    print("Configuration Options for TwoStageRosetta")
    print("="*60)
    
    # Example configurations (following load_model_from_checkpoint pattern)
    configs = {
        "basic": {
            "context_model_path": "Qwen/Qwen3-4B",
            "rosetta_checkpoint_dir": "/path/to/checkpoints",
            "rosetta_subfolder": "final",
            "device": "cuda",
            "max_new_tokens": 512
        },
        "different_subfolder": {
            "context_model_path": "Qwen/Qwen3-4B",
            "rosetta_checkpoint_dir": "/path/to/checkpoints", 
            "rosetta_subfolder": "checkpoint-1000",
            "device": "cuda",
            "max_new_tokens": 1024
        },
        "custom_prompt": {
            "context_model_path": "Qwen/Qwen3-4B",
            "rosetta_checkpoint_dir": "/path/to/checkpoints",
            "rosetta_subfolder": "final",
            "device": "cuda",
            "max_new_tokens": 512,
            "background_prompt": "Provide detailed background information for this question:\n\n{question}"
        }
    }
    
    for config_name, config in configs.items():
        print(f"\n--- {config_name.upper()} CONFIG ---")
        for key, value in config.items():
            print(f"{key}: {value}")
        print("-" * 40)


if __name__ == "__main__":
    print("TwoStageRosetta Example")
    print("=" * 50)
    
    # Note: For actual usage, replace model paths with real model names
    print("\nThis script demonstrates the usage of TwoStageRosetta.")
    print("To run with real models, update the model paths and checkpoint directories.")
    
    # Show the formatting example
    example_data = {
        'question': 'What is the capital of France?',
        'choices': ['London', 'Berlin', 'Paris', 'Madrid']
    }
    
    q_without, q_with = format_question_for_stages(example_data)
    print("\nExample Question Formatting:")
    print("\n1. Question without options (for context generation):")
    print(q_without)
    print("\n2. Question with options (for final answer):")
    print(q_with)
    
    example_standalone()
    example_different_prompts()
    example_configuration_options()
