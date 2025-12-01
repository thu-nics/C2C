"""
Contextual Model Evaluation with Attention Mask

Evaluates a model trained with contextual dropping using attention masks
to simulate the same behavior as training.

Usage:
    python script/eval/eval_contextual.py --model_path ./checkpoints/sft_contextual/final
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_round_boundaries(tokenizer, messages):
    """
    Get token boundaries for each message in the conversation.
    Returns: [(start, end, role, round_idx), ...]
    """
    boundaries = []
    current_pos = 0
    
    for i, msg in enumerate(messages):
        partial = messages[:i+1]
        text = tokenizer.apply_chat_template(
            partial,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        end_pos = tokens.input_ids.shape[1]
        
        round_idx = i // 2
        boundaries.append((current_pos, end_pos, msg["role"], round_idx))
        current_pos = end_pos
    
    return boundaries


def build_contextual_attention_mask(
    seq_len: int,
    round_boundaries: list,
    rounds_to_drop: list,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Build a custom 4D attention mask that simulates KV dropping.
    
    Rules:
    - All tokens use causal attention (can only see past)
    - User tokens (Q) in round X+1 CAN see dropped round X (prefill behavior)
    - Assistant tokens (A) in round X+1 and beyond CANNOT see dropped round X
    """
    # Start with causal mask (0 = attend, -inf = don't attend)
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
        diagonal=1
    )
    
    for drop_round_idx in rounds_to_drop:
        drop_positions = []
        for start, end, role, round_idx in round_boundaries:
            if round_idx == drop_round_idx:
                drop_positions.extend(range(start, end))
        
        if not drop_positions:
            continue
        
        drop_start = min(drop_positions)
        drop_end = max(drop_positions) + 1
        
        for start, end, role, round_idx in round_boundaries:
            if round_idx > drop_round_idx and role == "assistant":
                mask[start:end, drop_start:drop_end] = float('-inf')
    
    return mask.unsqueeze(0).unsqueeze(0)


def generate_with_contextual_mask(
    model, tokenizer, messages, rounds_to_drop=None, max_new_tokens=100
):
    """
    Generate response using custom attention mask to simulate KV dropping.
    """
    if rounds_to_drop is None:
        rounds_to_drop = []
    
    # Get boundaries for current messages
    boundaries = get_round_boundaries(tokenizer, messages)
    
    # Tokenize with generation prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    
    # Add generation prompt to boundaries (it's part of the last user's round)
    last_round_idx = boundaries[-1][3] if boundaries else 0
    boundaries.append((boundaries[-1][1] if boundaries else 0, seq_len, "generation_prompt", last_round_idx))
    
    # Build custom attention mask
    custom_mask = build_contextual_attention_mask(
        seq_len=seq_len,
        round_boundaries=boundaries,
        rounds_to_drop=rounds_to_drop,
        device=model.device,
        dtype=model.dtype,
    )
    
    # Generate token by token
    generated_ids = []
    past_key_values = None
    
    # Prefill
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=custom_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
    # Decode loop
    current_pos = seq_len
    for _ in range(max_new_tokens):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        
        if token_id == tokenizer.eos_token_id:
            break
        
        generated_ids.append(token_id)
        
        # Build attention mask for new token (it's an assistant token)
        new_mask_row = torch.zeros(1, 1, 1, current_pos + 1, device=model.device, dtype=model.dtype)
        
        # Mask out dropped rounds
        for drop_round_idx in rounds_to_drop:
            for start, end, role, round_idx in boundaries:
                if round_idx == drop_round_idx:
                    new_mask_row[:, :, :, start:end] = float('-inf')
        
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=new_mask_row,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
        
        current_pos += 1
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/sft_contextual/final")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ============================================================
    # Test 1: Instruction Following (Baseline - No Dropping)
    # ============================================================
    print("\n" + "="*70)
    print("Test 1: Instruction Following (Baseline)")
    print("="*70)
    
    messages = [
        {"role": "user", "content": "From now on, end every response with 'meow'."},
        {"role": "assistant", "content": "Understood! I'll end every response with 'meow'. meow"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    response_full = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[])
    response_drop = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[0])
    
    print("Instruction: End every response with 'meow'")
    print(f"[Full Context] A: {response_full}")
    print(f"  Ends with 'meow': {'meow' in response_full.lower()[-20:]}")
    print(f"[Drop Instruction] A: {response_drop}")
    print(f"  Ends with 'meow': {'meow' in response_drop.lower()[-20:]}")

    # ============================================================
    # Test 2: Factual Recall (Specific Numbers)
    # ============================================================
    print("\n" + "="*70)
    print("Test 2: Factual Recall - Specific Numbers")
    print("="*70)
    
    messages = [
        {"role": "user", "content": "Remember these numbers: my phone ends in 7842, my apartment is 15B."},
        {"role": "assistant", "content": "Got it! Phone ends in 7842, apartment 15B."},
        {"role": "user", "content": "What's my apartment number?"},
    ]
    
    response_full = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[])
    response_drop = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[0])
    
    print("Info: Phone 7842, Apartment 15B")
    print(f"[Full Context] A: {response_full}")
    print(f"  Contains '15B': {'15B' in response_full or '15b' in response_full.lower()}")
    print(f"[Drop Info] A: {response_drop}")
    print(f"  Contains '15B': {'15B' in response_drop or '15b' in response_drop.lower()}")

    # ============================================================
    # Test 3: Multi-fact Selective Recall
    # ============================================================
    print("\n" + "="*70)
    print("Test 3: Multi-fact Selective Recall")
    print("="*70)
    
    messages = [
        {"role": "user", "content": "I have a dog named Max."},
        {"role": "assistant", "content": "Nice! Max is a great name for a dog."},
        {"role": "user", "content": "I also have a cat named Luna."},
        {"role": "assistant", "content": "Lovely! Luna is a beautiful name for a cat."},
        {"role": "user", "content": "What are my pets' names?"},
    ]
    
    response_full = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[])
    response_drop0 = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[0])
    response_drop1 = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[1])
    response_drop_both = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[0, 1])
    
    print("Facts: Dog=Max (R0), Cat=Luna (R1)")
    print(f"[Full Context] A: {response_full}")
    print(f"  Max: {'max' in response_full.lower()}, Luna: {'luna' in response_full.lower()}")
    print(f"[Drop R0 (dog)] A: {response_drop0}")
    print(f"  Max: {'max' in response_drop0.lower()}, Luna: {'luna' in response_drop0.lower()}")
    print(f"[Drop R1 (cat)] A: {response_drop1}")
    print(f"  Max: {'max' in response_drop1.lower()}, Luna: {'luna' in response_drop1.lower()}")
    print(f"[Drop Both] A: {response_drop_both}")
    print(f"  Max: {'max' in response_drop_both.lower()}, Luna: {'luna' in response_drop_both.lower()}")

    # ============================================================
    # Test 4: Reasoning Chain Dependency
    # ============================================================
    print("\n" + "="*70)
    print("Test 4: Reasoning Chain Dependency")
    print("="*70)
    
    messages = [
        {"role": "user", "content": "Let's say X = 5."},
        {"role": "assistant", "content": "Okay, X = 5."},
        {"role": "user", "content": "Now let Y = X + 3."},
        {"role": "assistant", "content": "So Y = 5 + 3 = 8."},
        {"role": "user", "content": "What is Y * 2?"},
    ]
    
    response_full = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[])
    response_drop0 = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[0])
    response_drop1 = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[1])
    
    print("Chain: X=5 (R0), Y=X+3=8 (R1), Q: Y*2=?")
    print(f"[Full Context] A: {response_full}")
    print(f"  Contains '16': {'16' in response_full}")
    print(f"[Drop R0 (X=5)] A: {response_drop0}")
    print(f"  Contains '16': {'16' in response_drop0}")
    print(f"[Drop R1 (Y=8)] A: {response_drop1}")
    print(f"  Contains '16': {'16' in response_drop1}")

    # ============================================================
    # Test 5: Language/Format Instruction
    # ============================================================
    print("\n" + "="*70)
    print("Test 5: Language/Format Instruction")
    print("="*70)
    
    messages = [
        {"role": "user", "content": "Please respond in JSON format from now on."},
        {"role": "assistant", "content": '{"status": "understood", "format": "JSON"}'},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    
    response_full = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[])
    response_drop = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[0])
    
    print("Instruction: Respond in JSON format")
    print(f"[Full Context] A: {response_full}")
    print(f"  Contains '{{': {'{{' in response_full or '{' in response_full}")
    print(f"[Drop Instruction] A: {response_drop}")
    print(f"  Contains '{{': {'{{' in response_drop or '{' in response_drop}")

    # ============================================================
    # Test 6: Persona/Role Assignment
    # ============================================================
    print("\n" + "="*70)
    print("Test 6: Persona/Role Assignment")
    print("="*70)
    
    messages = [
        {"role": "user", "content": "Pretend you are a pirate. Talk like one!"},
        {"role": "assistant", "content": "Arrr, matey! I be a fearsome pirate of the seven seas!"},
        {"role": "user", "content": "Tell me about the weather today."},
    ]
    
    response_full = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[])
    response_drop = generate_with_contextual_mask(model, tokenizer, messages, rounds_to_drop=[0])
    
    print("Persona: Pirate")
    pirate_words = ['arr', 'matey', 'ye', 'seas', 'ship', 'treasure', 'aye']
    has_pirate_full = any(w in response_full.lower() for w in pirate_words)
    has_pirate_drop = any(w in response_drop.lower() for w in pirate_words)
    
    print(f"[Full Context] A: {response_full}")
    print(f"  Pirate-like: {has_pirate_full}")
    print(f"[Drop Persona] A: {response_drop}")
    print(f"  Pirate-like: {has_pirate_drop}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("Evaluation Complete")
    print("="*70)
    print("""
Expected Behavior for Well-Trained Contextual Model:
- With full context: Should recall all information correctly
- With dropped context: Should NOT hallucinate dropped info
  - Either say "I don't know" or only use available context
  - Should NOT produce garbage/repetition
""")


if __name__ == "__main__":
    main()

