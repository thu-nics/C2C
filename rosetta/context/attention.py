"""
Contextual attention mask utilities for simulating KV cache dropping.

This module provides functions to build custom 4D attention masks that simulate
dropping conversation rounds. The mask prevents assistant tokens from attending
to dropped rounds while allowing user tokens (prefill) to still see them.

Key insight: Position IDs must be preserved across drops to maintain correct
positional encoding. When we "drop" a round via attention mask, the tokens
still exist in the sequence, so position IDs continue monotonically.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any

from rosetta.context.utils import top_k_top_p_filtering
from rosetta.context.utils import _print_eval_results, _log_to_wandb, HAS_WANDB

def tokenize_conversation_round_by_round(
    tokenizer, 
    messages: List[dict], 
    enable_thinking: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int, int, str, int]]]:
    """
    Tokenize conversation round by round, matching ContextualModel's incremental tokenization.
    
    Pattern (from contextual_chat_example.py):
    - For user: tokenize with add_generation_prompt=True, record user + gen_prompt boundary
    - For assistant: tokenize without gen_prompt, record assistant boundary (content after gen_prompt)
    
    Returns:
        (input_ids, boundaries) where boundaries is list of (start, end, role, msg_id)
    """
    if not messages:
        return torch.empty((1, 0), dtype=torch.long), []

    def _apply(messages_, add_generation_prompt: bool) -> torch.Tensor:
        return tokenizer.apply_chat_template(
            messages_,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    all_input_ids: List[torch.Tensor] = []
    boundaries: List[Tuple[int, int, str, int]] = []

    current_messages: List[dict] = []
    seq_len = 0

    for msg_id, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role: {role}")

        start = seq_len

        if role in ("system", "user"):
            # Add message to the chat-template stream, then slice ONLY the newly-added tokens.
            current_messages.append({"role": role, "content": content})

            full_no_gen = _apply(current_messages, add_generation_prompt=False)
            new_ids = full_no_gen[:, seq_len:]
            if new_ids.numel() > 0:
                all_input_ids.append(new_ids)
                seq_len += new_ids.shape[1]

            # User messages also include the generation prompt tokens, attributed to the same msg_id.
            if role == "user":
                full_with_gen = _apply(current_messages, add_generation_prompt=True)
                gen_prompt_ids = full_with_gen[:, seq_len:]
                if gen_prompt_ids.numel() > 0:
                    all_input_ids.append(gen_prompt_ids)
                    seq_len += gen_prompt_ids.shape[1]

            boundaries.append((start, seq_len, role, msg_id))
            continue

        # Assistant: before appending assistant content, ensure the stream up to the previous user
        # has already contributed its generation-prompt tokens (handled in the "user" branch above).
        #
        # Then append assistant tokens directly after the stream (no chat-template wrapper here).
        assistant_ids = tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
        if assistant_ids.numel() > 0:
            all_input_ids.append(assistant_ids)
            seq_len += assistant_ids.shape[1]

        boundaries.append((start, seq_len, role, msg_id))
        current_messages.append({"role": "assistant", "content": content})

    if not all_input_ids:
        return torch.empty((1, 0), dtype=torch.long), boundaries
    return torch.cat(all_input_ids, dim=1), boundaries


def get_round_boundaries(tokenizer, messages, max_length: int = 2048) -> List[Tuple[int, int, str, int]]:
    """
    Deprecated: use tokenize_conversation_round_by_round instead

    Get token boundaries for each message in the conversation.
    
    Args:
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role' and 'content'
        max_length: Maximum sequence length
        
    Returns:
        List of (start_idx, end_idx, role, msg_id) tuples.
        msg_id is the message index (0, 1, 2, ...) - each message gets its own ID.
        This matches the ID assignment in ContextualModel.generate_step().
    """
    boundaries = []
    current_pos = 0
    
    for i, msg in enumerate(messages):
        partial = messages[:i+1]
        text = tokenizer.apply_chat_template(
            partial,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        tokens = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=max_length, add_special_tokens=False)
        end_pos = tokens.input_ids.shape[1]
        msg_id = i  # Each message gets its own ID (matching generate_step behavior)
        boundaries.append((current_pos, end_pos, msg["role"], msg_id))
        current_pos = end_pos
    
    return boundaries
    
def build_contextual_attention_mask(
    seq_len: int,
    msg_boundaries: List[Tuple[int, int, str, int]],
    messages_to_drop: Optional[Dict[int, List[int]]] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a custom 4D attention mask that simulates KV dropping.
    
    This matches the behavior of ContextualModel where dropped messages are
    permanently removed from the cache.
    
    Rules (for messages in messages_to_drop[X] dropped at message X):
    - Messages with msg_id <= X CAN see dropped messages (before/at drop point)
    - Messages with msg_id > X CANNOT see dropped messages (after drop point)
    
    Args:
        seq_len: Total sequence length
        msg_boundaries: List of (start_idx, end_idx, role, msg_id)
        messages_to_drop: Dict mapping {msg_id_when_drop_happens: [msg_ids_to_drop]}.
                         E.g., {3: [1, 2]} means drop messages 1 and 2 at message 3.
                         Messages 0-3 can see 1,2; messages 4+ cannot see 1,2.
        device: Torch device
        dtype: Torch dtype
    
    Returns:
        attention_mask: (1, 1, seq_len, seq_len) mask
        Values: 0.0 = can attend, -inf = cannot attend
    """
    # Start with causal mask
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
        diagonal=1
    )
    
    if messages_to_drop is None or len(messages_to_drop) == 0:
        return mask.unsqueeze(0).unsqueeze(0)
    
    # Build effective drop mapping: for each dropped message, at which message was it dropped?
    # drop_info[dropped_msg_id] = msg_id_when_it_was_dropped
    drop_info: Dict[int, int] = {}
    
    for drop_at_msg_id, dropped_msg_ids in messages_to_drop.items():
        for dm in dropped_msg_ids:
            # If a message is dropped at multiple points, use the earliest
            if dm not in drop_info or drop_at_msg_id < drop_info[dm]:
                drop_info[dm] = drop_at_msg_id
    
    # Apply masking for each dropped message
    for dropped_msg_id, dropped_at_msg_id in drop_info.items():
        # Find the token range of the dropped message
        drop_start, drop_end = None, None
        for start, end, role, msg_id in msg_boundaries:
            if msg_id == dropped_msg_id:
                drop_start = start
                drop_end = end
                break
        
        if drop_start is None:
            continue
        
        # Apply masking based on when the drop happened
        # The drop happens AT dropped_at_msg_id, meaning:
        # - msg_id <= dropped_at_msg_id: CAN see dropped messages (before/at drop point)
        # - msg_id > dropped_at_msg_id: CANNOT see dropped messages (after drop)
        for start, end, role, msg_id in msg_boundaries:
            if msg_id > dropped_at_msg_id:
                # All messages after the drop point cannot see dropped messages
                mask[start:end, drop_start:drop_end] = float('-inf')
    
    return mask.unsqueeze(0).unsqueeze(0)


def generate_with_contextual_mask(
    model,
    tokenizer,
    messages: List[dict],
    drop_ids: Optional[Dict[int, List[int]]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> str:
    """
    Generate a response using contextual attention mask to simulate dropping.
    
    This is the unified generation function that should be used for both training
    evaluation (with dropping) and standard generation (without dropping) to ensure
    consistency.
    
    Behavior matches ContextualModel.generate_step():
    - User tokens (prefill) CAN see messages to be dropped
    - Assistant tokens (generated) CANNOT see dropped messages
    - Message IDs are assigned per-message (0, 1, 2, ...), matching generate_step
    
    Args:
        model: The model (unwrapped)
        tokenizer: Tokenizer
        messages: Conversation messages (ending with a user message)
        messages_to_drop: Dict mapping {msg_id_when_drop_happens: [msg_ids_to_drop]}.
                         E.g., {2: [0, 1]} means drop messages 0 and 1 when generating
                         the response after message 2.
                         If None or empty, standard generation.
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated response string
    """
    """
    One-shot prefill + decode generation that simulates KV-dropping via attention masks.

    This function is intentionally a *single-turn* primitive: it generates the assistant
    response for a conversation that ends with a user message. Multi-round behavior is
    achieved by calling this function repeatedly outside (as in the examples).

    Semantics (matching `ContextualModel.generate_step` / examples.yaml ID scheme):
    - Let `input_id` be the ID of the last user message (the last element in `messages`).
    - The generation prompt tokens are treated as part of `input_id`.
    - Generated assistant tokens are treated as `output_id = input_id + 1`.
    - Drops at key `input_id` are applied AFTER prefill of the generation prompt:
        - Prefill uses drops with drop_at < input_id
        - Decode uses drops with drop_at <= input_id
    """
    ctx = prepare_context(
        tokenizer=tokenizer,
        messages=messages,
        drop_ids=drop_ids,
        enable_thinking=False,
    )
    gen_ids = generate(
        model=model,
        input_ids=ctx["input_ids"],
        generation_config={"max_new_tokens": max_new_tokens, "eos_token_id": tokenizer.eos_token_id},
        left_padding=None,
        context_kwargs=ctx,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def prepare_context(
    tokenizer,
    messages: List[dict],
    drop_ids: Optional[Dict[int, List[int]]] = None,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """
    Prepare a single-turn generation context (prefill ids + boundaries + drop metadata).

    The conversation must end with a user message. The returned dict is intended to be
    passed as `context_kwargs` into `generate(...)`.
    """
    if drop_ids is None:
        drop_ids = {}

    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must be non-empty and end with a user message.")

    # Message IDs are the message indices in `messages` (0, 1, 2, ...).
    input_id = len(messages) - 1

    # Tokenize conversation round by round (user includes gen_prompt, assistant is content only).
    input_ids, boundaries = tokenize_conversation_round_by_round(
        tokenizer, messages, enable_thinking=enable_thinking,
    )

    # Drops are keyed by the user-message ID at which the drop happens.
    # Prefill happens before applying the drop-at-input_id; decode happens after.
    prefill_drop = {int(k): v for k, v in drop_ids.items() if int(k) < input_id} or None
    decode_drop = {int(k): v for k, v in drop_ids.items() if int(k) <= input_id} or None

    return {
        "messages": messages,
        "input_id": input_id,
        "drop_ids": drop_ids,
        "prefill_drop": prefill_drop,
        "decode_drop": decode_drop,
        "boundaries": boundaries,
        "input_ids": input_ids,
    }


def generate(
    model,
    input_ids,
    generation_config=None,
    left_padding=None,
    context_kwargs=None,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    **kwargs,
):
    """
    One-shot prefill + decode generation that simulates KV-dropping via attention masks.

    Args:
        model: HF causal LM (unwrapped).
        input_ids: Prefill tokens (1, seq_len).
        generation_config: dict-like or HF GenerationConfig. Supports max_new_tokens, eos_token_id.
        left_padding: reserved for future batching; must be None for now.
        context_kwargs: dict containing at least: boundaries, prefill_drop, decode_drop.
        **kwargs: reserved for future extensions.

    Returns:
        List[int] of generated token IDs (excluding the prompt).
    """
    if context_kwargs is None:
        raise ValueError("context_kwargs is required (needs boundaries/drop info).")
    if left_padding is not None:
        raise NotImplementedError("left_padding is not supported in this single-sample generator.")

    max_new_tokens = 256
    eos_token_id = getattr(generation_config, "eos_token_id", None) if generation_config is not None else None
    if isinstance(generation_config, dict):
        max_new_tokens = int(generation_config.get("max_new_tokens", max_new_tokens))
        eos_token_id = generation_config.get("eos_token_id", eos_token_id)
    elif generation_config is not None:
        max_new_tokens = int(getattr(generation_config, "max_new_tokens", max_new_tokens))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    boundaries: List[Tuple[int, int, str, int]] = context_kwargs["boundaries"]
    prefill_drop = context_kwargs.get("prefill_drop")
    decode_drop = context_kwargs.get("decode_drop")

    # Prefill with a 4D contextual mask
    attn_mask_4d = build_contextual_attention_mask(
        seq_len=seq_len,
        msg_boundaries=boundaries,
        messages_to_drop=prefill_drop,
        device=device,
        dtype=dtype,
    )

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attn_mask_4d,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    # For decode, compute dropped message spans (columns) that the generated tokens must not attend to.
    dropped_msg_set = set()
    if decode_drop:
        for _, ids in decode_drop.items():
            dropped_msg_set.update(ids)

    dropped_spans: List[Tuple[int, int]] = []
    if dropped_msg_set:
        for start, end, _, mid in boundaries:
            if mid in dropped_msg_set:
                dropped_spans.append((start, end))

    generated_ids: List[int] = []
    current_pos = seq_len

    for _ in range(max_new_tokens):
        if float(temperature) > 0.0:
            logits = next_token_logits / float(temperature)
            logits = top_k_top_p_filtering(logits, top_k=int(top_k), top_p=float(top_p))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        if eos_token_id is not None and token_id == int(eos_token_id):
            break

        generated_ids.append(token_id)

        new_mask_row = torch.zeros(1, 1, 1, current_pos + 1, device=device, dtype=dtype)
        for start, end in dropped_spans:
            new_mask_row[:, :, :, start:end] = float("-inf")

        new_position_ids = torch.tensor([[current_pos]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                position_ids=new_position_ids,
                attention_mask=new_mask_row,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        current_pos += 1

    return generated_ids


@torch.no_grad()
def run_evaluation(
    model,
    tokenizer,
    eval_examples: List[Dict[str, Any]],
    accelerator,
    global_step: int,
    max_new_tokens: int = 256,
    use_drop: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate responses for evaluation examples with sequential multi-round generation.
    Each round's response is generated by the model and used as context for the next round.
    
    YAML format expected:
        - system_prompt: optional system prompt (ID 0 if present)
        - user_messages: list of user message strings
        - drop_messages: {msg_id_when_drop_happens: [msg_ids_to_drop]}
    
    Message IDs (matching generate_step):
        - ID 0 = system prompt (if present)
        - ID 1 = first user message
        - ID 2 = first assistant response
        - ID 3 = second user message
        - etc.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        eval_examples: List of evaluation examples from YAML
        accelerator: Accelerator instance
        global_step: Current training step
        max_new_tokens: Max tokens to generate per response
        use_drop: If True, apply drop_messages using contextual attention mask
        
    Returns:
        List of evaluation results
    """
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    eval_results = []
    
    for example in eval_examples:
        name = example.get("name", "unnamed")
        system_prompt = example.get("system_prompt", None)
        user_messages = example.get("user_messages", [])
        # drop_messages: {msg_id_when_drop_happens: [msg_ids_to_drop]}
        drop_messages = example.get("drop_messages", {})
        
        if not user_messages:
            continue
        
        # Generate responses sequentially, using model's own responses as context
        # Include system prompt if present (ID 0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        generated_responses = []
        
        for turn_idx, user_msg in enumerate(user_messages):
            # Add user message
            messages.append({"role": "user", "content": user_msg})
            
            # Current message ID = len(messages) - 1 (0-indexed)
            # With system prompt: ID 0 = system, ID 1 = first user, etc.
            current_msg_id = len(messages) - 1
            
            # Determine which messages to drop for THIS turn
            # drop_messages format: {msg_id_when_drop_happens: [msg_ids_to_drop]}
            if use_drop and drop_messages:
                # Pass the full drop_messages dict; generate_with_contextual_mask
                # will handle accumulating drops based on current_msg_id
                effective_drop_messages = drop_messages
            else:
                effective_drop_messages = {}
            
            response = generate_with_contextual_mask(
                unwrapped_model, tokenizer, messages, effective_drop_messages, max_new_tokens
            )
            
            # Add model's response to conversation context
            messages.append({"role": "assistant", "content": response})
            generated_responses.append(response)
        
        eval_results.append({
            "name": name,
            "user_messages": user_messages,
            "generated_responses": generated_responses,
            "drop_messages": drop_messages,
            "full_conversation": messages,
        })
    
    # Log results
    if accelerator.is_main_process and eval_results:
        _print_eval_results(eval_results, global_step, use_drop)
        if HAS_WANDB:
            try:
                _log_to_wandb(accelerator, eval_results, global_step)
            except Exception as e:
                print(f"Failed to log to wandb: {e}")
    
    model.train()
    return eval_results

