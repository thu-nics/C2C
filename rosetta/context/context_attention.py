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
    messages_to_drop: Optional[Dict[int, List[int]]] = None,
    max_new_tokens: int = 256,
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

    if messages_to_drop is None:
        messages_to_drop = {}

    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must be non-empty and end with a user message.")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    input_id = len(messages) - 1
    output_id = input_id + 1  # (kept for clarity; output tokens conceptually have this id)

    # Normalize drop dict keys to ints (YAML gives ints; some callers may pass strings)
    drop_events: Dict[int, List[int]] = messages_to_drop 
    
    # Tokenize conversation round by round (user includes gen_prompt, assistant is content only)
    input_ids, boundaries = tokenize_conversation_round_by_round(
        tokenizer, messages, enable_thinking=False,
    )
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    # Prefill happens before applying drop at input_id; decode happens after.
    prefill_drop = {k: v for k, v in drop_events.items() if k < input_id} or None
    decode_drop = {k: v for k, v in drop_events.items() if k <= input_id} or None

    # One-shot prefill with a 4D contextual mask
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

    # For decode, compute the dropped message spans (columns) that output_id must not attend to.
    dropped_msg_ids: List[int] = []
    if decode_drop:
        for _, ids in decode_drop.items():
            dropped_msg_ids.extend(ids)
    dropped_msg_set = set(dropped_msg_ids)

    dropped_spans: List[Tuple[int, int]] = []
    if dropped_msg_set:
        for start, end, _, mid in boundaries:
            if mid in dropped_msg_set:
                dropped_spans.append((start, end))

    # Decode token by token
    generated_ids: List[int] = []
    current_pos = seq_len

    for _ in range(max_new_tokens):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)

        # Build attention mask row for the new assistant token (conceptually msg_id=output_id)
        # It must not attend to dropped message columns (after drop is applied).
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

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


class ContextualAttentionModel:
    """
    Wrapper for attention-mask-based contextual generation.
    
    Unlike ContextualModel (which manipulates KV cache directly), this class
    simulates dropping by building custom attention masks. Position IDs are
    preserved monotonically, matching the behavior of ContextualModel.
    
    Key difference from ContextualModel:
    - ContextualModel: Actually removes KV cache entries
    - ContextualAttentionModel: Keeps all KV entries but masks attention
    
    Both preserve position IDs for correct positional encoding.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.past_key_values = None
        self.round_ids = []          # Segment ID for each cached token
        self.position_ids = []       # Position ID for each cached token
        self._next_position = 0      # Monotonically increasing
        self._seq_length = 0         # Total tokens processed (for slicing)
        self.dropped_rounds = set()  # Rounds that are "dropped" (masked out)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def cache_length(self):
        """Actual number of tokens in KV cache (same as seq_length for attention masking)."""
        return len(self.round_ids)
    
    @property
    def seq_length(self):
        """Total tokens processed from conversation (use this for slicing new tokens)."""
        return self._seq_length
    
    @property
    def next_position(self):
        """Next position ID to use (monotonically increasing)."""
        return self._next_position
    
    def _get_past_length(self):
        if self.past_key_values is None:
            return 0
        return self.past_key_values.get_seq_length()
    
    def reset(self):
        """Clear all cached state."""
        self.past_key_values = None
        self.round_ids = []
        self.position_ids = []
        self._next_position = 0
        self._seq_length = 0
        self.dropped_rounds = set()
    
    def tokenize(self, text):
        """Tokenize text and move to model device."""
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
    
    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        """Apply chat template and return token IDs on model device."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )
        return self.tokenize(text)
    
    def _build_attention_mask(self, new_seq_len: int) -> torch.Tensor:
        """
        Build attention mask for new tokens, respecting dropped rounds.
        
        Args:
            new_seq_len: Number of new tokens being added
            
        Returns:
            Attention mask of shape (1, 1, new_seq_len, past_len + new_seq_len)
        """
        past_len = self._get_past_length()
        total_len = past_len + new_seq_len
        dtype = next(self.model.parameters()).dtype
        
        # Start with mask that allows attending to everything in the past
        # Shape: (1, 1, new_seq_len, total_len)
        mask = torch.zeros(1, 1, new_seq_len, total_len, device=self.device, dtype=dtype)
        
        # Apply causal masking within new tokens (if more than 1)
        if new_seq_len > 1:
            causal_mask = torch.triu(
                torch.full((new_seq_len, new_seq_len), float('-inf'), device=self.device, dtype=dtype),
                diagonal=1
            )
            mask[:, :, :, past_len:] = causal_mask
        
        # Mask out dropped rounds
        for i, rid in enumerate(self.round_ids):
            if rid in self.dropped_rounds:
                mask[:, :, :, i] = float('-inf')
        
        return mask
    
    def append(self, input_ids, id: int):
        """
        Prefill: expand KV cache without generation.
        
        Args:
            input_ids: Token IDs to process (1D or 2D tensor).
            id: Segment ID to assign to these tokens.
        Returns:
            Model outputs (including logits).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        seq_len = input_ids.shape[1]
        
        # Compute position IDs (continue from last position)
        start_pos = self.next_position
        position_ids = torch.arange(
            start_pos, start_pos + seq_len,
            dtype=torch.long, device=self.device
        ).unsqueeze(0)
        
        # Build attention mask respecting dropped rounds
        attention_mask = self._build_attention_mask(seq_len)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=self.past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        
        self.past_key_values = outputs.past_key_values
        self.round_ids.extend([id] * seq_len)
        self.position_ids.extend(range(start_pos, start_pos + seq_len))
        self._next_position = start_pos + seq_len
        self._seq_length += seq_len  # Track total tokens processed
        
        return outputs
    
    def generate_step(
        self,
        input_ids,
        input_id: int,
        output_id: int = None,
        max_new_tokens=50,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        drop_ids_after_prefill=None,
    ):
        """
        Generate response using manual decoding loop with attention masking.
        
        Args:
            input_ids: Prompt tokens (will be appended to cache first).
            input_id: Segment ID for the input/prompt tokens.
            output_id: Segment ID for the generated tokens. If None, uses input_id.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold (1.0 disables).
            top_k: Top-k sampling threshold (0 disables).
            drop_ids_after_prefill: IDs to mark as dropped after prefilling.
                These rounds will be masked out during generation.
        Returns:
            Generated text (excluding prompt).
        """
        if output_id is None:
            output_id = input_id
            
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot be empty.")
        
        # Prefill prompt (with full context - can still see dropped rounds during prefill)
        outputs = self.append(input_ids, id=input_id)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Mark rounds as dropped (they'll be masked out during generation)
        if drop_ids_after_prefill is not None and len(drop_ids_after_prefill) > 0:
            self.drop_context(drop_ids_after_prefill)
        
        generated_ids = []
        
        for _ in range(max_new_tokens):
            # Sample next token
            if temperature > 0:
                logits = next_token_logits / temperature
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            token_id = next_token.item()
            
            if token_id == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(token_id)
            
            # Append token to cache with output_id (different from input_id)
            outputs = self.append(next_token, id=output_id)
            next_token_logits = outputs.logits[:, -1, :]
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def drop_context(self, remove_ids):
        """
        Mark rounds as dropped (will be masked out in attention).
        
        Unlike ContextualModel, this doesn't actually remove KV cache entries.
        Instead, it marks rounds so they're masked out in future attention.
        
        Args:
            remove_ids: Single ID or list of IDs to drop.
        """
        if not isinstance(remove_ids, list):
            remove_ids = [remove_ids]
        
        # Check if any of these IDs exist
        existing_ids = set(self.round_ids)
        valid_ids = [rid for rid in remove_ids if rid in existing_ids]
        
        if not valid_ids:
            print(f"No rounds with IDs {remove_ids} found.")
            return
        
        self.dropped_rounds.update(valid_ids)
        
        # Count tokens that are now masked
        masked_count = sum(1 for rid in self.round_ids if rid in self.dropped_rounds)
        print(f"Dropped IDs {remove_ids}. Masked: {masked_count} tokens, Cache: {len(self.round_ids)} tokens.")


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

