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
from typing import List, Tuple, Optional, Dict


def get_round_boundaries(tokenizer, messages, max_length: int = 2048) -> List[Tuple[int, int, str, int]]:
    """
    Get token boundaries for each message in the conversation.
    
    Args:
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role' and 'content'
        max_length: Maximum sequence length
        
    Returns:
        List of (start_idx, end_idx, role, round_idx) tuples.
        round_idx groups user+assistant pairs: user[0], assistant[0] -> round 0
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
        round_idx = i // 2
        boundaries.append((current_pos, end_pos, msg["role"], round_idx))
        current_pos = end_pos
    
    return boundaries


def build_contextual_attention_mask(
    seq_len: int,
    round_boundaries: List[Tuple[int, int, str, int]],
    rounds_to_drop: List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    drop_at_round: Optional[Dict[int, List[int]]] = None,
) -> torch.Tensor:
    """
    Build a custom 4D attention mask that simulates KV dropping.
    
    This matches the behavior of ContextualModel where dropped rounds are
    permanently removed from the cache.
    
    Rules (for a round dropped at round X):
    - User tokens in round X CAN see dropped round (prefill behavior)
    - Assistant tokens in round X CANNOT see dropped round (generation behavior)
    - ALL tokens in rounds > X CANNOT see dropped round (it's gone!)
    
    Args:
        seq_len: Total sequence length
        round_boundaries: List of (start_idx, end_idx, role, round_idx)
        rounds_to_drop: List of round indices to drop (simple mode: all dropped at last round)
        device: Torch device
        dtype: Torch dtype
        drop_at_round: Optional dict mapping {round_where_drop_happens: [rounds_to_drop]}.
                       If provided, overrides rounds_to_drop for fine-grained control.
                       E.g., {2: [1]} means round 1 is dropped when generating round 2.
    
    Returns:
        attention_mask: (1, 1, seq_len, seq_len) mask
        Values: 0.0 = can attend, -inf = cannot attend
    """
    # Start with causal mask
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
        diagonal=1
    )
    
    # Build effective drop mapping: for each dropped round, at which round was it dropped?
    # drop_info[dropped_round] = round_where_it_was_dropped
    drop_info: Dict[int, int] = {}
    
    if drop_at_round is not None:
        # Fine-grained control: {round_where_drop_happens: [rounds_to_drop]}
        for drop_at, dropped_rounds in drop_at_round.items():
            for dr in dropped_rounds:
                drop_info[dr] = drop_at
    else:
        # Simple mode: all rounds in rounds_to_drop are dropped at the last round
        max_round = max(b[3] for b in round_boundaries) if round_boundaries else 0
        for dr in rounds_to_drop:
            drop_info[dr] = max_round
    
    # Apply masking for each dropped round
    for drop_round_idx, dropped_at_round in drop_info.items():
        # Find the token range of the dropped round
        drop_positions = []
        for start, end, role, round_idx in round_boundaries:
            if round_idx == drop_round_idx:
                drop_positions.extend(range(start, end))
        
        if not drop_positions:
            continue
        
        drop_start = min(drop_positions)
        drop_end = max(drop_positions) + 1
        
        # Apply masking based on when the drop happened
        for start, end, role, round_idx in round_boundaries:
            if round_idx == dropped_at_round:
                # Round where drop happens:
                # - User tokens CAN see (prefill)
                # - Assistant tokens CANNOT see (generation after drop)
                if role == "assistant":
                    mask[start:end, drop_start:drop_end] = float('-inf')
            elif round_idx > dropped_at_round:
                # All subsequent rounds: dropped round is gone
                mask[start:end, drop_start:drop_end] = float('-inf')
    
    return mask.unsqueeze(0).unsqueeze(0)


def generate_with_contextual_mask(
    model,
    tokenizer,
    messages: List[dict],
    rounds_to_drop: Optional[List[int]] = None,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate a response using contextual attention mask to simulate dropping.
    
    This is the unified generation function that should be used for both training
    evaluation (with dropping) and standard generation (without dropping) to ensure
    consistency.
    
    Behavior matches ContextualModel:
    - User tokens in current round CAN see dropped rounds (prefill)
    - Assistant tokens (generated) CANNOT see dropped rounds
    
    Args:
        model: The model (unwrapped)
        tokenizer: Tokenizer
        messages: Conversation messages (ending with a user message)
        rounds_to_drop: Which rounds are already dropped. If None or empty, standard generation.
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated response string
    """
    if rounds_to_drop is None:
        rounds_to_drop = []
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Tokenize conversation with generation prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    
    # Get round boundaries
    boundaries = get_round_boundaries(tokenizer, messages)
    
    # Determine current round (the one we're generating for)
    # The last message should be a user message, and we're generating the assistant response
    # Round index = message_index // 2, so for user at index 2, round = 1
    # The assistant response will be in the same round as the last user message
    current_round = max(b[3] for b in boundaries) if boundaries else 0
    
    # Build drop_at_round: all specified rounds are dropped at current round
    # This means: user tokens in current round CAN see them (prefill), 
    # but the assistant tokens we generate CANNOT see them
    drop_at_round = {current_round: rounds_to_drop} if rounds_to_drop else None
    
    # Build contextual attention mask
    attn_mask_4d = build_contextual_attention_mask(
        seq_len=seq_len,
        round_boundaries=boundaries,
        rounds_to_drop=[],  # Not used when drop_at_round is provided
        device=device,
        dtype=dtype,
        drop_at_round=drop_at_round,
    )
    
    # Prefill with custom mask
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask_4d,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
    # Generate token by token
    generated_ids = []
    current_pos = seq_len
    
    for _ in range(max_new_tokens):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        
        if token_id == tokenizer.eos_token_id:
            break
        
        generated_ids.append(token_id)
        
        # Build attention mask row for new assistant token
        # It cannot attend to dropped rounds (if any)
        new_mask_row = torch.zeros(1, 1, 1, current_pos + 1, device=device, dtype=dtype)
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
    
    def generate(self, input_ids, id: int, max_new_tokens=50, temperature=0.0, drop_ids_after_prefill=None):
        """
        Generate response using manual decoding loop with attention masking.
        
        Args:
            input_ids: Prompt tokens (will be appended to cache first).
            id: Segment ID for both prompt and generated tokens.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            drop_ids_after_prefill: IDs to mark as dropped after prefilling.
                These rounds will be masked out during generation.
        Returns:
            Generated text (excluding prompt).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot be empty.")
        
        # Prefill prompt (with full context - can still see dropped rounds during prefill)
        outputs = self.append(input_ids, id=id)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Mark rounds as dropped (they'll be masked out during generation)
        if drop_ids_after_prefill is not None and len(drop_ids_after_prefill) > 0:
            self.drop_context(drop_ids_after_prefill)
        
        generated_ids = []
        
        for _ in range(max_new_tokens):
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            token_id = next_token.item()
            
            if token_id == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(token_id)
            
            # Append token to cache and get next logits
            outputs = self.append(next_token, id=id)
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




