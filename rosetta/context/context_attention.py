import torch
from typing import List, Tuple, Dict, Any

from rosetta.context.utils import top_k_top_p_filtering



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