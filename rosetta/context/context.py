from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.context.utils import top_k_top_p_filtering

class ContextualModel:
    """Wrapper around HF model for explicit KV cache management with segment IDs.
    
    Tracks both segment IDs and position IDs for each cached token.
    Position IDs are preserved across drops to maintain correct positional encoding.
    
    Important: Use `seq_length` (not `cache_length`) to determine how many tokens
    have been processed from the conversation. `cache_length` may be smaller after
    drops, but `seq_length` reflects the logical position in the conversation.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.past_key_values = None
        self.round_ids = []      # Segment ID for each cached token
        self.position_ids = []   # Position ID for each cached token
        self._next_position = 0  # Monotonically increasing, not affected by drops
        self._seq_length = 0     # Total tokens processed (not affected by drops)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    @property
    def device(self):
        return self.model.device
    
    @property
    def cache_length(self):
        """Actual number of tokens in the KV cache (may decrease after drops)."""
        return len(self.round_ids)
    
    @property
    def seq_length(self):
        """Total tokens processed from conversation (not affected by drops).
        
        Use this to slice new tokens from the full conversation.
        """
        return self._seq_length
    
    @property
    def next_position(self):
        """Next position ID to use (monotonically increasing, not affected by drops)."""
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
        past_len = self._get_past_length()
        
        # Compute position IDs (continue from last position)
        start_pos = self.next_position
        position_ids = torch.arange(
            start_pos, start_pos + seq_len, 
            dtype=torch.long, device=self.device
        ).unsqueeze(0)
        
        # Attention mask covers all cached + new tokens
        attention_mask = torch.ones(
            (1, past_len + seq_len), 
            dtype=torch.long, 
            device=self.device
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=self.past_key_values,
                attention_mask=attention_mask,
                use_cache=True
            )
        
        # Ensure we always have a DynamicCache (some models return tuple)
        past = outputs.past_key_values
        if past is not None and not hasattr(past, "key_cache"):
            past = DynamicCache.from_legacy_cache(past)
        self.past_key_values = past
        self.round_ids.extend([id] * seq_len)
        self.position_ids.extend(range(start_pos, start_pos + seq_len))
        self._next_position = start_pos + seq_len  # Update for next append
        self._seq_length += seq_len  # Track total tokens processed
        
        return outputs

    def generate_step(
        self,
        input_ids,
        input_id: int,
        output_id: Optional[int] = None,
        max_new_tokens=50,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        drop_ids_after_prefill=None,
    ):
        """
        Generate response using manual decoding loop.
        
        Args:
            input_ids: Prompt tokens (will be appended to cache first).
            input_id: Segment ID for the input/prompt tokens.
            output_id: Segment ID for the generated tokens. If None, uses input_id.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold (1.0 disables).
            top_k: Top-k sampling threshold (0 disables).
            drop_ids_after_prefill: IDs to drop after prefilling but before generation.
                This enables "context transfer" where the prompt sees the context,
                but the generated response does not.
        Returns:
            Generated text (excluding prompt).
        """
        if output_id is None:
            output_id = input_id
            
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot be empty.")
        
        # Prefill prompt (with full context) using input_id
        outputs = self.append(input_ids, id=input_id)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Lazy drop: remove context after prefill but before generation
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
        Remove KV cache entries matching given segment IDs.
        Position IDs are preserved for remaining tokens.
        
        Args:
            remove_ids: Single ID or list of IDs to remove.
        """
        if not isinstance(remove_ids, list):
            remove_ids = [remove_ids]
            
        if self.past_key_values is None:
            return

        indices_to_keep = [i for i, rid in enumerate(self.round_ids) if rid not in remove_ids]
        
        if len(indices_to_keep) == len(self.round_ids):
            print(f"No rounds with IDs {remove_ids} found.")
            return
            
        if len(indices_to_keep) == 0:
            print("Dropping all rounds.")
            self.reset()
            return

        idx = torch.tensor(indices_to_keep, dtype=torch.long)
        
        # Modify DynamicCache in place
        # New transformers uses .layers with .keys/.values, old uses .key_cache/.value_cache
        if hasattr(self.past_key_values, "layers"):
            # New API: DynamicCache with CacheLayer objects
            for layer in self.past_key_values.layers:
                idx_dev = idx.to(layer.keys.device)
                layer.keys = layer.keys.index_select(2, idx_dev)
                layer.values = layer.values.index_select(2, idx_dev)
        else:
            # Old API: DynamicCache with key_cache/value_cache lists
            for i in range(len(self.past_key_values.key_cache)):
                k = self.past_key_values.key_cache[i]
                v = self.past_key_values.value_cache[i]
                idx_dev = idx.to(k.device)
                self.past_key_values.key_cache[i] = k.index_select(2, idx_dev)
                self.past_key_values.value_cache[i] = v.index_select(2, idx_dev)
        
        # Update seen_tokens count
        if hasattr(self.past_key_values, "_seen_tokens"):
            self.past_key_values._seen_tokens = len(indices_to_keep)
        
        # Update token_ids and position_ids (positions are preserved)
        self.round_ids = [self.round_ids[i] for i in indices_to_keep]
        self.position_ids = [self.position_ids[i] for i in indices_to_keep]
        
        print(f"Dropped IDs {remove_ids}. Cache: {len(self.round_ids)} tokens.")