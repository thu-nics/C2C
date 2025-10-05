"""
Token Aligner for handling different tokenizers between SLM and LLM models.

This module provides functionality to align tokens between two different tokenizers,
handling cases where the same text is tokenized differently.
"""

from typing import List, Tuple, Optional, Dict, Literal, Union
import torch
from transformers import PreTrainedTokenizerBase
from enum import Enum


class AlignmentStrategy(Enum):
    """Strategies for handling 1-to-many token alignments"""
    FIRST = "first"    # Always take the first LLM token
    LONGEST = "longest"  # Take the LLM token with the longest string


class TokenAligner:
    """
    Aligns tokens between SLM (Small Language Model) and LLM (Large Language Model) tokenizers.
    
    This class handles the case where the same text sequence is tokenized differently
    by different tokenizers, using the SLM tokenization as the base and finding
    corresponding LLM tokens for each SLM token.
    """
    
    def __init__(
        self,
        slm_tokenizer: PreTrainedTokenizerBase,
        llm_tokenizer: PreTrainedTokenizerBase,
        strategy: Union[AlignmentStrategy, str] = AlignmentStrategy.FIRST,
        verbose: bool = False
    ):
        """
        Initialize the TokenAligner.
        
        Args:
            slm_tokenizer: The tokenizer for the Small Language Model (base)
            llm_tokenizer: The tokenizer for the Large Language Model
            strategy: Strategy for handling 1-to-many token mappings
                     Either AlignmentStrategy enum or string ('first' or 'longest')
            verbose: Whether to print debug information during alignment
        """
        self.slm_tokenizer = slm_tokenizer
        self.llm_tokenizer = llm_tokenizer
        
        if self.slm_tokenizer.pad_token is None:
            self.slm_tokenizer.pad_token = self.slm_tokenizer.eos_token
            self.slm_tokenizer.pad_token_id = self.slm_tokenizer.eos_token_id
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        # Handle string strategy input
        if isinstance(strategy, str):
            strategy = AlignmentStrategy(strategy.lower())
        self.strategy = strategy
        self.verbose = verbose
        
        # Cache for token mappings to improve performance
        self._alignment_cache: Dict[Tuple[int, ...], List[int]] = {}
    
    def align_tokens(
        self,
        slm_token_ids: Union[List[int], torch.Tensor],
        return_mapping: bool = False
    ) -> Union[List[int], Tuple[List[int], List[Tuple[int, List[int]]]]]:
        """
        Align SLM tokens to LLM tokens.
        
        Args:
            slm_token_ids: Token IDs from the SLM tokenizer
            return_mapping: If True, also return the detailed mapping
        
        Returns:
            If return_mapping is False: List of aligned LLM token IDs
            If return_mapping is True: Tuple of (aligned_llm_token_ids, mapping_details)
                where mapping_details is a list of (slm_token_id, [candidate_llm_token_ids])
        """
        # Convert to list if tensor
        if isinstance(slm_token_ids, torch.Tensor):
            slm_token_ids = slm_token_ids.tolist()
        
        # Check cache
        cache_key = tuple(slm_token_ids)
        if cache_key in self._alignment_cache and not return_mapping:
            return self._alignment_cache[cache_key]
        
        aligned_llm_tokens = []
        mapping_details = []
        
        for slm_token_id in slm_token_ids:
            # Decode SLM token to string (without special token processing)
            slm_token_str = self.slm_tokenizer.decode(
                [slm_token_id], 
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            
            # Handle special tokens
            if slm_token_id in self.slm_tokenizer.all_special_ids:
                # Try to find corresponding special token in LLM tokenizer
                llm_token_id = self._map_special_token(slm_token_id, slm_token_str)
                aligned_llm_tokens.append(llm_token_id)
                mapping_details.append((slm_token_id, [llm_token_id]))
                continue
            
            # Tokenize the string with LLM tokenizer
            llm_token_ids = self.llm_tokenizer.encode(
                slm_token_str,
                add_special_tokens=False,
                return_tensors=None
            )
            
            if len(llm_token_ids) == 0:
                # Handle empty tokenization (shouldn't normally happen)
                if self.verbose:
                    print(f"Warning: SLM token {slm_token_id} ('{slm_token_str}') "
                          f"resulted in empty LLM tokenization")
                # Use unknown token as fallback
                llm_token_id = self.llm_tokenizer.unk_token_id or 0
                aligned_llm_tokens.append(llm_token_id)
                mapping_details.append((slm_token_id, [llm_token_id]))
                
            elif len(llm_token_ids) == 1:
                # Perfect 1-to-1 mapping
                aligned_llm_tokens.append(llm_token_ids[0])
                mapping_details.append((slm_token_id, llm_token_ids))
                
            else:
                # 1-to-many mapping, apply strategy
                selected_token = self._apply_strategy(llm_token_ids, slm_token_str)
                aligned_llm_tokens.append(selected_token)
                mapping_details.append((slm_token_id, llm_token_ids))
                
                if self.verbose:
                    selected_str = self.llm_tokenizer.decode(
                        [selected_token],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )
                    print(f"SLM token {slm_token_id} ('{slm_token_str}') -> "
                          f"LLM tokens {llm_token_ids}, selected {selected_token} ('{selected_str}')")
        
        # Cache the result
        self._alignment_cache[cache_key] = aligned_llm_tokens
        
        if return_mapping:
            return aligned_llm_tokens, mapping_details
        return aligned_llm_tokens
    
    def _map_special_token(self, slm_token_id: int, slm_token_str: str) -> int:
        """
        Map special tokens between tokenizers.
        
        Args:
            slm_token_id: The SLM special token ID
            slm_token_str: The string representation of the special token
        
        Returns:
            The corresponding LLM token ID
        """
        # Common special token mappings
        special_token_map = {
            self.slm_tokenizer.pad_token_id: self.llm_tokenizer.pad_token_id,
            self.slm_tokenizer.eos_token_id: self.llm_tokenizer.eos_token_id,
            self.slm_tokenizer.bos_token_id: self.llm_tokenizer.bos_token_id,
            self.slm_tokenizer.unk_token_id: self.llm_tokenizer.unk_token_id,
        }
        
        # Direct mapping if available
        if slm_token_id in special_token_map and special_token_map[slm_token_id] is not None:
            return special_token_map[slm_token_id]
        
        # Try to find by string representation
        try:
            llm_token_id = self.llm_tokenizer.convert_tokens_to_ids(slm_token_str)
            if llm_token_id != self.llm_tokenizer.unk_token_id:
                return llm_token_id
        except:
            pass
        
        # Fallback to unknown token
        return self.llm_tokenizer.unk_token_id or 0
    
    def _apply_strategy(self, llm_token_ids: List[int], original_str: str) -> int:
        """
        Apply the selected strategy to choose one LLM token from multiple candidates.
        
        Args:
            llm_token_ids: List of candidate LLM token IDs
            original_str: The original string from SLM token
        
        Returns:
            The selected LLM token ID
        """
        if self.strategy == AlignmentStrategy.FIRST:
            return llm_token_ids[0]
        
        elif self.strategy == AlignmentStrategy.LONGEST:
            # Find the token with the longest string representation
            longest_token = llm_token_ids[0]
            longest_length = 0
            
            for token_id in llm_token_ids:
                token_str = self.llm_tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                if len(token_str) > longest_length:
                    longest_length = len(token_str)
                    longest_token = token_id
            
            return longest_token
        
        else:
            # Default to first token if unknown strategy
            return llm_token_ids[0]
    
    def align_sequence(
        self,
        text: str,
        return_details: bool = False
    ) -> Union[Tuple[List[int], List[int]], Dict[str, any]]:
        """
        Tokenize text with both tokenizers and return aligned sequences.
        
        Args:
            text: The input text to tokenize and align
            return_details: If True, return detailed alignment information
        
        Returns:
            If return_details is False: Tuple of (slm_token_ids, aligned_llm_token_ids)
            If return_details is True: Dictionary with detailed alignment information
        """
        # Tokenize with SLM
        slm_tokens = self.slm_tokenizer.encode(
            text,
            add_special_tokens=True,
            return_tensors=None
        )
        
        # Get aligned LLM tokens
        if return_details:
            aligned_llm_tokens, mapping = self.align_tokens(slm_tokens, return_mapping=True)
            
            # Decode tokens for inspection
            slm_decoded = [
                self.slm_tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for tid in slm_tokens
            ]
            llm_decoded = [
                self.llm_tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for tid in aligned_llm_tokens
            ]
            
            # Original LLM tokenization for comparison
            original_llm_tokens = self.llm_tokenizer.encode(
                text,
                add_special_tokens=True,
                return_tensors=None
            )
            
            # One-to-one mapping statistics
            num_tokens = len(slm_tokens)
            one_to_one_count = sum(1 for _slm_id, candidates in mapping if len(candidates) == 1)
            one_to_one_rate = (one_to_one_count / num_tokens) if num_tokens > 0 else 0.0
            
            return {
                'text': text,
                'slm_token_ids': slm_tokens,
                'slm_decoded': slm_decoded,
                'aligned_llm_token_ids': aligned_llm_tokens,
                'aligned_llm_decoded': llm_decoded,
                'original_llm_token_ids': original_llm_tokens,
                'mapping': mapping,
                'strategy': self.strategy.value,
                'num_tokens': num_tokens,
                'one_to_one_count': one_to_one_count,
                'one_to_one_rate': one_to_one_rate
            }
        else:
            aligned_llm_tokens = self.align_tokens(slm_tokens)
            return slm_tokens, aligned_llm_tokens
    
    def visualize_alignment(self, text: str):
        """
        Print a visual representation of the token alignment.
        
        Args:
            text: The text to analyze
        """
        details = self.align_sequence(text, return_details=True)
        
        print("=" * 80)
        print(f"Text: {text}")
        print(f"Strategy: {details['strategy']}")
        print("=" * 80)
        print(f"SLM tokens ({len(details['slm_token_ids'])}): {details['slm_token_ids']}")
        print(f"Aligned LLM tokens ({len(details['aligned_llm_token_ids'])}): {details['aligned_llm_token_ids']}")
        print(f"Original LLM tokens ({len(details['original_llm_token_ids'])}): {details['original_llm_token_ids']}")
        print("-" * 80)
        print("Token-by-token alignment:")
        
        for i, (slm_id, llm_id) in enumerate(zip(details['slm_token_ids'], details['aligned_llm_token_ids'])):
            slm_str = details['slm_decoded'][i]
            llm_str = details['aligned_llm_decoded'][i]
            mapping_info = details['mapping'][i]
            
            if len(mapping_info[1]) > 1:
                candidates_str = ', '.join([
                    f"{tid}:'{self.llm_tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)}'"
                    for tid in mapping_info[1]
                ])
                print(f"  [{i:3d}] SLM {slm_id:6d} ('{slm_str}') -> "
                      f"LLM {llm_id:6d} ('{llm_str}') "
                      f"[candidates: {candidates_str}]")
            else:
                print(f"  [{i:3d}] SLM {slm_id:6d} ('{slm_str}') -> "
                      f"LLM {llm_id:6d} ('{llm_str}')")
        print("=" * 80)
    
    def clear_cache(self):
        """Clear the alignment cache."""
        self._alignment_cache.clear()

    # ========================
    # Chat messages alignment
    # ========================
    def _apply_chat_template_to_ids(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool,
        enable_thinking: bool,
        remove_last_surfix: bool
    ) -> Tuple[str, List[int], Optional[List[Tuple[int, int]]]]:
        """
        Apply chat template (no tokenization) then tokenize to ids with optional offsets.
        If remove_last_surfix is True, remove the last suffix from the LLM text
        Returns (templated_text, input_ids, offsets) where offsets may be None.
        """
        if remove_last_surfix:
            assert messages[-1]["role"] == "assistant", "Last message must be an assistant message"
            templated_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            templated_text += messages[-1]["content"]
        else:
            templated_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking
            )
        encoded = tokenizer(
            templated_text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        input_ids: List[int] = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        return templated_text, input_ids, offsets

    @staticmethod
    def _first_non_empty_content(messages: List[Dict[str, str]]) -> Optional[str]:
        for m in messages:
            content = m.get("content")
            if isinstance(content, str) and len(content.strip()) > 0:
                return content
        return None

    def _find_boundary_token_index(
        self,
        tokenizer: PreTrainedTokenizerBase,
        templated_text: str,
        offsets: Optional[List[Tuple[int, int]]],
        content_text: Optional[str]
    ) -> int:
        """
        Find token index where the first non-empty message content starts.
        Falls back to 0 if not found.
        """
        if not content_text:
            return 0
        char_idx = templated_text.find(content_text)
        if char_idx < 0:
            # Try a shorter probe to improve chances
            probe = content_text[: min(32, len(content_text))]
            if len(probe) > 0:
                char_idx = templated_text.find(probe)
        if char_idx < 0:
            return 0

        if offsets:
            for idx, (start, _end) in enumerate(offsets):
                if start >= char_idx:
                    return idx
            return len(offsets)

        # Fallback without offsets: tokenize prefix and count tokens
        prefix = templated_text[:char_idx]
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        return len(prefix_ids)

    @staticmethod
    def _compute_content_spans(templated_text: str, messages: List[Dict[str, str]]) -> List[Tuple[int, int]]:
        """
        Compute character spans in templated_text that correspond to message contents.
        Searches sequentially to reduce ambiguity when contents repeat.
        Enhanced matching: ensures the found content is followed by '<' (special token start)
        to avoid matching content inside special tokens like <begin_of_text>.
        """
        spans: List[Tuple[int, int]] = []
        search_from = 0
        for m in messages:
            content = m.get("content")
            if not isinstance(content, str) or len(content) == 0:
                continue
            
            # Find all possible matches starting from search_from
            idx = search_from
            found_valid_match = False
            
            while idx < len(templated_text):
                idx = templated_text.find(content, idx)
                if idx < 0:
                    break
                
                # Check if this match is valid (followed by '<' indicating a special token)
                end_pos = idx + len(content)
                if end_pos < len(templated_text) and templated_text[end_pos] == '<':
                    # Valid match: content is followed by a special token
                    spans.append((idx, end_pos))
                    search_from = end_pos
                    found_valid_match = True
                    break
                else:
                    # Check if this is the end of the text (also valid for last message)
                    if end_pos == len(templated_text):
                        spans.append((idx, end_pos))
                        search_from = end_pos
                        found_valid_match = True
                        break
                
                # Invalid match, try next occurrence
                idx += 1
            
            # Fallback: if no valid match found with '<' requirement, use the old method
            # but only as a last resort and with additional validation
            if not found_valid_match:
                idx = templated_text.find(content, search_from)
                if idx < 0:
                    # Try searching from start as last resort
                    idx = templated_text.find(content)
                
                if idx >= 0:
                    end_pos = idx + len(content)
                    # Additional check: avoid matching inside obvious special tokens
                    # Check if we're inside a special token (preceded by '<' and not followed by '>')
                    start_context = templated_text[max(0, idx-10):idx]
                    end_context = templated_text[end_pos:min(len(templated_text), end_pos+10)]
                    
                    # Skip if we're clearly inside a special token
                    if ('<' in start_context and '>' not in start_context and 
                        'begin_of_text' in templated_text[max(0, idx-20):idx+20]):
                        # This looks like we're matching inside <begin_of_text> or similar
                        continue
                    
                    spans.append((idx, end_pos))
                    search_from = end_pos
        
        return spans

    @staticmethod
    def _build_token_mask_from_spans(
        offsets: Optional[List[Tuple[int, int]]],
        num_tokens: int,
        spans: List[Tuple[int, int]]
    ) -> List[bool]:
        """
        Build a boolean mask for tokens whose offset range overlaps any span.
        If offsets are missing, default to all False.
        """
        if not offsets or len(offsets) != num_tokens:
            return [False] * num_tokens
        mask: List[bool] = []
        for (start, end) in offsets:
            if end <= start:
                mask.append(False)
                continue
            is_msg = False
            for s, e in spans:
                # overlap check
                if start < e and end > s:
                    is_msg = True
                    break
            mask.append(is_msg)
        return mask

    @staticmethod
    def _spans_to_token_ranges(
        offsets: List[Tuple[int, int]],
        spans: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Convert character spans to token index ranges using offsets.
        start token = first token with end > span_start
        end token = first token with start >= span_end
        """
        ranges: List[Tuple[int, int]] = []
        n = len(offsets)
        for s, e in spans:
            # find start index
            start_idx = 0
            while start_idx < n and offsets[start_idx][1] <= s:
                start_idx += 1
            # find end index
            end_idx = start_idx
            while end_idx < n and offsets[end_idx][0] < e:
                end_idx += 1
            ranges.append((start_idx, end_idx))
        return ranges

    def align_chat_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        return_details: bool = False,
        remove_last_surfix: bool = False
    ) -> Dict[str, any]:
        """
        Align chat-templated sequences by sections (template/message/template...):
        - Preserve all template tokens (pad the shorter template section)
        - For each message section, map SLM tokens to LLM tokens 1:1 via strategy
        - If remove_last_surfix is True, remove the last suffix from the LLM text
        Returns essentials: slm_ids_padded, llm_ids_padded, message_mask (shared),
        slm_padding_mask, llm_padding_mask (True where token is padding inserted).
        When return_details=True, also returns 'sections' with aligned ranges.
        """
        assert not (add_generation_prompt and remove_last_surfix), "add_generation_prompt and remove_last_surfix cannot be True at the same time"

        # Build templated sequences with offsets
        slm_text, slm_ids, slm_offsets = self._apply_chat_template_to_ids(
            self.slm_tokenizer, messages, add_generation_prompt, enable_thinking, remove_last_surfix
        )
        llm_text, llm_ids, llm_offsets = self._apply_chat_template_to_ids(
            self.llm_tokenizer, messages, add_generation_prompt, enable_thinking, remove_last_surfix
        )

        # Required pad tokens
        assert self.slm_tokenizer.pad_token_id is not None, "SLM pad_token_id required"
        assert self.llm_tokenizer.pad_token_id is not None, "LLM pad_token_id required"
        slm_pad_id = self.slm_tokenizer.pad_token_id
        llm_pad_id = self.llm_tokenizer.pad_token_id

        # Content spans (char) and token ranges
        content_spans_slm = self._compute_content_spans(slm_text, messages)
        content_spans_llm = self._compute_content_spans(llm_text, messages)
        assert slm_offsets is not None and llm_offsets is not None, "offset_mapping required"
        slm_msg_ranges = self._spans_to_token_ranges(slm_offsets, content_spans_slm)
        llm_msg_ranges = self._spans_to_token_ranges(llm_offsets, content_spans_llm)
        # Build section ranges (template/message alternating)
        def build_sections(total_len: int, msg_ranges: List[Tuple[int,int]]):
            sections: List[Tuple[str,int,int]] = []
            prev = 0
            for (s, e) in msg_ranges:
                if prev < s:
                    sections.append(("template", prev, s))
                sections.append(("message", s, e))
                prev = e
            if prev < total_len:
                sections.append(("template", prev, total_len))
            return sections
        slm_sections = build_sections(len(slm_ids), slm_msg_ranges)
        llm_sections = build_sections(len(llm_ids), llm_msg_ranges)
        assert len(slm_sections) == len(llm_sections), "Section count mismatch"

        slm_out: List[int] = []
        llm_out: List[int] = []
        mask_out: List[bool] = []
        slm_pad_mask_out: List[bool] = []
        llm_pad_mask_out: List[bool] = []
        detailed_sections: List[Dict[str, Union[str, Tuple[int,int]]]] = []

        for (stype_s, s_s, e_s), (stype_l, s_l, e_l) in zip(slm_sections, llm_sections):
            assert stype_s == stype_l, "Section type mismatch"
            slm_start_out = len(slm_out)
            llm_start_out = len(llm_out)
            if stype_s == "template":
                slm_seg_len = e_s - s_s
                llm_seg_len = e_l - s_l
                target_len = slm_seg_len if slm_seg_len >= llm_seg_len else llm_seg_len
                slm_pad_needed = target_len - slm_seg_len
                llm_pad_needed = target_len - llm_seg_len
                slm_seg = slm_ids[s_s:e_s] + [slm_pad_id] * slm_pad_needed
                llm_seg = llm_ids[s_l:e_l] + [llm_pad_id] * llm_pad_needed
                slm_out.extend(slm_seg)
                llm_out.extend(llm_seg)
                mask_out.extend([False] * target_len)
                slm_pad_mask_out.extend([False] * slm_seg_len + [True] * slm_pad_needed)
                llm_pad_mask_out.extend([False] * llm_seg_len + [True] * llm_pad_needed)
            else:  # message
                slm_msg = slm_ids[s_s:e_s]
                llm_msg = self.align_tokens(slm_msg)
                assert len(llm_msg) == len(slm_msg)
                slm_out.extend(slm_msg)
                llm_out.extend(llm_msg)
                mask_out.extend([True] * len(slm_msg))
                # no padding in message sections
                slm_pad_mask_out.extend([False] * len(slm_msg))
                llm_pad_mask_out.extend([False] * len(slm_msg))
            slm_end_out = len(slm_out)
            llm_end_out = len(llm_out)
            detailed_sections.append({
                'type': stype_s,
                'slm_range': (slm_start_out, slm_end_out),
                'llm_range': (llm_start_out, llm_end_out)
            })

        result_min = {
            'slm_ids_padded': slm_out,
            'llm_ids_padded': llm_out,
            'message_mask': mask_out,
            'slm_padding_mask': slm_pad_mask_out,
            'llm_padding_mask': llm_pad_mask_out
        }
        if return_details:
            result_min['sections'] = detailed_sections
            result_min['slm_text'] = slm_text
            result_min['llm_text'] = llm_text
        return result_min
