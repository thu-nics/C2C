from typing import Tuple, List

from camel.memories import MemoryRecord, ContextRecord

from rosetta.workflow.track import InteractionTracker
from rosetta.workflow.camel_utils import context_records_to_memory_records

class ContextSelector:
    """Selects context records to share between agents.
    
    Args:
        filter_fn: (records, messages, tracker, llm_id) -> (filtered_records, indices)
        select_fn: (memory_records) -> (selected_records, indices_in_input)
    
    Example filter_fn and select_fn are provided as static methods.
    """
    
    def __init__(self, filter_fn=None, select_fn=None):
        self.filter_fn = filter_fn
        self.select_fn = select_fn

    def select(self, context_records: list[ContextRecord], messages: list[dict], 
               tracker: InteractionTracker, llm_id: int) -> Tuple[list[MemoryRecord], str, List[int]]:
        """Apply filter and selection, return (records, content, original_indices).
        
        Returns:
            records: The selected memory records.
            content: The content of the last record (before selection).
            original_indices: Indices of selected records in the original unfiltered context_records.
        """
        # Apply filter
        if self.filter_fn:
            context_records, filter_indices = self.filter_fn(context_records, messages, tracker, llm_id)
        else:
            filter_indices = list(range(len(context_records)))
        
        memory_records = context_records_to_memory_records(context_records)
        
        if len(memory_records) < 2:
            content = memory_records[-1].message.content if memory_records else ""
            # Map back to original indices
            original_indices = filter_indices if memory_records else []
            return memory_records, content, original_indices
        
        # Apply selection
        if self.select_fn:
            records, select_indices = self.select_fn(memory_records)
        else:
            records = memory_records
            select_indices = list(range(len(memory_records)))
        
        content = memory_records[-1].message.content
        # Map select indices back to original indices
        original_indices = [filter_indices[i] for i in select_indices]
        return records, content, original_indices

    # --- Example filter functions ---
    # All filter functions return (filtered_records, indices_in_original)
    @staticmethod
    def filter_none(records, messages, tracker, llm_id):
        """No filtering, keep all records."""
        indices = list(range(len(records)))
        return records, indices

    @staticmethod
    def filter_search_only(records, messages, tracker, llm_id):
        """Keep only UIDs unique to this agent (not in main agent)."""
        message_uids = tracker.messages_to_uids(messages)
        main_uids = set(tracker.get_uids(llm_id=0))
        search_uids = set(tracker.get_uids(llm_id=llm_id))
        search_only_uids = search_uids - main_uids
        indices = [i for i, uid in enumerate(message_uids) if uid in search_only_uids]
        return [records[i] for i in indices], indices

    @staticmethod
    def filter_shared(records, messages, tracker, llm_id):
        """Keep UIDs shared between LLM 0 and at least one other LLM (union of intersections)."""
        message_uids = tracker.messages_to_uids(messages)
        main_uids = set(tracker.get_uids(llm_id=0))
        other_llm_ids = [lid for lid in tracker.get_unique_llm_ids() if lid != 0]
        if not other_llm_ids:
            indices = list(range(len(records)))
            return records, indices
        # Union of (main ∩ each_search_agent)
        shared_uids = set()
        for lid in other_llm_ids:
            shared_uids |= main_uids & set(tracker.get_uids(llm_id=lid))
        indices = [i for i, uid in enumerate(message_uids) if uid in shared_uids]
        return [records[i] for i in indices], indices

    # --- Example select functions ---
    # All select functions return (selected_records, indices_in_input)
    @staticmethod
    def select_all(records):
        """Keep all records."""
        indices = list(range(len(records)))
        return records, indices

    @staticmethod
    def select_skip_system(records):
        """Skip system messages by filtering on role."""
        from camel.types import OpenAIBackendRole
        indices = [i for i, r in enumerate(records) if r.role_at_backend != OpenAIBackendRole.SYSTEM]
        return [records[i] for i in indices], indices

    @staticmethod
    def select_query_response_with_system(records):
        """Keep query and final response: [records[1], records[-1]]"""
        indices = [1, len(records) - 1]
        return [records[1], records[-1]], indices

    @staticmethod
    def select_query_response(records):
        """Keep query and final response: [records[1], records[-1]]"""
        indices = [0, len(records) - 1]
        return [records[0], records[-1]], indices

    @staticmethod
    def select_initial_with_system(records):
        """Keep first two records: [records[0], records[1]]"""
        indices = [1, 2]
        return [records[i] for i in indices], indices
    
    @staticmethod
    def select_initial(records):
        """Keep first two records: [records[1], records[2]]"""
        indices = [0, 1]
        return [records[i] for i in indices], indices

    @staticmethod
    def select_none(records):
        """Keep none: []"""
        return [], []
