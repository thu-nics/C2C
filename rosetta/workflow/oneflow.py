import re
from contextlib import contextmanager
from typing import Tuple, Optional, Any
from rich.console import Console

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.memories import ContextRecord, MemoryRecord

from rosetta.context.track import InteractionTracker, record_interaction
from rosetta.workflow.prompt import SEARCH_TASK_DECOMPOSE_PROMPT, TASK_REVISE_PROMPT, FORCE_ANSWER_PROMPT, SEARCH_AGENT_PROMPT
from rosetta.workflow.camel_utils import context_records_to_memory_records, MemoryRecord_flip_role


class StatusLogger:
    """Handles console status display with spinner and history."""
    
    def __init__(self, enabled: bool = True):
        self.console = Console() if enabled else None

    @contextmanager
    def task(self, msg: str):
        """Context manager: show spinner during execution, print ✓ after."""
        if self.console:
            with self.console.status(msg):
                yield
            self.console.print(f"[green]✓[/green] {msg}", highlight=False)
        else:
            yield

def _parse_tasks(text: str) -> list:
    """Extract tasks from <tasks><task>...</task></tasks> format."""
    matches = re.findall(r'<task>(.*?)</task>', text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]

def _parse_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> format."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None

class ContextSelector:
    """Selects context records to share between agents.
    
    Args:
        filter_fn: (records, messages, tracker, llm_id) -> filtered_records
        select_fn: (memory_records) -> selected_records
    
    Example filter_fn and select_fn are provided as static methods.
    """
    
    def __init__(self, filter_fn=None, select_fn=None):
        self.filter_fn = filter_fn
        self.select_fn = select_fn

    def select(self, context_records: list[ContextRecord], messages: list[dict], 
               tracker: InteractionTracker, llm_id: int) -> Tuple[list[MemoryRecord], str]:
        """Apply filter and selection, return (records, content)."""
        if self.filter_fn:
            context_records = self.filter_fn(context_records, messages, tracker, llm_id)
        
        memory_records = context_records_to_memory_records(context_records)
        
        if len(memory_records) < 2:
            content = memory_records[-1].message.content if memory_records else ""
            return memory_records, content
        
        records = self.select_fn(memory_records) if self.select_fn else memory_records
        content = memory_records[-1].message.content
        return records, content

    # --- Example filter functions ---
    @staticmethod
    def filter_none(records, messages, tracker, llm_id):
        """No filtering, keep all records."""
        return records

    @staticmethod
    def filter_search_only(records, messages, tracker, llm_id):
        """Keep only UIDs unique to this agent (not in main agent)."""
        message_uids = tracker.messages_to_uids(messages)
        main_uids = set(tracker.get_uids(llm_id=0))
        search_uids = set(tracker.get_uids(llm_id=llm_id))
        search_only_uids = search_uids - main_uids
        indices = [i for i, uid in enumerate(message_uids) if uid in search_only_uids]
        return [records[i] for i in indices]

    # --- Example select functions ---
    @staticmethod
    def select_all(records):
        """Keep all records."""
        return records

    @staticmethod
    def select_skip_system(records):
        """Skip system message: records[1:]"""
        return records[1:]

    @staticmethod
    def select_query_response(records):
        """Keep query and final response: [records[1], records[-1]]"""
        return [records[1], records[-1]]

    @staticmethod
    def select_none(records):
        """Keep none: []"""
        return []

# Pre-configured selectors
search_to_main_selector = ContextSelector(
    filter_fn=ContextSelector.filter_search_only,
    select_fn=ContextSelector.select_query_response
)
# main_to_search_selector = ContextSelector(
#     filter_fn=None,
#     select_fn=ContextSelector.select_skip_system
# )
main_to_search_selector = ContextSelector(
    filter_fn=None,
    select_fn=ContextSelector.select_none
)

def do_research(
    question: str,
    main_agent: ChatAgent,
    search_model: BaseModelBackend,
    tracker: InteractionTracker = None,
    search_tool: FunctionTool = None,
    max_rounds: int = 10,
    show_status: bool = True,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Iterative subagent research with task revision loop.
    
    Args:
        question: The question to answer.
        main_agent: The main agent.
        search_model: The search model.
        tracker: The tracker.
        search_tool: The search tool. default is Google search.
        max_rounds: Maximum iterations. default is 10.
        show_status: Show spinner status. Disable when using with tqdm.

    Returns:
        The response message and the tracker.
    """
    logger = StatusLogger(enabled=show_status)
    if search_tool is None:
        search_tool = FunctionTool(SearchToolkit().search_google)

    # Initial decomposition
    response = main_agent.step(SEARCH_TASK_DECOMPOSE_PROMPT.format(content=question))
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    
    collected_info = []
    for round_idx in range(max_rounds):
        # Check for answer
        answer = _parse_answer(response.msg.content)
        if answer:
            return answer, tracker
        
        # Check for tasks
        tasks = _parse_tasks(response.msg.content)
        if not tasks:
            break
        
        # Execute round with status display
        with logger.task(f"Round {round_idx+1} | Tasks: {len(tasks)} | {tasks[0]}"):
            search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[search_tool])
            
            # Forward context to search agent
            forward_context, _ = main_to_search_selector.select(
                main_agent.memory.retrieve(), main_agent.chat_history, tracker, round_idx + 1
            )
            search_agent.memory.write_records(forward_context)
            search_agent.step(tasks[0])
            record_interaction(tracker, search_agent.chat_history, llm_id=round_idx + 1)

            # Feedback to main agent
            feedback_context, feedback_content = search_to_main_selector.select(
                search_agent.memory.retrieve(), search_agent.chat_history, tracker, round_idx + 1
            )
            collected_info.append(feedback_content)
            main_agent.memory.write_records(feedback_context)
            response = main_agent.step(TASK_REVISE_PROMPT.format(question=question))
            record_interaction(tracker, main_agent.chat_history, llm_id=0)

    # Force answer if no answer yet
    info = "\n\n".join(collected_info) if collected_info else "No information found."
    response = main_agent.step(FORCE_ANSWER_PROMPT.format(question=question, info=info))
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    answer = _parse_answer(response.msg.content)
    return answer if answer else response.msg.content, tracker

