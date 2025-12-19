import re
from contextlib import contextmanager
from typing import Tuple, Optional, Any
from rich.console import Console

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.memories import ContextRecord, MemoryRecord
from camel.types import OpenAIBackendRole

from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.selector import ContextSelector
from rosetta.workflow.prompt import SEARCH_TASK_DECOMPOSE_PROMPT, TASK_REVISE_PROMPT, FORCE_ANSWER_PROMPT, SEARCH_AGENT_PROMPT
from rosetta.workflow.camel_utils import context_records_to_memory_records, MemoryRecord_flip_role

class StatusLogger:
    """Handles console status display with spinner and history."""
    
    def __init__(self, enabled: bool = True):
        self.console = Console() if enabled else None

    def _format_tasks(self, round_idx: int, tasks: list) -> str:
        """Format all tasks with current highlighted. Numbering starts from round_idx."""
        lines = [f"Round {round_idx} | Tasks: {len(tasks)}"]
        for i, task in enumerate(tasks):
            prefix = "  → " if i == 0 else "    "
            lines.append(f"{prefix}[{round_idx + i}] {task}")
        return "\n".join(lines)

    @contextmanager
    def round(self, round_idx: int, tasks: list):
        """Context manager: show spinner with all tasks, print ✓ with current task after."""
        status_msg = self._format_tasks(round_idx, tasks)
        done_msg = f"Round {round_idx} | Tasks: {len(tasks)} | {tasks[0]}"
        if self.console:
            with self.console.status(status_msg):
                yield
            self.console.print(f"[green]✓[/green] {done_msg}", highlight=False)
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

def _compute_drop_messages(forward_records: list, contextual_selector: ContextSelector, offset: int) -> dict:
    """Compute drop_messages based on contextual selector.
    
    Args:
        forward_records: Records forwarded to the target agent.
        contextual_selector: Selector that determines what to KEEP (drop = complement).
        offset: Index offset in target context (e.g., 1 for system prompt).
    
    Returns:
        drop_messages dict: {last_msg_index: [drop_indices]}
    """
    if not forward_records or not contextual_selector or not contextual_selector.select_fn:
        return {}
    
    # Get indices to KEEP from forwarded records
    _, keep_indices = contextual_selector.select_fn(forward_records)
    keep_set = set(keep_indices)
    
    # Compute indices to DROP (complement of keep)
    drop_indices_in_forward = [i for i in range(len(forward_records)) if i not in keep_set]
    
    if not drop_indices_in_forward:
        return {}
    
    # Map to target context indices (offset for system prompt, etc.)
    drop_indices_in_target = [offset + i for i in drop_indices_in_forward]
    
    # drop_at = last message index = offset + len(forward) (task message index)
    drop_at = offset + len(forward_records)
    
    return {drop_at: drop_indices_in_target}


def do_research(
    question: str,
    main_agent: ChatAgent,
    search_model: BaseModelBackend,
    tracker: InteractionTracker = None,
    search_tool: FunctionTool = None,
    context_plan: dict = None,
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
        context_plan: Dict with selectors:
            - 'search_to_main_selector': What memory from search goes to main.
            - 'main_to_search_selector': What memory from main goes to search.
            - 'search_contextual': What search model attends to (select_fn returns KEEP indices).
            - 'main_contextual': What main model attends to (select_fn returns KEEP indices).
        max_rounds: Maximum iterations. default is 10.
        show_status: Show spinner status. Disable when using with tqdm.

    Returns:
        The response message and the tracker.
    """
    logger = StatusLogger(enabled=show_status)
    if search_tool is None:
        search_tool = FunctionTool(SearchToolkit().search_google)
    
    # Setup context selectors
    if context_plan is None:
        context_plan = {}
    search_to_main_selector: ContextSelector = context_plan.get('search_to_main_selector', ContextSelector(
        filter_fn=ContextSelector.filter_search_only,
        select_fn=ContextSelector.select_query_response
    ))
    main_to_search_selector: ContextSelector = context_plan.get('main_to_search_selector', ContextSelector(
        filter_fn=None,
        select_fn=ContextSelector.select_skip_system
    ))
    
    # Contextual selectors (for attention dropping)
    search_contextual: ContextSelector = context_plan.get('search_contextual')
    main_contextual: ContextSelector = context_plan.get('main_contextual')

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
        with logger.round(round_idx + 1, tasks):
            search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[search_tool])
            
            # Register tools for this search agent
            if tracker is not None:
                tracker.register_tools(llm_id=round_idx + 1, tools=[search_tool])
            
            # Forward context to search agent
            forward_context, _, _ = main_to_search_selector.select(
                main_agent.memory.retrieve(), main_agent.chat_history, tracker, round_idx + 1
            )
            # Register UUIDs for alignment before writing to search agent
            if tracker is not None:
                tracker.register_shared_records(forward_context)
            search_agent.memory.write_records(forward_context)
            
            # Compute drop_messages for search model (drop from main context)
            # Search context: [0:sys, 1:fwd1, ..., N:fwdN, N+1:task]
            # Offset = 1 (system prompt at index 0)
            if search_contextual:
                drop_msgs = _compute_drop_messages(forward_context, search_contextual, offset=1)
                if "extra_body" not in search_model.model_config_dict:
                    search_model.model_config_dict["extra_body"] = {}
                search_model.model_config_dict["extra_body"]["drop_messages"] = drop_msgs
            
            search_agent.step(tasks[0])
            record_interaction(tracker, search_agent.chat_history, llm_id=round_idx + 1)

            # Feedback to main agent
            feedback_context, feedback_content, _ = search_to_main_selector.select(
                search_agent.memory.retrieve(), search_agent.chat_history, tracker, round_idx + 1
            )
            collected_info.append(feedback_content)
            
            # Compute offset for main context (existing messages before feedback)
            main_offset = len(main_agent.memory.retrieve())
            # Register UUIDs for alignment before writing to main agent
            if tracker is not None:
                tracker.register_shared_records(feedback_context)
            main_agent.memory.write_records(feedback_context)
            
            # Compute drop_messages for main model (drop from search context)
            # Main context: [...existing..., fwd1, ..., fwdM, revision_prompt]
            # Offset = main_offset (number of existing messages)
            # NOTE: drop_messages accumulates across rounds for main agent
            if main_contextual:
                drop_msgs = _compute_drop_messages(feedback_context, main_contextual, offset=main_offset)
                if "extra_body" not in main_agent.model_backend.model_config_dict:
                    main_agent.model_backend.model_config_dict["extra_body"] = {}
                # Accumulate drop_messages across rounds
                existing_drops = main_agent.model_backend.model_config_dict["extra_body"].get("drop_messages", {})
                existing_drops.update(drop_msgs)
                main_agent.model_backend.model_config_dict["extra_body"]["drop_messages"] = existing_drops
            
            response = main_agent.step(TASK_REVISE_PROMPT.format(question=question))
            record_interaction(tracker, main_agent.chat_history, llm_id=0)

    # Force answer if no answer yet
    info = "\n\n".join(collected_info) if collected_info else "No information found."
    response = main_agent.step(FORCE_ANSWER_PROMPT.format(question=question, info=info))
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    answer = _parse_answer(response.msg.content)
    return answer if answer else response.msg.content, tracker

