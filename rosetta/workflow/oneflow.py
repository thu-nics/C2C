import re
from typing import List, Tuple, Optional, Any
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.memories import ContextRecord, MemoryRecord
from camel.types import OpenAIBackendRole

from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.selector import ContextSelector
from rosetta.workflow.prompt import SEARCH_TASK_DECOMPOSE_PROMPT, TASK_REVISE_PROMPT, FORCE_ANSWER_PROMPT, SEARCH_AGENT_PROMPT
from rosetta.workflow.display import StatusLogger

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
    search_tools: List[FunctionTool] = None,
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
        search_tools: List of tools for search agent. Defaults to [Google search].
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
    if search_tools is None:
        search_tools = [FunctionTool(SearchToolkit().search_google)]
    
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
    main_agent.model_backend.models[0].role = "main"
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
        with logger.round(round_idx + 1, tasks) as update_response:
            search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=search_tools, summarize_threshold=None, step_timeout=3000)
            search_agent.model_backend.models[0].role = "search"
            # Register tools for this search agent
            if tracker is not None:
                tracker.register_tools(llm_id=round_idx + 1, tools=search_tools)
            
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
                search_model.model_config_dict.setdefault("extra_body", {}).setdefault("search", {})["drop_messages"] = drop_msgs
            
            search_resp = search_agent.step(tasks[0])

            # Extract tool names from chat history
            tools_used = []
            for msg in search_agent.chat_history:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_name = tc.get("function", {}).get("name")
                        if tool_name and tool_name not in tools_used:
                            tools_used.append(tool_name)

            update_response(getattr(search_resp.msg, "content", None), tools_used=tools_used if tools_used else None)
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
                main_agent.model_backend.model_config_dict.setdefault("extra_body", {}).setdefault("main", {})
                existing_drops = main_agent.model_backend.model_config_dict["extra_body"]["main"].get("drop_messages", {})
                existing_drops.update(drop_msgs)
                main_agent.model_backend.model_config_dict["extra_body"]["main"]["drop_messages"] = existing_drops
            
            main_agent.model_backend.models[0].role = "main"
            response = main_agent.step(TASK_REVISE_PROMPT.format(question=question))
            record_interaction(tracker, main_agent.chat_history, llm_id=0)

    # Force answer if no answer yet
    info = "\n\n".join(collected_info) if collected_info else "No information found."
    main_agent.model_backend.models[0].role = "main"
    response = main_agent.step(FORCE_ANSWER_PROMPT.format(question=question, info=info))
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    answer = _parse_answer(response.msg.content)
    return answer if answer else response.msg.content, tracker

