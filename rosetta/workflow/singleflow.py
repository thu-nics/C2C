from typing import Optional, Tuple

from camel.agents import ChatAgent
from camel.types import OpenAIBackendRole

from rosetta.context.track import InteractionTracker, record_interaction
from rosetta.context.selector import ContextSelector
from rosetta.workflow.camel_utils import context_records_to_memory_records
from rosetta.workflow.prompt import SIMPLE_RESEARCH_PROMPT


def _compute_drop_messages(
    forward_records: list,
    contextual_selector: ContextSelector,
    offset: int,
) -> dict:
    if not forward_records or not contextual_selector or not contextual_selector.select_fn:
        return {}

    _, keep_indices = contextual_selector.select_fn(forward_records)
    keep_set = set(keep_indices)
    drop_indices_in_forward = [i for i in range(len(forward_records)) if i not in keep_set]
    if not drop_indices_in_forward:
        return {}

    drop_indices_in_target = [offset + i for i in drop_indices_in_forward]
    drop_at = offset + len(forward_records)
    return {drop_at: drop_indices_in_target}


def single_research(
    question: str,
    main_agent: ChatAgent,
    tracker: Optional[InteractionTracker] = None,
    context_plan: Optional[dict] = None,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Single-step research with optional contextual attention dropping."""
    if tracker is not None:
        tools = list(main_agent.tool_dict.values())
        if tools:
            tracker.register_tools(llm_id=0, tools=tools)

    main_contextual = context_plan.get("main_contextual") if context_plan else None
    if main_contextual and main_contextual.select_fn:
        context_records = main_agent.memory.retrieve()
        memory_records = context_records_to_memory_records(context_records)
        if memory_records and memory_records[0].role_at_backend == OpenAIBackendRole.SYSTEM:
            memory_records = memory_records[1:]
        drop_msgs = _compute_drop_messages(memory_records, main_contextual, offset=1)
        if drop_msgs:
            extra_body = main_agent.model_backend.model_config_dict.setdefault("extra_body", {})
            extra_body["drop_messages"] = drop_msgs

    response = main_agent.step(SIMPLE_RESEARCH_PROMPT.format(question=question))
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    return response.msg.content, tracker
