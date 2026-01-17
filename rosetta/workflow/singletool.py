import json
from typing import List, Optional, Tuple

from camel.toolkits import FunctionTool
from camel.models import BaseModelBackend

from rosetta.workflow.display import ConvLogger
from rosetta.workflow.basic_utils import msg_system, msg_user, msg_assistant, msg_tool, execute_tool, _clean_for_api
from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.contextManage import ContextManager

def run_with_tools(
    question: str,
    model: BaseModelBackend,
    tools: List[FunctionTool],
    tracker: Optional[InteractionTracker] = None,
    logger: Optional[ConvLogger] = None,
    ctx_manager: Optional[ContextManager] = None,
    system_prompt: str = "You are a helpful assistant.",
    max_iterations: int = 10,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Run model with tools, handling one tool call per round."""
    tool_map = {t.get_function_name(): t for t in tools}
    tool_schemas = [t.get_openai_tool_schema() for t in tools]

    if tracker:
        tracker.register_tools(llm_id=0, tools=tools)

    # Start live display mode
    if logger:
        logger.start()

    messages = [msg_system(system_prompt), msg_user(question)]
    logger and logger.update(messages)

    for _ in range(max_iterations):
        response = model.run(_clean_for_api(messages), tools=tool_schemas)
        assistant_msg = response.choices[0].message
        tool_call = assistant_msg.tool_calls[0] if assistant_msg.tool_calls else None

        messages.append(msg_assistant(assistant_msg.content, tool_call))
        record_interaction(tracker, messages, llm_id=0, usage=response.usage)

        if not tool_call:
            logger and logger.update(messages)
            break

        # execute the tool
        args = json.loads(tool_call.function.arguments)
        result = execute_tool(tool_map, tool_call.function.name, args)
        messages.append(msg_tool(tool_call.id, result))

        logger and logger.update(messages)

        # Apply context management after each complete round
        if ctx_manager:
            messages = ctx_manager.apply(messages)
            logger and logger.update(messages)

    tracker.final_messages = messages

    logger and logger.stop()
    return assistant_msg.content or "", tracker
    