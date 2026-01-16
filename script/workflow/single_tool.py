"""Minimal example: External tools with manual tool execution loop."""

import os
import json
from typing import List, Optional
from transformers import AutoTokenizer
from dotenv import find_dotenv, load_dotenv

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool

from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.retriever import search_engine
from rosetta.workflow.basic_utils import msg_system, msg_user, msg_assistant, msg_tool, execute_tool

load_dotenv(find_dotenv())

def run_with_tools(
    question: str,
    model,
    tools: List[FunctionTool],
    tracker: Optional[InteractionTracker] = None,
    system_prompt: str = "You are a helpful assistant.",
    max_iterations: int = 10,
    verbose: bool = True,
) -> str:
    """Run model with tools, handling one tool call per round."""
    tool_map = {t.get_function_name(): t for t in tools}
    tool_schemas = [t.get_openai_tool_schema() for t in tools]

    if tracker:
        tracker.register_tools(llm_id=0, tools=tools)

    messages = [msg_system(system_prompt), msg_user(question)]

    for _ in range(max_iterations):
        response = model.run(messages, tools=tool_schemas)
        assistant_msg = response.choices[0].message
        tool_call = assistant_msg.tool_calls[0] if assistant_msg.tool_calls else None

        messages.append(msg_assistant(assistant_msg.content, tool_call))
        record_interaction(tracker, messages, llm_id=0, usage=response.usage)

        if not tool_call:
            return assistant_msg.content or ""

        args = json.loads(tool_call.function.arguments)
        result = execute_tool(tool_map, tool_call.function.name, args)
        messages.append(msg_tool(tool_call.id, result))

        if verbose:
            print(f"[Tool: {tool_call.function.name}] args={args}")
            print(f"[Result] {result[:200]}..." if len(result) > 200 else f"[Result] {result}")

    return "Max iterations reached."


# Configuration
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    model_config_dict={"temperature": 0.0, "max_tokens": 32768, "stream": False},
    api_key=os.getenv("FIREWORKS_API_KEY"),
    url="https://api.fireworks.ai/inference/v1",
)
tokenizer_model_name = "Qwen/Qwen3-32B"
tools = [FunctionTool(search_engine)]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tracker = InteractionTracker(tokenizer=tokenizer)

    question = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"
    answer = run_with_tools(question, model, tools, tracker=tracker)

    print("\n" + "=" * 50)
    print("Final Answer:")
    print(answer)
    print("\n" + "=" * 50)
    print("Chat History:")
    print(tracker.get_message_text(llm_id=0))
    print("\n" + "=" * 50)
    print("Tracker Summary:")
    print(tracker)