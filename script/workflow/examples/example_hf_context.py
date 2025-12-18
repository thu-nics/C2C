"""
Example: HFContextualQwenModel + drop_messages + real tool calling (weather).

Usage:
    conda activate c2c
    python script/workflow/examples/example_hf_contextual_qwen_drop_messages.py
    python script/workflow/examples/example_hf_contextual_qwen_drop_messages.py --model Qwen/Qwen3-0.6B
"""

import argparse

from camel.agents import ChatAgent
from camel.toolkits import FunctionTool

from rosetta.workflow.hf_qwen_model import HFContextAttentionQwenModel


def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: sunny, 22°C"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model = HFContextAttentionQwenModel(
        args.model,
        model_config_dict={
            "temperature": args.temperature,
            "max_tokens": args.max_new_tokens,
            "extra_body": {"drop_messages": {}},
        },
        enable_thinking=False,
    )
    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        tools=[FunctionTool(get_weather)],
    )

    agent.reset()

    q1 = "What's the weather in Paris? You must call the get_weather tool."
    model.model_config_dict["extra_body"]["drop_messages"] = {}
    r1 = agent.step(q1)
    print("\nQ1:", q1)
    print("A1:", r1.msg.content)
    if r1.info.get("tool_calls"):
        calls = r1.info["tool_calls"]
        print("Tool calls:", [(c.tool_name, c.args, c.result) for c in calls])

    next_user_id = len(agent.chat_history)
    drop_ids = list(range(1, next_user_id))
    drop_messages = {next_user_id: drop_ids}

    q2 = "What is the city I asked the weather for? Answer with just the city."
    model.model_config_dict["extra_body"]["drop_messages"] = drop_messages
    r2 = agent.step(q2)
    print("\nQ2:", q2)
    print("drop_messages:", drop_messages)
    print("A2:", r2.msg.content)


if __name__ == "__main__":
    main()
