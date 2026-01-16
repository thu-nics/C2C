"""Minimal example: External tools with context management."""

import os
from transformers import AutoTokenizer
from dotenv import find_dotenv, load_dotenv

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool

from rosetta.workflow.track import InteractionTracker
from rosetta.workflow.display import ConvLogger
from rosetta.workflow.contextManage import ContextManager
from rosetta.workflow.retriever import search_engine
from rosetta.workflow.singletool import run_with_tools

load_dotenv(find_dotenv())

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
    logger = ConvLogger(tokenizer=tokenizer)
    ctx_manager = None
    # ctx_manager = ContextManager(model, tokenizer=tokenizer)
    

    question = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"
    answer, tracker = run_with_tools(
        question, model, tools,
        tracker=tracker, logger=logger, ctx_manager=ctx_manager
    )

    print("\n" + "=" * 50)
    print("Final Answer:")
    print(answer)

    # register the final answer
    if ctx_manager:
        ctx_manager.apply(tracker.final_messages, dry_run=True)    
        print("\n" + "=" * 50)
        print(ctx_manager)
        print("\n" + "=" * 50)
        print("Node 0 details:")
        print(ctx_manager.nodes[0])
        print("\n" + "=" * 50)

    print(tracker)