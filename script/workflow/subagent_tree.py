"""
Test script for tree-based research workflow.

To launch the server:
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path Qwen/Qwen3-32B --host 0.0.0.0 --tp-size 2 --tool-call-parser qwen --port 30000
"""

import os
from transformers import AutoTokenizer

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool, SearchToolkit

from rosetta.workflow.track import InteractionTracker, TreeTracker
from rosetta.workflow.treeflow import do_tree_research
from rosetta.workflow.retriever import search_engine

### Environment Variables ###
from rosetta.workflow.API import FIRECRAWL_API_KEY, GOOGLE_API_KEY, SEARCH_ENGINE_ID
os.environ["FIRECRAWL_API_KEY"] = FIRECRAWL_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["SEARCH_ENGINE_ID"] = SEARCH_ENGINE_ID
### Environment Variables ###

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="contextual-model",
    model_config_dict={"temperature": 0.0, "max_tokens": 32768, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    api_key="not-needed",
    url="http://localhost:30000/v1",
)

worker_model = model
rewind_model = model

main_system_prompt = "You are a helpful assistant."
main_agent = ChatAgent(
    system_message=main_system_prompt,
    model=model
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    tracker = InteractionTracker(tokenizer=tokenizer, sort_by_llm_id=False)
    tree_tracker = TreeTracker()

    search_tool = FunctionTool(search_engine)
    # search_tool = FunctionTool(SearchToolkit().search_google)

    question = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"
    # question="A Japanese manga series based on a 16 year old high school student Ichitaka Seto, is written and illustrated by someone born in what year?"

    response, tracker = do_tree_research(
        question=question,
        main_agent=main_agent,
        worker_model=worker_model,
        rewind_model=rewind_model,
        tracker=tracker,
        tree_tracker=tree_tracker,
        worker_tool=search_tool,
        max_rounds=30,
    )

    # Print the tracker
    for llm_id in tracker.get_unique_llm_ids():
        print("=" * 10 + f" LLM {llm_id} " + "=" * 10)
        print(tracker.get_message_text(llm_id=llm_id))

    # print(tracker)
    print(tree_tracker)
    print("\nAnswer:", response)
