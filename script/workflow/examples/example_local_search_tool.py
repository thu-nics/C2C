"""Test the search_engine tool with ChatAgent.

To launch the embedding server:
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-0.6B --host 0.0.0.0 --port 30001 --is-embedding

To launch the LLM server:
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path Qwen/Qwen3-32B --host 0.0.0.0 --tp-size 2 --tool-call-parser qwen --port 30000
"""

import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType
from camel.toolkits import SearchToolkit
from dotenv import find_dotenv, load_dotenv

from rosetta.workflow.retriever import search_engine

# Environment Variables
load_dotenv(find_dotenv())

def main():
    # Create model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="contextual-model",
        model_config_dict={
            "temperature": 0.0,
            "max_tokens": 8192,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        api_key="not-needed",
        url="http://localhost:30000/v1",
    )

    # Create search tool
    tools = []

    # tools.append(FunctionTool(SearchToolkit().search_linkup)) # failed due to access denied
    # tools.append(FunctionTool(SearchToolkit().search_duckduckgo)) # failed due to rate limit
    # tools.append(FunctionTool(SearchToolkit().search_baidu)) # failed due to security check
    # tools.append(FunctionTool(SearchToolkit().search_bing)) # failed due to chaotic results
    
    # tools.append(FunctionTool(search_engine))
    # tools.append(FunctionTool(SearchToolkit().search_wiki)) # successful
    # tools.append(FunctionTool(SearchToolkit().search_brave)) # successful
    tools.append(FunctionTool(SearchToolkit().search_google)) # successful
    # tools.append(FunctionTool(SearchToolkit().search_tavily)) # successful
    # tools.append(FunctionTool(SearchToolkit().search_exa)) # successful
    # tools.append(FunctionTool(SearchToolkit().search_alibaba_tongxiao)) # successful
    # tools.append(FunctionTool(SearchToolkit().search_metaso)) # successful

    # Test 1: ChatAgent with tool
    print("=" * 60)
    print("Test 1: ChatAgent with search_engine tool")
    print("=" * 60)

    agent = ChatAgent(
        system_message="You are a helpful assistant to do search task.",
        model=model,
        tools=tools,
    )

    query = "What nationality is Scott Derrickson?"
    response = agent.step(query)
    print(f"\nQuery: {query}")
    print(f"Response: {response.msg.content}")

    if response.info.get("tool_calls"):
        print(f"\nTool calls made: {len(response.info['tool_calls'])}")
        for tc in response.info["tool_calls"]:
            print(f"  - {tc.tool_name}({tc.args})")

    print(agent.chat_history)



    # Test 2: Direct function call
    print("\n" + "=" * 60)
    print("Test 2: Direct search_engine function call")
    print("=" * 60)

    results = search_engine("Scott Derrickson nationality", top_k=3)
    for r in results:
        print(f"\n[{r['result_id']}] {r['title']}")
        print(f"    {r['description'][:150]}...")


if __name__ == "__main__":
    main()

