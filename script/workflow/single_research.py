"""
To launch the server:
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path Qwen/Qwen3-32B --host 0.0.0.0 --tp-size 2 --tool-call-parser qwen --port 30000
"""

import os
from transformers import AutoTokenizer
from dotenv import find_dotenv, load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool, SearchToolkit
from camel.configs import ChatGPTConfig

from rosetta.workflow.track import InteractionTracker
from rosetta.workflow.selector import ContextSelector
from rosetta.workflow.singleflow import single_research
from rosetta.workflow.retriever import search_engine

# Environment Variables
load_dotenv(find_dotenv())

# Local
# model = ModelFactory.create(
#     model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
#     model_type="contextual-model",
#     model_config_dict={"temperature": 0.0, "max_tokens": 32768, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
#     api_key="not-needed",
#     url="http://localhost:30000/v1",
# )

# Gemini
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="gemini-3-flash-preview",
    model_config_dict=ChatGPTConfig(max_tokens=32768, temperature=0.0, reasoning_effort="medium").as_dict(),
    api_key=os.getenv("GEMINI_API_KEY"),
    url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

tools = []
# tools.append(FunctionTool(search_engine))
tools.append(FunctionTool(SearchToolkit().search_wiki))  # successful

main_system_prompt = "You are a helpful assistant."
step_timeout = None  # Optional: set timeout in seconds for agent step
kwargs = {}
if step_timeout is not None:
    kwargs["step_timeout"] = step_timeout
main_agent = ChatAgent(
    system_message=main_system_prompt,
    model=model,
    tools=tools,
    **kwargs,
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    # tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    tracker = InteractionTracker(tokenizer=tokenizer)

    # question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    # question="What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?" # answer: Animorphs
    question="Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"

    context_plan = {
        "main_contextual": ContextSelector(
            select_fn=ContextSelector.select_query_response
        ),
    }

    response, tracker = single_research(
        question=question,
        main_agent=main_agent,
        tracker=tracker,
        context_plan=context_plan,
    )

    for llm_id in tracker.get_unique_llm_ids():
        print("=" * 10 + f" LLM {llm_id} " + "=" * 10)
        print(tracker.get_message_text(llm_id=llm_id))

    print(tracker)
    print(response)
