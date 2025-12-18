"""
To launch the server:
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path Qwen/Qwen3-32B --host 0.0.0.0 --tp-size 2 --tool-call-parser qwen --port 30000
"""

import os
from transformers import AutoTokenizer

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool, SearchToolkit

from rosetta.context.track import InteractionTracker
from rosetta.workflow.research_flow import direct_subagent_research, extend_subagent_research, extend_sequential_subagent_research, full_subagent_research
from rosetta.workflow.oneflow import do_research
from rosetta.workflow.retriever import search_engine
from rosetta.workflow.hf_qwen_model import HFQwenModel, HFContextAttentionQwenModel
from rosetta.workflow.hf_contextual_qwen_model import HFContextualQwenModel

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

main_system_prompt = "You are a helpful assistant."
main_agent = ChatAgent(
    system_message=main_system_prompt, 
    model=model
)

if __name__ == "__main__":
    # weave.init("nics-efc/camel")
    # with weave.thread(thread_id="subagent_research"):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    tracker = InteractionTracker(tokenizer=tokenizer)

    # search_tool = FunctionTool(search_engine)
    search_tool = FunctionTool(SearchToolkit().search_google)

    response, tracker = direct_subagent_research(
    # response, tracker = extend_subagent_research(
    # response, tracker = extend_sequential_subagent_research(
    # response, tracker = full_subagent_research(
        # question="Were Scott Derrickson and Ed Wood of the same nationality?", 
        # question="What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?", # answer: Animorphs
        question="Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?",
        main_agent=main_agent, 
        search_model=model,
        tracker=tracker,
        search_tool=search_tool
    )

    # Print the tracker
    for llm_id in tracker.get_unique_llm_ids():
        print("="*10 + f" LLM {llm_id} " + "="*10)
        print(tracker.get_message_text(llm_id=llm_id))
    
    print(tracker)
    print(response)