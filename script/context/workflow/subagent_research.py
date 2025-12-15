"""
To launch the server:
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path Qwen/Qwen3-32B --host 0.0.0.0 --tp-size 2 --tool-call-parser qwen --port 30000
"""

import os
from transformers import AutoTokenizer

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from rosetta.context.track import InteractionTracker
from rosetta.context.workflow.research_flow import direct_subagent_research

### Environment Variables ###
from rosetta.context.workflow.API import FIRECRAWL_API_KEY, GOOGLE_API_KEY, SEARCH_ENGINE_ID
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

    response, tracker = direct_subagent_research(
        question="Which magazine was started first Arthur's Magazine or First for Women?", 
        main_agent=main_agent, 
        search_model=model,
        tracker=tracker
    )

    # Print the tracker
    for llm_id in tracker.get_unique_llm_ids():
        print("="*10 + f" LLM {llm_id} " + "="*10)
        print(tracker.get_message_text(llm_id=llm_id))
    
    print(tracker)
    print(response)