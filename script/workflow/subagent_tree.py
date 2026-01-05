"""
Test script for tree-based research workflow.

To launch the server:

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path Qwen/Qwen3-32B --host 0.0.0.0 --tp-size 2 --dp-size 2 --tool-call-parser qwen --port 30000 --mem-fraction-static 0.8

CUDA_VISIBLE_DEVICES=4,5 python -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-8B --host 0.0.0.0 --tp-size 1 --dp-size 2 --is-embedding --port 30001 --mem-fraction-static 0.8
"""
import os
from transformers import AutoTokenizer
from dotenv import find_dotenv, load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory, GeminiModel, OpenAICompatibleModel
from camel.types import ModelPlatformType, ModelType
from camel.toolkits import FunctionTool, SearchToolkit
from camel.configs import ChatGPTConfig, GeminiConfig

from rosetta.workflow.track import InteractionTracker, TreeTracker
from rosetta.workflow.treeflow import do_tree_research
from rosetta.workflow.retriever import search_engine
from rosetta.workflow.feedback import FeedbackAgent

# Environment Variables
load_dotenv(find_dotenv())

# Local
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="contextual-model",
    model_config_dict={"temperature": 0.0, "max_tokens": 32768, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    api_key="not-needed",
    url="http://localhost:30000/v1",
)

thinking_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="contextual-model",
    model_config_dict={"temperature": 0.0, "max_tokens": 32768, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    api_key="not-needed",
    url="http://localhost:30000/v1",
)

# GPT
# model = ModelFactory.create(
#     model_platform=ModelPlatformType.OPENAI,
#     model_type=ModelType.GPT_5_MINI,
#     model_config_dict=ChatGPTConfig(reasoning_effort="minimal").as_dict(),
#     # model_type=ModelType.GPT_4_1,
#     # model_config_dict=ChatGPTConfig().as_dict()
# )

# Gemini
# model = ModelFactory.create(
#     model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
#     model_type="gemini-3-flash-preview",
#     model_config_dict=ChatGPTConfig(max_tokens=32768, temperature=0.0, reasoning_effort="medium").as_dict(),
#     api_key=os.getenv("GEMINI_API_KEY"),
#     url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

worker_model = model
rewind_model = model
exam_model = thinking_model
think_model = thinking_model

main_system_prompt = "You are a helpful assistant."
main_agent = ChatAgent(
    system_message=main_system_prompt,
    model=model
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    tracker = InteractionTracker(tokenizer=tokenizer, sort_by_llm_id=False)
    tree_tracker = TreeTracker()

    # Choose search tool
    tools = []
    tools.append(FunctionTool(search_engine))
    # tools.append(FunctionTool(SearchToolkit().search_wiki))  # successful
    # tools.append(FunctionTool(SearchToolkit().search_brave))  # successful, but rate limited
    # tools.append(FunctionTool(SearchToolkit().search_google))  # successful
    # tools.append(FunctionTool(SearchToolkit().search_tavily))  # successful
    # tools.append(FunctionTool(SearchToolkit().search_exa))  # successful
    # tools.append(FunctionTool(SearchToolkit().search_alibaba_tongxiao))  # successful
    # tools.append(FunctionTool(SearchToolkit().search_metaso))  # successful

    question = "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"
    # question = "A Japanese manga series based on a 16 year old high school student Ichitaka Seto, is written and illustrated by someone born in what year?"
    # question = "Alfred Balk served as the secretary of the Committee on the Employment of Minority Groups in the News Media under which United States Vice President?"
    # question = "How many copies of Roald Dahl's variation on a popular anecdote sold?" # 250 million
    # question = "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary." # 17
    # question = "I’m researching species that became invasive after people who kept them as pets released them. There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, separated by commas if there is more than one place." # 34689
    # question = "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?"

    state_rule_actions = ["execute", "plan", "answer"]

    response, tracker = do_tree_research(
        question=question,
        state_rule_actions=state_rule_actions,
        main_agent=main_agent,
        worker_model=worker_model,
        rewind_model=rewind_model,
        exam_model=exam_model,
        think_model=think_model,
        tracker=tracker,
        tree_tracker=tree_tracker,
        worker_tools=tools,
        max_rounds=30,
    )

    # Export feedback to CSV
    csv_path = FeedbackAgent.to_csv()
    print(f"\nFeedback exported to: {csv_path}")

    # Print the tracker
    for llm_id in tracker.get_unique_llm_ids():
        print("=" * 10 + f" LLM {llm_id} " + "=" * 10)
        print(tracker.get_message_text(llm_id=llm_id))

    print(tracker)
    print(tree_tracker)
    print("\nAnswer:", response)
