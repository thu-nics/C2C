import os
import uuid
from typing import List, Tuple, Dict, Optional
from colorama import Fore
from dotenv import load_dotenv

from camel.agents import ChatAgent, SearchAgent
from camel.loaders import Firecrawl
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.prompts import TextPrompt
from camel.responses import ChatAgentResponse
from camel.types import RoleType, OpenAIBackendRole
from camel.types import ModelPlatformType
from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.toolkits import FunctionTool, SearchToolkit
from camel.tasks import Task
from camel.tasks.task_prompt import TASK_DECOMPOSE_PROMPT, TASK_COMPOSE_PROMPT
from camel.memories import ChatHistoryMemory, MemoryRecord
from transformers import AutoTokenizer

from rosetta.context.track import InteractionTracker, record_interaction
from rosetta.context.workflow.prompt import SEARCH_TASK_DECOMPOSE_PROMPT

def direct_subagent_research(
    question: str,
    main_agent: ChatAgent,
    search_model: BaseModelBackend,
    tracker: InteractionTracker = None,
) -> Tuple[str, Optional[InteractionTracker]]:
    task = Task(
        content=question,
        id="0"
    )
    subtasks = task.decompose(agent=main_agent, prompt=SEARCH_TASK_DECOMPOSE_PROMPT.format(content=question))

    # Track decomposition interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    # Search the internet for information, then summarize by LLM.
    information_list = []
    google_search_tool = FunctionTool(SearchToolkit().search_google)
    for i, subtask in enumerate(subtasks):
        query = subtask.content
        # Create a new search agent for each query
        search_agent = ChatAgent(model=search_model, tools=[google_search_tool])
        response = search_agent.step(query)
        information = response.msg.content
        information_list.append(information)

        # Track search interaction (llm_id=1,2,3... for each search agent)
        record_interaction(tracker, search_agent.chat_history, llm_id=i + 1)

    # Direct reply
    search_info = "\n\n".join(information_list)
    response_prompt = f"""Answer the question based on the Search Results.

Question: {question}

Search Results:
{search_info}

Return exactly:
{question}

<1-2 short sentences of justification>
\\[
\\boxed{{<final magazine name only>}}
\\]"""

    response = main_agent.step(response_prompt)

    # Track final response interaction (llm_id=0 for main_agent)
    # Note: chat_history now includes both decompose and final response,
    # but tracker deduplicates by (role, content), so decompose messages
    # are reused from pool and correctly appear as context for this response.
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    return response.msg.content, tracker
