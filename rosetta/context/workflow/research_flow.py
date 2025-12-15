from typing import Tuple, Optional

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.tasks import Task

from rosetta.context.track import InteractionTracker, record_interaction
from rosetta.context.workflow.prompt import SEARCH_TASK_DECOMPOSE_PROMPT, RESPONSE_PROMPT_DIRECT, SEARCH_AGENT_PROMPT
from rosetta.context.workflow.camel_utils import convert_to_camel_messages

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
        search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[google_search_tool])
        response = search_agent.step(query)
        information = response.msg.content
        information_list.append(information)

        # Track search interaction (llm_id=1,2,3... for each search agent)
        record_interaction(tracker, search_agent.chat_history, llm_id=i + 1)

    # Direct reply
    search_info = "\n\n".join(information_list)
    response_prompt = RESPONSE_PROMPT_DIRECT.format(question=question, search_info=search_info)

    response = main_agent.step(response_prompt)

    # Track final response interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    return response.msg.content, tracker


def extend_subagent_research(
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
    messages = main_agent.chat_history
    camel_messages = convert_to_camel_messages(messages, skip_system=True)
    for i, subtask in enumerate(subtasks):
        query = subtask.content
        # Create a new search agent for each query
        search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[google_search_tool])

        # Expand the context from the main agent's chat history
        search_agent.memory.write_records(camel_messages)

        response = search_agent.step(query)
        information = response.msg.content
        information_list.append(information)

        # Track search interaction (llm_id=1,2,3... for each search agent)
        record_interaction(tracker, search_agent.chat_history, llm_id=i + 1)

    # Direct reply
    search_info = "\n\n".join(information_list)
    response_prompt = RESPONSE_PROMPT_DIRECT.format(question=question, search_info=search_info)

    response = main_agent.step(response_prompt)

    # Track final response interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    return response.msg.content, tracker