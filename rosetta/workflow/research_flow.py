from typing import Tuple, Optional

from camel.agents import ChatAgent
from camel.messages import FunctionCallingMessage
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.tasks import Task
from camel.types import OpenAIBackendRole

from rosetta.context.track import InteractionTracker, record_interaction
from rosetta.workflow.prompt import (
    SEARCH_TASK_DECOMPOSE_PROMPT, SEARCH_AGENT_PROMPT, RESPONSE_PROMPT_DIRECT
)
from rosetta.workflow.camel_utils import messages_to_camel_messages, context_records_to_memory_records, MemoryRecord_flip_role

def direct_subagent_research(
    question: str,
    main_agent: ChatAgent,
    search_model: BaseModelBackend,
    tracker: InteractionTracker = None,
    search_tool: FunctionTool = None,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Direct subagent research with a single search tool.
    
    Args:
        question: The question to answer.
        main_agent: The main agent.
        search_model: The search model.
        tracker: The tracker.
        search_tool: The search tool. default is Google search.

    Returns:
        The response message and the tracker.
    """
    if search_tool is None:
        search_tool = FunctionTool(SearchToolkit().search_google)


    task = Task(
        content=question,
        id="0"
    )
    subtasks = task.decompose(agent=main_agent, prompt=SEARCH_TASK_DECOMPOSE_PROMPT.format(content=question))

    # Track decomposition interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    # Search the internet for information, then summarize by LLM.
    information_list = []
    for i, subtask in enumerate(subtasks):
        query = subtask.content
        # Create a new search agent for each query
        search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[search_tool])
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
    search_tool: FunctionTool = None,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Extend subagent research with a single search tool.
    
    Args:
        question: The question to answer.
        main_agent: The main agent.
        search_model: The search model.
        tracker: The tracker.
        search_tool: The search tool. default is Google search.

    Returns:
        The response message and the tracker.
    """
    if search_tool is None:
        search_tool = FunctionTool(SearchToolkit().search_google)

    task = Task(
        content=question,
        id="0"
    )
    subtasks = task.decompose(agent=main_agent, prompt=SEARCH_TASK_DECOMPOSE_PROMPT.format(content=question))

    # Track decomposition interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    # Search the internet for information, then summarize by LLM.
    information_list = []
    messages = main_agent.chat_history
    camel_messages = messages_to_camel_messages(messages, skip_system=True)
    for i, subtask in enumerate(subtasks):
        query = subtask.content
        # Create a new search agent for each query
        search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[search_tool])

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

def extend_sequential_subagent_research(
    question: str,
    main_agent: ChatAgent,
    search_model: BaseModelBackend,
    tracker: InteractionTracker = None,
    search_tool: FunctionTool = None,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Extend subagent research with a single search tool.
    
    Args:
        question: The question to answer.
        main_agent: The main agent.
        search_model: The search model.
        tracker: The tracker.
        search_tool: The search tool. default is Google search.

    Returns:
        The response message and the tracker.
    """
    if search_tool is None:
        search_tool = FunctionTool(SearchToolkit().search_google)

    task = Task(
        content=question,
        id="0"
    )
    subtasks = task.decompose(agent=main_agent, prompt=SEARCH_TASK_DECOMPOSE_PROMPT.format(content=question))

    # Track decomposition interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

        # Search the internet for information, then summarize by LLM.
    information_list = []
    camel_history = []
    messages = main_agent.chat_history
    camel_messages = messages_to_camel_messages(messages, skip_system=True)
    camel_history.append(camel_messages)
    for i, subtask in enumerate(subtasks):
        query = subtask.content
        # Create a new search agent for each query
        search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[search_tool])

        # Expand the context from the main agent's chat history
        # Expand the all previous response from the search agent
        camel_messages = camel_history[-1]
        extend_messages = [message for message in camel_messages if (message.role_at_backend in (OpenAIBackendRole.USER, OpenAIBackendRole.ASSISTANT) and type(message.message) is not FunctionCallingMessage)]
        search_agent.memory.write_records(extend_messages) # ignore the duplicated parts

        response = search_agent.step(query)
        information = response.msg.content
        information_list.append(information)

        # Track search interaction (llm_id=1,2,3... for each search agent)
        record_interaction(tracker, search_agent.chat_history, llm_id=i + 1)

        # Record search agent camel history
        camel_history.append(context_records_to_memory_records(search_agent.memory.retrieve()))


    # Direct reply
    search_info = "\n\n".join(information_list)
    response_prompt = RESPONSE_PROMPT_DIRECT.format(question=question, search_info=search_info)

    response = main_agent.step(response_prompt)

    # Track final response interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    return response.msg.content, tracker


def full_subagent_research(
    question: str,
    main_agent: ChatAgent,
    search_model: BaseModelBackend,
    tracker: InteractionTracker = None,
    search_tool: FunctionTool = None,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Extend subagent research with a single search tool.
    
    Args:
        question: The question to answer.
        main_agent: The main agent.
        search_model: The search model.
        tracker: The tracker.
        search_tool: The search tool. default is Google search.

    Returns:
        The response message and the tracker.
    """
    if search_tool is None:
        search_tool = FunctionTool(SearchToolkit().search_google)

    task = Task(
        content=question,
        id="0"
    )
    subtasks = task.decompose(agent=main_agent, prompt=SEARCH_TASK_DECOMPOSE_PROMPT.format(content=question))

    # Track decomposition interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    # Search the internet for information, then summarize by LLM.
    information_list = []
    camel_history = []
    messages = main_agent.chat_history
    camel_messages = messages_to_camel_messages(messages, skip_system=True)
    for i, subtask in enumerate(subtasks):
        query = subtask.content
        # Create a new search agent for each query
        search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[search_tool])

        # Expand the context from the main agent's chat history
        search_agent.memory.write_records(camel_messages)

        response = search_agent.step(query)
        information = response.msg.content
        information_list.append(information)

        # Track search interaction (llm_id=1,2,3... for each search agent)
        record_interaction(tracker, search_agent.chat_history, llm_id=i + 1)

        # Record search agent camel history
        camel_history.append(context_records_to_memory_records(search_agent.memory.retrieve()))

    # Expand the context from the search agent's chat history
    for camel_messages in camel_history:
        extend_messages = [message for message in camel_messages if message.role_at_backend is not OpenAIBackendRole.SYSTEM]
        main_agent.memory.write_records(extend_messages[2:]) # ignore the duplicated parts

    
    # Direct reply
    search_info = "\n\n".join(information_list)
    response_prompt = RESPONSE_PROMPT_DIRECT.format(question=question, search_info=search_info)

    response = main_agent.step(response_prompt)

    # Track final response interaction (llm_id=0 for main_agent)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    return response.msg.content, tracker