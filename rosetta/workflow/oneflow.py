import re
from contextlib import nullcontext
from typing import Tuple, Optional
from rich.console import Console

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.memories import ContextRecord, MemoryRecord

from rosetta.context.track import InteractionTracker, record_interaction
from rosetta.workflow.prompt import SEARCH_TASK_DECOMPOSE_PROMPT, TASK_REVISE_PROMPT, FORCE_ANSWER_PROMPT, SEARCH_AGENT_PROMPT
from rosetta.workflow.camel_utils import context_records_to_memory_records, MemoryRecord_flip_role

def _parse_tasks(text: str) -> list:
    """Extract tasks from <tasks><task>...</task></tasks> format."""
    matches = re.findall(r'<task>(.*?)</task>', text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]

def _parse_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> format."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def search_to_main_context_seletor(context_records: list[ContextRecord]) -> Tuple[Optional[list[MemoryRecord]], str]:
    """Select the messages from the search agent to the main agent's context."""
    memory_records = context_records_to_memory_records(context_records)
    if len(memory_records) < 2:
        feedback = memory_records[-1].message.content if memory_records else ""
        return memory_records, feedback
    history = [memory_records[1], memory_records[-1]]
    feedback = memory_records[-1].message.content
    return history, feedback

def do_research(
    question: str,
    main_agent: ChatAgent,
    search_model: BaseModelBackend,
    tracker: InteractionTracker = None,
    search_tool: FunctionTool = None,
    max_rounds: int = 10,
    show_status: bool = True,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Iterative subagent research with task revision loop.
    
    Args:
        question: The question to answer.
        main_agent: The main agent.
        search_model: The search model.
        tracker: The tracker.
        search_tool: The search tool. default is Google search.
        max_rounds: Maximum iterations. default is 10.
        show_status: Show spinner status. Disable when using with tqdm.

    Returns:
        The response message and the tracker.
    """
    status_ctx = Console().status("") if show_status else nullcontext()
    with status_ctx as status:
        def update(msg):
            if status:
                status.update(msg)
        
        update(f"Round 0 | Tasks: 1 | Decomposing: {question}")

        if search_tool is None:
            search_tool = FunctionTool(SearchToolkit().search_google)

        # Initial decomposition
        response = main_agent.step(SEARCH_TASK_DECOMPOSE_PROMPT.format(content=question))
        record_interaction(tracker, main_agent.chat_history, llm_id=0)
        
        collected_info = []
        for round_idx in range(max_rounds):
            # Check for answer first
            answer = _parse_answer(response.msg.content)
            if answer:
                return answer, tracker
            
            tasks = _parse_tasks(response.msg.content)
            update(f"Round {round_idx+1} | Tasks: {len(tasks)} | {tasks[0]}")

            # No tasks left, force answer
            if not tasks:
                info = "\n\n".join(collected_info) if collected_info else "No information found."
                response = main_agent.step(FORCE_ANSWER_PROMPT.format(question=question, info=info))
                record_interaction(tracker, main_agent.chat_history, llm_id=0)
                answer = _parse_answer(response.msg.content)
                return answer if answer else response.msg.content, tracker
            
            # Search agent executes first task
            search_agent = ChatAgent(system_message=SEARCH_AGENT_PROMPT, model=search_model, tools=[search_tool])
            search_response = search_agent.step(tasks[0])
            record_interaction(tracker, search_agent.chat_history, llm_id=round_idx + 1)

            # Return messages to the main agent's context
            shared_context, feedback_content = search_to_main_context_seletor(search_agent.memory.retrieve())
            collected_info.append(feedback_content)

            # Main agent revises tasks
            main_agent.memory.write_records(shared_context)
            response = main_agent.step(TASK_REVISE_PROMPT.format(question=question))
            record_interaction(tracker, main_agent.chat_history, llm_id=0)
    
    # Max rounds reached, force answer
    info = "\n\n".join(collected_info) if collected_info else "No information found."
    response = main_agent.step(FORCE_ANSWER_PROMPT.format(question=question, info=info))
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    answer = _parse_answer(response.msg.content)
    return answer if answer else response.msg.content, tracker

