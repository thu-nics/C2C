"""Tree-based research workflow with rewind capability."""

import re
from typing import Tuple, Optional, List, Any
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from rosetta.workflow.tree_prompt import (
    INIT_PROMPT, REWIND_PROMPT, ANSWER_PROMPT, EXAM_PROMPT
)
from rosetta.workflow.tree_prompt import DECISION_PROMPT
from rosetta.workflow.prompt import SEARCH_AGENT_PROMPT as WORKER_PROMPT
from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.display import StatusLogger
from rosetta.workflow.camel_utils import messages_to_memoryRecords


def _parse_tasks(text: str) -> List[str]:
    """Extract tasks from <tasks><task>...</task></tasks> format."""
    matches = re.findall(r'<task>(.*?)</task>', text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def _parse_task(text: str) -> Optional[str]:
    """Extract single task from <task>...</task> format."""
    match = re.search(r'<task>(.*?)</task>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> format."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_rewind(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract rewind index and summary from rewind agent response."""
    idx_match = re.search(r'<rewind_to>(\d+)</rewind_to>', text)
    sum_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    idx = int(idx_match.group(1)) if idx_match else None
    summary = sum_match.group(1).strip() if sum_match else None
    return idx, summary


def _parse_step(text: str) -> Optional[int]:
    """Extract step index from <step>...</step> format."""
    match = re.search(r'<step>(\d+)</step>', text)
    return int(match.group(1)) if match else None


def _parse_exam_result(text: str) -> Tuple[str, str, str]:
    """Extract verdict, reason, and correction from exam agent response."""
    verdict_match = re.search(r'<verdict>(.*?)</verdict>', text, re.DOTALL)
    reason_match = re.search(r'<reason>(.*?)</reason>', text, re.DOTALL)
    correction_match = re.search(r'<correction>(.*?)</correction>', text, re.DOTALL)
    verdict = verdict_match.group(1).strip().lower() if verdict_match else "unknown"
    reason = reason_match.group(1).strip() if reason_match else ""
    correction = correction_match.group(1).strip() if correction_match else ""
    return verdict, reason, correction


def _parse_action(text: str) -> Optional[str]:
    """Extract action from <action>...</action> format."""
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    return match.group(1).strip().lower() if match else None


def _parse_decision(text: str) -> Tuple[str, dict]:
    """Parse main agent response to determine next state."""
    action = _parse_action(text)
    if action == "answer":
        return "answer", {"answer": _parse_answer(text)}
    if action == "rewind":
        return "rewind", {}
    if action == "revise":
        return "revise", {"tasks": _parse_tasks(text)}
    if action == "execute":
        return "execute", {"task": _parse_task(text)}
    if action == "exam":
        return "exam", {"step": _parse_step(text)}
    return "unknown", {}


def _format_history_numbered(messages: List[dict]) -> str:
    """Format chat history messages as numbered steps for rewind agent.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Should be subtask/feedback pairs (user/assistant alternating).
    """
    lines = []
    step = 0
    # Skip system message if present
    start_idx = 1 if messages and messages[0].get('role') == 'system' else 0
    for i in range(start_idx, len(messages), 2):
        subtask = messages[i].get('content', '') if i < len(messages) else ""
        feedback = messages[i + 1].get('content', '') if i + 1 < len(messages) else ""
        lines.append(f"Step {step}:\n[Subtask] {subtask}\n[Feedback] {feedback}")
        step += 1
    return "\n\n".join(lines)


def _build_decision_prompt(state: str, tasks: List[str], question: str) -> str:
    """Build prompt based on current state."""
    if state == "init":
        return INIT_PROMPT.format(question=question)
    if state == "answer":
        return ANSWER_PROMPT.format(question=question)
    tasks_str = "\n".join(f"- {t}" for t in tasks) if tasks else "(none)"
    return DECISION_PROMPT.format(question=question, tasks=tasks_str)


def _rollback_history(main_agent: ChatAgent, messages: List[dict], rewind_idx: int, summary: str):
    """Rollback history to a rewind point by popping from memory, then adding summary.

    Args:
        main_agent: The main agent.
        messages: Chat history messages (standard format). (Used only for compatibility; the
                  latest history is read from ``main_agent.chat_history``.)
        rewind_idx: Step index to rewind to **after**. Steps `0..rewind_idx` are kept; steps
                   `rewind_idx+1..` are removed.
        summary: Summary of failed path.
    """
    # Mirror `_format_history_numbered`: steps are (user, assistant) pairs starting after an optional system message.
    history = main_agent.chat_history
    start_idx = 1 if history and history[0].get("role") == "system" else 0
    n = max(0, rewind_idx + 1)
    keep_end = start_idx + n * 2 + 1  # keep through the user message of step `rewind_idx`
    keep_end = min(max(keep_end, start_idx), len(history))
    drop_count = len(history) - keep_end
    if drop_count:
        main_agent.memory.pop_records(drop_count)

    # Add summary as assistant message
    if summary:
        summary_records = messages_to_memoryRecords([
            {"role": "assistant", "content": summary}
        ])
        main_agent.memory.write_records(summary_records)


def do_execute(task: str, worker_model: BaseModelBackend, worker_tool: FunctionTool,
               tracker: InteractionTracker = None, step_idx: int = 0,
               update_status: callable = None) -> str:
    """Execute a subtask using worker agent."""
    worker_agent = ChatAgent(
        system_message=WORKER_PROMPT,
        model=worker_model,
        tools=[worker_tool]
    )
    if tracker is not None:
        tracker.register_tools(llm_id=step_idx + 1, tools=[worker_tool])

    response = worker_agent.step(task)
    if update_status:
        update_status(response.msg.content)
    record_interaction(tracker, worker_agent.chat_history, llm_id=step_idx + 1)
    return response.msg.content


def do_rewind(rewind_model: BaseModelBackend, messages: List[dict]) -> Tuple[int, str]:
    """Analyze history and determine rewind point.

    Args:
        rewind_model: Model for rewind agent.
        messages: Chat history messages (standard format).

    Returns:
        Tuple of (rewind_idx, summary).
    """
    history_str = _format_history_numbered(messages)
    rewind_agent = ChatAgent(
        system_message="You analyze research history to find rewind points.",
        model=rewind_model
    )
    response = rewind_agent.step(REWIND_PROMPT.format(history=history_str))
    rewind_idx, summary = _parse_rewind(response.msg.content)
    return rewind_idx or 0, summary or ""


def do_exam(exam_model: BaseModelBackend, messages: List[dict], step_idx: int,
            question: str, tracker: InteractionTracker = None) -> Tuple[str, str, str]:
    """Examine a step's result for correctness.

    Args:
        exam_model: Model for exam agent.
        messages: Chat history messages (standard format).
        step_idx: Step index to examine (0-indexed).
        question: Original question.
        tracker: Interaction tracker.

    Returns:
        Tuple of (verdict, reason, correction).
    """
    # Skip system message
    start_idx = 1 if messages and messages[0].get('role') == 'system' else 0

    # Format main context (all steps before the examined one)
    context_end = start_idx + step_idx * 2
    main_context = _format_history_numbered(messages[:context_end]) if step_idx > 0 else "(no prior steps)"

    # Get the task and result for the step to examine
    task_idx = start_idx + step_idx * 2
    result_idx = start_idx + step_idx * 2 + 1
    task = messages[task_idx].get('content', '') if task_idx < len(messages) else ""
    result = messages[result_idx].get('content', '') if result_idx < len(messages) else ""

    exam_agent = ChatAgent(
        system_message="You are a verification agent that checks for errors in research results.",
        model=exam_model
    )

    prompt = EXAM_PROMPT.format(
        question=question,
        main_context=main_context,
        step_idx=step_idx,
        task=task,
        result=result
    )
    response = exam_agent.step(prompt)
    record_interaction(tracker, exam_agent.chat_history, llm_id=-1)  # Use -1 for exam agent

    return _parse_exam_result(response.msg.content)


def _state_update(state: str, main_agent: ChatAgent, tasks: List[str],
                  question: str, tracker: InteractionTracker = None) -> Tuple[str, dict]:
    """Prompt main agent and return next state."""
    prompt = _build_decision_prompt(state, tasks, question)
    response = main_agent.step(prompt)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    # Remove the prompt+response from history (keep only subtask/feedback pairs)
    main_agent.memory.pop_records(2)

    next_state, data = _parse_decision(response.msg.content)
    message = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response.msg.content}]
    return next_state, data, message


def do_tree_research(
    question: str,
    main_agent: ChatAgent,
    worker_model: BaseModelBackend,
    rewind_model: BaseModelBackend = None,
    exam_model: BaseModelBackend = None,
    tracker: InteractionTracker = None,
    tree_tracker: Any = None,
    worker_tool: FunctionTool = None,
    max_rounds: int = 10,
    show_status: bool = True,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Tree-based research with execute/revise/rewind/exam/answer states.

    Args:
        question: The question to answer.
        main_agent: Main agent for decision making.
        worker_model: Model for worker agent.
        rewind_model: Model for rewind agent. Defaults to worker_model.
        exam_model: Model for exam agent. Defaults to worker_model.
        tracker: Interaction tracker.
        tree_tracker: Tree structure tracker.
        worker_tool: Tool for worker. Defaults to Google search.
        max_rounds: Maximum iterations.
        show_status: Show spinner status.

    Returns:
        Answer string and tracker.
    """
    logger = StatusLogger(enabled=show_status)
    if worker_tool is None:
        worker_tool = FunctionTool(SearchToolkit().search_google)
    if rewind_model is None:
        rewind_model = worker_model
    if exam_model is None:
        exam_model = worker_model

    state = "init"
    tasks = []
    tasks_at_step: dict[int, list] = {}  # snapshot of tasks before each step
    end_idx = 0
    node_id = 0  # monotonically increasing, never resets on rewind

    for _ in range(max_rounds):
        next_state, data, state_conversation = _state_update(state, main_agent, tasks, question, tracker)

        if tree_tracker is not None:
            parent_id = tree_tracker.get_current_parent()
            tree_tracker.add_node(node_id, parent_id, next_state, data)
            node_id += 1

        if next_state == "answer":
            return data["answer"], tracker

        # Build task description for status display
        if next_state == "execute":
            status_tasks = [data["task"]] + tasks[1:]
        elif next_state == "revise":
            status_tasks = ["Revising tasks"]
        elif next_state == "rewind":
            status_tasks = ["Analyzing rewind point"]
        elif next_state == "exam":
            status_tasks = [f"Examining step {data.get('step', 0)}"]
        else:
            break  # Unknown state

        with logger.round(end_idx, status_tasks, state=next_state.capitalize()) as update_status:
            if next_state == "execute":
                tasks_at_step[end_idx] = list(tasks)  # snapshot before execute
                if tree_tracker is not None:
                    tree_tracker.register_step(end_idx, node_id - 1)  # node_id was incremented after add_node
                feedback = do_execute(data["task"], worker_model, worker_tool,
                                      tracker, end_idx, update_status)
                records = messages_to_memoryRecords([
                    {"role": "user", "content": data["task"]},
                    {"role": "assistant", "content": feedback},
                ])
                main_agent.memory.write_records(records)
                tasks = tasks[1:] if tasks else []
                end_idx += 1

            elif next_state == "revise":
                update_status(state_conversation[1]["content"])  # model response with subtasks
                records = messages_to_memoryRecords(state_conversation)
                main_agent.memory.write_records(records)
                tasks = data["tasks"]

            elif next_state == "rewind":
                messages = main_agent.chat_history
                rewind_idx, summary = do_rewind(rewind_model, messages)
                update_status(f"Rewinding to step {rewind_idx}")
                _rollback_history(main_agent, messages, rewind_idx, summary)
                # Recompute end_idx from current history
                history = main_agent.chat_history
                start_idx = 1 if history and history[0].get("role") == "system" else 0
                end_idx = max(0, (len(history) - start_idx) // 2)
                # Restore tasks to the snapshot at rewound step
                tasks = list(tasks_at_step.get(rewind_idx, []))
                if tree_tracker is not None:
                    tree_tracker.mark_rewind(rewind_idx, summary)

            elif next_state == "exam":
                exam_step = data.get("step", 0)
                messages = main_agent.chat_history
                verdict, reason, correction = do_exam(exam_model, messages, exam_step, question, tracker)
                update_status(f"Verdict: {verdict}")
                exam_result = f"[Exam Step {exam_step}] Verdict: {verdict}. {reason}"
                if verdict == "incorrect" and correction:
                    exam_result += f" Correction: {correction}"
                exam_records = messages_to_memoryRecords([
                    {"role": "user", "content": f"Examine step {exam_step}"},
                    {"role": "assistant", "content": exam_result},
                ])
                main_agent.memory.write_records(exam_records)
            else:
                break  # Unknown state

        state = next_state

    # Fallback: force answer
    prompt = ANSWER_PROMPT.format(question=question)
    response = main_agent.step(prompt)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    answer = _parse_answer(response.msg.content)
    return answer or response.msg.content, tracker
