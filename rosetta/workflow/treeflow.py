"""Tree-based research workflow with rewind capability."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any, Dict
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

from rosetta.workflow.tree_prompt import (
    INIT_PROMPT, REWIND_PROMPT, EXAM_PROMPT, SELECT_PROMPT, build_decision_prompt
)
from rosetta.workflow.prompt import SEARCH_AGENT_PROMPT as WORKER_PROMPT
from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.display import StatusLogger
from rosetta.workflow.camel_utils import messages_to_memoryRecords

# select_model = ModelFactory.create(
#     model_platform=ModelPlatformType.OPENAI,
#     model_type=ModelType.GPT_4_1,
#     model_config_dict=ChatGPTConfig().as_dict()
# )


@dataclass
class StateResult:
    """Result from a state handler.

    Attributes:
        feedback: Display string for status updates.
        records: Messages to write to main agent memory.
        tasks: New task list (None = keep current).
        kwargs: Special state-specific data (e.g., rewind_to, summary).
    """
    feedback: str
    records: List[Dict[str, str]] = field(default_factory=list)
    tasks: Optional[List[str]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class FlowParser:
    """Parser for flow-specific agent responses."""

    _TASK_RE = re.compile(r"<task>(.*?)</task>", re.DOTALL)
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    _REWIND_TO_RE = re.compile(r"<rewind_to>(\d+)</rewind_to>")
    _SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
    _STEP_RE = re.compile(r"<step>(\d+)</step>")
    _VERDICT_RE = re.compile(r"<verdict>(.*?)</verdict>", re.DOTALL)
    _REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)
    _CORRECTION_RE = re.compile(r"<correction>(.*?)</correction>", re.DOTALL)
    _ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
    _SELECT_RE = re.compile(r"<select>(\d+)</select>")

    @classmethod
    def parse_tasks(cls, text: str) -> List[str]:
        """Extract tasks from <tasks><task>...</task></tasks> format."""
        matches = cls._TASK_RE.findall(text)
        return [m.strip() for m in matches if m.strip()]

    @classmethod
    def parse_task(cls, text: str) -> Optional[str]:
        """Extract single task from <task>...</task> format."""
        match = cls._TASK_RE.search(text)
        return match.group(1).strip() if match else None

    @classmethod
    def parse_answer(cls, text: str) -> Optional[str]:
        """Extract answer from <answer>...</answer> format."""
        match = cls._ANSWER_RE.search(text)
        return match.group(1).strip() if match else None

    @classmethod
    def parse_rewind(cls, text: str) -> Tuple[Optional[int], Optional[str]]:
        """Extract rewind index and summary from rewind agent response."""
        idx_match = cls._REWIND_TO_RE.search(text)
        sum_match = cls._SUMMARY_RE.search(text)
        idx = int(idx_match.group(1)) if idx_match else None
        summary = sum_match.group(1).strip() if sum_match else None
        return idx, summary

    @classmethod
    def parse_step(cls, text: str) -> Optional[int]:
        """Extract step index from <step>...</step> format."""
        match = cls._STEP_RE.search(text)
        return int(match.group(1)) if match else None

    @classmethod
    def parse_exam_result(cls, text: str) -> Tuple[str, str, str]:
        """Extract verdict, reason, and correction from exam agent response."""
        verdict_match = cls._VERDICT_RE.search(text)
        reason_match = cls._REASON_RE.search(text)
        correction_match = cls._CORRECTION_RE.search(text)
        verdict = verdict_match.group(1).strip().lower() if verdict_match else "unknown"
        reason = reason_match.group(1).strip() if reason_match else ""
        correction = correction_match.group(1).strip() if correction_match else ""
        return verdict, reason, correction

    @classmethod
    def parse_action(cls, text: str) -> Optional[str]:
        """Extract action from <action>...</action> format."""
        match = cls._ACTION_RE.search(text)
        return match.group(1).strip().lower() if match else None

    @classmethod
    def parse_select(cls, text: str) -> Optional[int]:
        """Extract selection index from <select>...</select> format (1-indexed)."""
        match = cls._SELECT_RE.search(text)
        return int(match.group(1)) if match else None

    @classmethod
    def parse_decision(cls, text: str) -> Tuple[str, dict]:
        """Parse main agent response to determine next state."""
        action = cls.parse_action(text)
        if action == "answer":
            return "answer", {"answer": cls.parse_answer(text)}
        if action == "rewind":
            return "rewind", {}
        if action == "revise":
            return "revise", {"tasks": cls.parse_tasks(text)}
        if action == "execute":
            return "execute", {"task": cls.parse_task(text)}
        if action == "exam":
            return "exam", {"step": cls.parse_step(text)}
        return "unknown", {}


class FlowFormater:
    """Formatter for flow-specific prompts and history."""

    @classmethod
    def format_history_numbered(cls, messages: List[dict]) -> str:
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

    @classmethod
    def build_decision_prompt(cls, state: str, tasks: List[str], question: str) -> str:
        """Build prompt based on available states from state rule.

        Args:
            state: Current state (used to determine available next states).
            tasks: Current task list.
            question: Research question.

        Returns:
            Formatted decision prompt.
        """
        available_actions = state_rule(state)
        return cls.build_action_prompt(available_actions, question, tasks)

    @classmethod
    def build_action_prompt(cls, actions: List[str], question: str, tasks: List[str]) -> str:
        """Build prompt with an explicit set of actions.

        Args:
            actions: List of action names to include in the prompt.
            question: Research question.
            tasks: Current task list.

        Returns:
            Formatted decision prompt.
        """
        if tasks:
            tasks_str = "\n".join(f"- {t}" for t in tasks)
            prompt_template = build_decision_prompt(actions, include_tasks=True)
            return prompt_template.format(question=question, tasks=tasks_str)
        else:
            prompt_template = build_decision_prompt(actions, include_tasks=False)
            return prompt_template.format(question=question)



def state_rule(_current_state: str) -> List[str]:
    """Define available next states based on current state.

    Currently implements trivial rule: all states always available.
    Override this function to implement state-dependent transitions.

    Args:
        _current_state: Current workflow state (unused in trivial implementation).

    Returns:
        List of available action names for next transition.
    """
    return ["execute", "revise", "rewind", "answer"]


def _rollback_history(main_agent: ChatAgent, rewind_idx: int, summary: str):
    """Rollback history to a rewind point by popping from memory, then adding summary.

    Args:
        main_agent: The main agent.
        rewind_idx: Step index to rewind to **after**. Steps `0..rewind_idx` are kept; steps
                   `rewind_idx+1..` are removed.
        summary: Summary of failed path.
    """
    # Steps are (user, assistant) pairs after optional system message
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


def do_execute(
    task: str,
    worker_model: BaseModelBackend,
    worker_tools: List[FunctionTool],
    tracker: InteractionTracker = None,
    step_idx: int = 0,
) -> StateResult:
    """Execute a subtask using worker agent.

    Args:
        task: The subtask to execute.
        worker_model: Model for worker agent.
        worker_tools: Tools available to worker.
        tracker: Interaction tracker.
        step_idx: Current step index.

    Returns:
        StateResult with feedback and records to write.
    """
    worker_agent = ChatAgent(
        system_message=WORKER_PROMPT,
        model=worker_model,
        tools=worker_tools
    )
    if tracker is not None:
        tracker.register_tools(llm_id=step_idx + 1, tools=worker_tools)

    response = worker_agent.step(task)
    feedback = response.msg.content

    # Extract tool names from chat history
    tools_used = []
    for msg in worker_agent.chat_history:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                tool_name = tc.get("function", {}).get("name")
                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

    record_interaction(tracker, worker_agent.chat_history, llm_id=step_idx + 1)

    return StateResult(
        feedback=feedback,
        records=[
            {"role": "user", "content": task},
            {"role": "assistant", "content": feedback},
        ],
        kwargs={"tools_used": tools_used} if tools_used else {},
    )


def _execute_single_tool(
    task: str,
    tool: FunctionTool,
    worker_model: BaseModelBackend,
) -> Tuple[str, str]:
    """Execute task with a single tool. Returns (tool_name, feedback)."""
    worker_agent = ChatAgent(
        system_message=WORKER_PROMPT,
        model=worker_model,
        tools=[tool]
    )
    response = worker_agent.step(task)
    return tool.func.__name__, response.msg.content


def do_wide_execute(
    task: str,
    worker_model: BaseModelBackend,
    worker_tools: List[FunctionTool],
    question: str,
    tracker: InteractionTracker = None,
    step_idx: int = 0,
) -> StateResult:
    """Execute a subtask using multiple worker agents in parallel, one per tool.

    If multiple tools are available, each tool gets its own worker agent.
    A select agent then picks the best response.

    Args:
        task: The subtask to execute.
        worker_model: Model for worker agents.
        worker_tools: Tools available (one agent per tool).
        question: Original research question (for selection).
        tracker: Interaction tracker.
        step_idx: Current step index.

    Returns:
        StateResult with feedback and records to write.
    """
    if len(worker_tools) == 1:
        # Single tool: just use regular execute
        return do_execute(task, worker_model, worker_tools, tracker, step_idx)

    # Execute each tool in parallel
    results: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=len(worker_tools)) as executor:
        futures = {
            executor.submit(_execute_single_tool, task, tool, worker_model): tool
            for tool in worker_tools
        }
        for future in as_completed(futures):
            tool_name, feedback = future.result()
            results.append((tool_name, feedback))

    # Build responses string for select agent (truncate to avoid token limits)
    # ~4 chars per token, 8k tokens total budget for responses
    max_total_chars = 8000 * 4
    max_chars_per_response = max_total_chars // len(results)
    responses_str = "\n\n".join(
        f"Response {i+1} (tool: {name}):\n{feedback[:max_chars_per_response]}{'...' if len(feedback) > max_chars_per_response else ''}"
        for i, (name, feedback) in enumerate(results)
    )

    # Use select agent to pick best response
    select_model = worker_model
    select_agent = ChatAgent(
        system_message="You select the best response among multiple tool outputs.",
        model=select_model
    )
    select_prompt = SELECT_PROMPT.format(
        question=question,
        task=task,
        responses=responses_str
    )
    select_response = select_agent.step(select_prompt)
    selected_idx = FlowParser.parse_select(select_response.msg.content)

    # Default to first if parsing fails
    if selected_idx is None or selected_idx < 1 or selected_idx > len(results):
        selected_idx = 1

    selected_tool, selected_feedback = results[selected_idx - 1]
    # tools_used = [name for name, _ in results]

    return StateResult(
        feedback=selected_feedback,
        records=[
            {"role": "user", "content": task},
            {"role": "assistant", "content": selected_feedback},
        ],
        # kwargs={"tools_used": tools_used, "selected_tool": selected_tool},
        kwargs={"tools_used": [selected_tool]},
    )


def do_revise(
    state_conversation: List[Dict[str, str]],
    data: dict,
    question: str,
    tasks: List[str],
) -> StateResult:
    """Process revised task list from decision.

    Args:
        state_conversation: The decision prompt/response messages.
        data: Parsed decision data containing tasks.
        question: Research question.
        tasks: Current task list.

    Returns:
        StateResult with conversation records and new tasks.
    """
    revise_prompt = FlowFormater.build_action_prompt(["revise"], question, tasks)
    return StateResult(
        feedback=state_conversation[1]["content"],
        records=[
            {"role": "user", "content": revise_prompt},
            state_conversation[1],
        ],
        tasks=data.get("tasks", []),
    )


def do_rewind(rewind_model: BaseModelBackend, messages: List[dict]) -> StateResult:
    """Analyze history and determine rewind point.

    Args:
        rewind_model: Model for rewind agent.
        messages: Chat history messages (standard format).

    Returns:
        StateResult with rewind_to and summary in kwargs.
    """
    history_str = FlowFormater.format_history_numbered(messages)
    rewind_agent = ChatAgent(
        system_message="You analyze research history to find rewind points.",
        model=rewind_model
    )
    response = rewind_agent.step(REWIND_PROMPT.format(history=history_str))
    rewind_idx, summary = FlowParser.parse_rewind(response.msg.content)

    return StateResult(
        feedback=f"Rewinding to step {rewind_idx or 0}",
        records=[],  # Rewind doesn't add records directly
        kwargs={"rewind_to": rewind_idx or 0, "summary": summary or ""},
    )


def do_exam(
    exam_model: BaseModelBackend,
    messages: List[dict],
    step_idx: int,
    question: str,
    tracker: InteractionTracker = None,
) -> StateResult:
    """Examine a step's result for correctness.

    Args:
        exam_model: Model for exam agent.
        messages: Chat history messages (standard format).
        step_idx: Step index to examine (0-indexed).
        question: Original question.
        tracker: Interaction tracker.

    Returns:
        StateResult with exam verdict and records.
    """
    # Skip system message
    start_idx = 1 if messages and messages[0].get('role') == 'system' else 0

    # Format main context (all steps before the examined one)
    context_end = start_idx + step_idx * 2
    main_context = FlowFormater.format_history_numbered(messages[:context_end]) if step_idx > 0 else "(no prior steps)"

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

    verdict, reason, correction = FlowParser.parse_exam_result(response.msg.content)

    # Build exam result feedback
    exam_result = f"[Exam Step {step_idx}] Verdict: {verdict}. {reason}"
    if verdict == "incorrect" and correction:
        exam_result += f" Correction: {correction}"

    return StateResult(
        feedback=f"Verdict: {verdict}",
        records=[
            {"role": "user", "content": f"Examine step {step_idx}"},
            {"role": "assistant", "content": exam_result},
        ],
        kwargs={"verdict": verdict, "reason": reason, "correction": correction},
    )


def main_decision(
    state: str,
    main_agent: ChatAgent,
    tasks: List[str],
    question: str,
    tracker: InteractionTracker = None,
) -> Tuple[str, dict, List[Dict[str, str]]]:
    """Master decision state: prompt main agent and determine next state.

    This is the central hub of the state machine. After each state handler completes,
    control returns here to decide the next transition.

    Args:
        state: Current workflow state.
        main_agent: Main agent for decision making.
        tasks: Current task list.
        question: Research question.
        tracker: Interaction tracker.

    Returns:
        Tuple of (next_state, data, conversation).
    """
    prompt = FlowFormater.build_decision_prompt(state, tasks, question)
    response = main_agent.step(prompt)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    # Remove the prompt+response from history (keep only subtask/feedback pairs)
    main_agent.memory.pop_records(2)

    next_state, data = FlowParser.parse_decision(response.msg.content)
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.msg.content},
    ]
    return next_state, data, conversation


def _post_process_rewind(
    main_agent: ChatAgent,
    result: StateResult,
    tasks_at_step: Dict[int, list],
    tree_tracker: Any = None,
) -> Tuple[int, List[str]]:
    """Post-process rewind state: rollback history and restore tasks.

    Args:
        main_agent: Main agent.
        result: StateResult from do_rewind.
        tasks_at_step: Snapshot of tasks at each step.
        tree_tracker: Tree structure tracker.

    Returns:
        Tuple of (new_end_idx, restored_tasks).
    """
    rewind_to = result.kwargs.get("rewind_to", 0)
    summary = result.kwargs.get("summary", "")

    _rollback_history(main_agent, rewind_to, summary)

    # Recompute end_idx from current history
    history = main_agent.chat_history
    start_idx = 1 if history and history[0].get("role") == "system" else 0
    end_idx = max(0, (len(history) - start_idx) // 2)

    # Restore tasks to the snapshot at rewound step
    restored_tasks = list(tasks_at_step.get(rewind_to, []))

    if tree_tracker is not None:
        tree_tracker.mark_rewind(rewind_to, summary)

    return end_idx, restored_tasks


def do_tree_research(
    question: str,
    main_agent: ChatAgent,
    worker_model: BaseModelBackend,
    rewind_model: BaseModelBackend = None,
    exam_model: BaseModelBackend = None,
    tracker: InteractionTracker = None,
    tree_tracker: Any = None,
    worker_tools: List[FunctionTool] = None,
    max_rounds: int = 10,
    show_status: bool = True,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Tree-based research with execute/revise/rewind/exam/answer states.

    State Machine Structure:
        - _master_decision: Central hub that prompts main agent for next action
        - do_execute: Execute subtask via worker agent
        - do_revise: Update task list from decision
        - do_rewind: Analyze history and backtrack (with post-processing)
        - do_exam: Verify a previous step's result
        - answer: Terminal state, return result

    Args:
        question: The question to answer.
        main_agent: Main agent for decision making.
        worker_model: Model for worker agent.
        rewind_model: Model for rewind agent. Defaults to worker_model.
        exam_model: Model for exam agent. Defaults to worker_model.
        tracker: Interaction tracker.
        tree_tracker: Tree structure tracker.
        worker_tools: List of tools for worker. Defaults to [Google search].
        max_rounds: Maximum iterations.
        show_status: Show spinner status.

    Returns:
        Answer string and tracker.
    """
    logger = StatusLogger(enabled=show_status)
    if worker_tools is None:
        worker_tools = [FunctionTool(SearchToolkit().search_google)]
    if rewind_model is None:
        rewind_model = worker_model
    if exam_model is None:
        exam_model = worker_model

    state = "initial"  # Initial state: start by planning tasks
    tasks: List[str] = []
    tasks_at_step: Dict[int, list] = {}  # snapshot of tasks before each step
    end_idx = 0
    node_id = 0  # monotonically increasing, never resets on rewind

    # Initial state: use INIT_PROMPT to get initial task decomposition
    init_prompt = INIT_PROMPT.format(question=question)
    response = main_agent.step(init_prompt)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    tasks = FlowParser.parse_tasks(response.msg.content)
    state = "revise"  # Move past initial state

    for _ in range(max_rounds):
        # Master decision: determine next state
        next_state, data, conversation = main_decision(
            state, main_agent, tasks, question, tracker
        )

        if tree_tracker is not None:
            parent_id = tree_tracker.get_current_parent()
            tree_tracker.add_node(node_id, parent_id, next_state, data)
            node_id += 1

        if next_state == "answer":
            return data["answer"], tracker

        # Build status description
        status_map = {
            "execute": [data.get("task", "")] + tasks[1:],
            "revise": ["Revising tasks"],
            "rewind": ["Analyzing rewind point"],
            "exam": [f"Examining step {data.get('step', 0)}"],
        }
        status_tasks = status_map.get(next_state)
        if status_tasks is None:
            break  # Unknown state

        with logger.round(end_idx, status_tasks, state=next_state.capitalize()) as update_status:
            # Dispatch to state handler
            if next_state == "execute":
                tasks_at_step[end_idx] = list(tasks)
                if tree_tracker is not None:
                    tree_tracker.register_step(end_idx, node_id - 1)
                result = do_execute(data["task"], worker_model, worker_tools, tracker, end_idx)
                # result = do_wide_execute(data["task"], worker_model, worker_tools, question, tracker, end_idx)
                tasks = tasks[1:] if tasks else []
                end_idx += 1

            elif next_state == "revise":
                result = do_revise(conversation, data, question, tasks)

            elif next_state == "rewind":
                result = do_rewind(rewind_model, main_agent.chat_history)

            elif next_state == "exam":
                result = do_exam(
                    exam_model, main_agent.chat_history, data.get("step", 0), question, tracker
                )

            else:
                break  # Unknown state

            # Update status display
            tools_used = result.kwargs.get("tools_used")
            update_status(result.feedback, tools_used=tools_used)

            # Record tools used for this round
            if tree_tracker is not None:
                tree_tracker.record_tools_used(tools_used or [])

            # Post-process kwargs (special handling)
            if "rewind_to" in result.kwargs:
                end_idx, tasks = _post_process_rewind(
                    main_agent, result, tasks_at_step, tree_tracker
                )
            else:
                # Standard processing: write records, update tasks
                if result.records:
                    main_agent.memory.write_records(messages_to_memoryRecords(result.records))
                if result.tasks is not None:
                    tasks = result.tasks

        state = next_state

    # Fallback: force answer
    if tasks:
        tasks_str = "\n".join(f"- {t}" for t in tasks)
        prompt_template = build_decision_prompt(["answer"], include_tasks=True)
        prompt = prompt_template.format(question=question, tasks=tasks_str)
    else:
        prompt_template = build_decision_prompt(["answer"], include_tasks=False)
        prompt = prompt_template.format(question=question)
    response = main_agent.step(prompt)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)
    answer = FlowParser.parse_answer(response.msg.content)
    return answer or response.msg.content, tracker
