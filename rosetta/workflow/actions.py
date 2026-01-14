"""Action tools for tree-based research workflow.

Each action class contains:
- interface: Static method that LLMs see as a tool. Docstring becomes tool description.
- do: Static method that executes the action. Called externally with full context.
- parse (optional): Static method for parsing action-specific responses.

Currently supported actions: execute, plan, think, answer
"""

import re
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, TYPE_CHECKING

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from rosetta.workflow.tool_prompt import (
    THINK_PROMPT,
    EXECUTE_CHECK_PROMPT,
    WORKER_PROMPT,
)
from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.camel_utils import context_records_to_memory_records
from rosetta.workflow.fewshot import FewShotManager

"""
ACTION REGISTRY
"""

# Global registries populated by @register_action decorator
ACTIONS: Dict[str, type] = {}
SUPPORTED_ACTIONS: set = set()


def register_action(name: str, supported: bool = True):
    """Decorator to register an action class.

    Args:
        name: Action name (used as tool name).
        supported: Whether do() is implemented. If False, do() should raise NotImplementedError.

    Example:
        @register_action("execute", supported=True)
        class ExecuteAction(Action):
            @staticmethod
            def interface(task: str) -> dict:
                ...
            @staticmethod
            def do(task: str, ...) -> StateResult:
                ...
    """
    def decorator(cls):
        cls.name = name
        ACTIONS[name] = cls
        if supported:
            SUPPORTED_ACTIONS.add(name)
        return cls
    return decorator


"""
STATE RESULT CLASS
"""

@dataclass
class StateResult:
    """Result from a state handler.

    Attributes:
        feedback: Display string for status updates.
        records: Messages to write to main agent memory.
        tasks: New task list (None = keep current).
        state: Workflow state that produced this result.
        chat_history: Subagent chat history as MemoryRecords.
        kwargs: Special state-specific data (e.g., rewind_to, summary).
    """
    feedback: str
    records: List[Dict[str, str]] = field(default_factory=list)
    tasks: Optional[List[str]] = None
    state: Optional[str] = None
    chat_history: Optional[List[Any]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

"""
Context
"""

class ContextData:
    """Data container for workflow state.

    Stores task lists, history records, and snapshots. Update logic is handled
    by Action.update() methods, not by this class.

    Task lifecycle:
        pending -> current (in_progress) -> finished
    """

    def __init__(self):
        # Task lists
        self.pending: List[str] = []
        self.current: List[str] = []  # Current task (0 or 1 item)
        self.finished: List[str] = []
        # Accumulated history records (for syncing to main agent each round)
        self.history_records: List[Dict[str, str]] = []
        # Tracking
        self.round: int = 0  # Always increases (iteration counter)
        self.step: int = 0   # Resets on rewind (execution step)
        # Snapshots: round -> full state (action, result, tasks, step, history)
        self._snapshots: Dict[int, dict] = {}
        # Step-to-round mapping for rewind lookup
        self._step_to_round: Dict[int, int] = {}

    @property
    def stateResults_sequence(self) -> List[StateResult]:
        """All state results in round order."""
        return [self._snapshots[r]["result"] for r in sorted(self._snapshots.keys())]

    @property
    def action_sequence(self) -> List[str]:
        """All actions in round order."""
        return [self._snapshots[r]["action"] for r in sorted(self._snapshots.keys())]

    def snapshot(self, action: str, result: StateResult) -> None:
        """Snapshot AFTER action execution. Records full state and increments round."""
        self._snapshots[self.round] = {
            "action": action,
            "result": result,
            "pending": list(self.pending),
            "current": list(self.current),
            "finished": list(self.finished),
            "step": self.step,
            "history_len": len(self.history_records),
        }
        self._step_to_round[self.step] = self.round
        self.round += 1

    def promote_task(self) -> None:
        """Promote first pending task to current if current is empty."""
        if not self.current and self.pending:
            self.current.append(self.pending.pop(0))

"""
BASE ACTION CLASS
"""


class ActionClass(ABC):
    """Base class for all workflow actions.

    Each action must implement:
    - interface: Static method that LLMs see as a tool. Docstring becomes tool description.
    - do: Static method that executes the action. Called externally with full context.

    Optional:
    - display: Static method returning status description for UI.
    - update: Static method to update ContextData after action execution.

    Attributes:
        name: Action name (set by @register_action decorator).
    """

    name: str = ""  # Set by @register_action decorator

    @staticmethod
    @abstractmethod
    def interface(*args, **kwargs) -> dict:
        """Define the tool interface that LLMs see.

        The docstring of this method becomes the tool description.
        Must return a dict echoing back the arguments.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def do(*args, **kwargs) -> StateResult:
        """Execute the action with full context.

        Called externally with all necessary dependencies injected.
        Must return a StateResult.
        """
        raise NotImplementedError

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for UI display.

        Args:
            data: Arguments passed to the action (from tool call).

        Returns:
            Human-readable status description.
        """
        return "Processing..."

    @staticmethod
    def update(ctx: "ContextData", result: StateResult) -> None:
        """Update ContextData after action execution.

        Default implementation:
        - Increments step counter
        - Extends history_records with result.records
        - Promotes first pending task to current if current is empty

        Override in subclasses for action-specific behavior.

        Args:
            ctx: The ContextData to update.
            result: The StateResult from action execution.
        """
        ctx.step += 1
        ctx.history_records.extend(result.records)
        ctx.promote_task()


"""
ACTION CLASSES
"""

@register_action("execute", supported=True)
class ExecuteAction(ActionClass):
    """Execute a subtask using a worker agent."""

    @staticmethod
    def _parse_status(text: str) -> tuple:
        """Extract completion status and note from execute check response.

        Args:
            text: Response text containing <status> and <note> tags.

        Returns:
            Tuple of (status, note). Status is one of: success, partial, fail.
        """
        status_match = re.search(r"<status>(.*?)</status>", text, re.DOTALL)
        note_match = re.search(r"<note>(.*?)</note>", text, re.DOTALL)
        status = status_match.group(1).strip().lower() if status_match else "success"
        note = note_match.group(1).strip() if note_match else ""
        if status not in ("success", "partial", "fail"):
            status = "success"
        return status, note

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for execute action."""
        return data.get("task", "Executing task")

    @staticmethod
    def interface(task: str) -> dict:
        """Execute a subtask using a worker agent.

        Work on the current task by delegating to a subagent that can search
        the internet or answer directly.

        Guidelines:
        - Include ALL context the subagent needs in the task description.
        - The subagent cannot see any prior conversation.
        - Include what was already found; focus only on the remaining information needed.

        Args:
            task: Complete description of the subtask with all necessary context.

        Returns:
            Dict with task echoed back.
        """
        return {"task": task}

    @staticmethod
    def do(
        task: str,
        worker_model: BaseModelBackend,
        worker_tools: List[FunctionTool],
        tracker: InteractionTracker = None,
        step_idx: int = 0,
        max_iteration: Optional[int] = None,
        num_fewshot: int = 0,
    ) -> StateResult:
        """Execute a subtask using worker agent with completion check.

        Args:
            task: The subtask to execute.
            worker_model: Model for worker agent.
            worker_tools: Tools available to worker.
            tracker: Interaction tracker.
            step_idx: Current step index.
            max_iteration: Maximum number of iterations.
            num_fewshot: Number of few-shot examples to add.

        Returns:
            StateResult with feedback, completion status, and records.
        """
        worker_agent = ChatAgent(
            system_message=WORKER_PROMPT,
            model=worker_model,
            tools=worker_tools,
            max_iteration=max_iteration
        )

        if num_fewshot > 0:
            fewshot = FewShotManager()
            fewshot.add_to_agent_memory(worker_agent, n=num_fewshot, selection="first", add_separator=True)

        if tracker is not None:
            tracker.register_tools(llm_id=step_idx + 1, tools=worker_tools)

        response = worker_agent.step(task)
        response_text = response.msg.content
        tool_msgs_id = [i for i, msg in enumerate(worker_agent.chat_history) if msg['role'] == 'tool']

        # Check completion status
        check_prompt = EXECUTE_CHECK_PROMPT.format(task=task)
        check_response = worker_agent.step(check_prompt)
        completion_status, completion_note = ExecuteAction._parse_status(check_response.msg.content)

        feedback = f"| {completion_status.capitalize()} | Sub-round {len(tool_msgs_id)} : {response_text}"
        if completion_status != "success" and completion_note:
            feedback += f"\n{completion_note}"

        chat_history = context_records_to_memory_records(worker_agent.memory.retrieve())

        # Extract tool names
        tools_used = []
        for msg in worker_agent.chat_history:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tool_name = tc.get("function", {}).get("name")
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)

        record_interaction(tracker, worker_agent.chat_history, llm_id=step_idx + 1)

        kwargs = {
            "num_tool_calls": len(tool_msgs_id),
            "completion_status": completion_status,
            "completion_note": completion_note,
            "raw_chat_history": worker_agent.chat_history,
        }
        if tools_used:
            kwargs["tools_used"] = tools_used

        return StateResult(
            feedback=feedback,
            records=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": response_text},
            ],
            state="execute",
            chat_history=chat_history,
            kwargs=kwargs,
        )

    @staticmethod
    def update(ctx: "ContextData", result: StateResult) -> None:
        """Update context after execute action.

        On success, moves current task to finished.
        """
        ctx.step += 1
        ctx.history_records.extend(result.records)
        status = result.kwargs.get("completion_status", "success")
        if status == "success" and ctx.current:
            ctx.finished.append(ctx.current.pop(0))
        ctx.promote_task()


@register_action("plan", supported=True)
class PlanAction(ActionClass):
    """Plan and update task list."""

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for plan action."""
        return "Planning tasks"

    @staticmethod
    def interface(tasks: List[str]) -> dict:
        """Replace current and pending tasks with a list of new tasks.

        Replaces all current and pending tasks with new tasks. Finished tasks
        are preserved. Use this to decompose the problem or adjust strategy.

        Guidelines:
        - Each task must be narrowly scoped and achievable within a few searches.
        - Each task should have a clear desired result.
        - For failed/partial tasks, change the approach rather than repeating.
        - You may split one subtask into multiple smaller subtasks when helpful.

        Args:
            tasks: List of new task descriptions.

        Returns:
            Dict with tasks echoed back.
        """
        return {"tasks": tasks}

    @staticmethod
    def do(
        tasks: List[str],
        conversation: List[Dict[str, str]],
        main_agent: ChatAgent,
        question: str,
    ) -> StateResult:
        """Process planned task list.

        Args:
            tasks: New task list from LLM.
            conversation: The decision prompt/response messages.
            main_agent: Main agent for chat history.

        Returns:
            StateResult with conversation records and new tasks.
        """
        numbered_tasks = [f"[{i+1}] {task}" for i, task in enumerate(tasks)]
        plan_str = " ".join(numbered_tasks)
        records = [
            {"role": "user", "content": f"Plan the next steps to solve the question: {question}"}, 
            {"role": "assistant", "content": "\n".join(numbered_tasks)}
        ]
        return StateResult(
            feedback=plan_str,
            records=records,
            state="plan",
            chat_history=dict(),
            tasks=tasks,
        )

    @staticmethod
    def update(ctx: "ContextData", result: StateResult) -> None:
        """Update context after plan action.

        Replaces current and pending tasks with new task list.
        """
        ctx.step += 1
        ctx.history_records.extend(result.records)
        ctx.current = []
        ctx.pending = result.tasks if result.tasks else []
        ctx.promote_task()

@register_action("answer", supported=True)
class AnswerAction(ActionClass):
    """Provide the final answer."""

    @staticmethod
    def interface(answer: str, justification: str = "") -> dict:
        """Provide the final answer to the user.

        Use this action only when you have enough verified information to respond
        conclusively.

        Guidelines:
        - Use only when sufficient information has been gathered.
        - Keep justification to 1-2 short sentences.
        - Put the final answer span in 'answer', brief rationale in 'justification'.

        Args:
            answer: The final answer span.
            justification: Brief 1-2 sentence rationale (optional).

        Returns:
            Dict with answer and justification echoed back.
        """
        return {"answer": answer, "justification": justification}

    @staticmethod
    def do(answer: str, justification: str = "") -> StateResult:
        """Return the final answer (no execution needed).

        Args:
            answer: The final answer.
            justification: Brief rationale.

        Returns:
            StateResult with answer.
        """
        
        return StateResult(
            feedback=f"Answer: {answer}",
            records=[
                {"role": "user", "content": f"Based on the research above, answer the question."},
                {"role": "assistant", "content": f"Answer: {answer}\nJustification: {justification}"},
            ],
            state="answer",
            kwargs={"answer": answer, "justification": justification},
        )



@register_action("think", supported=True)
class ThinkAction(ActionClass):
    """Reflect on current status."""

    @staticmethod
    def _parse(text: str) -> Optional[str]:
        """Extract thought summary from think response.

        Args:
            text: Response text containing <summarize> tag.

        Returns:
            Extracted thought or None if not found.
        """
        match = re.search(r"<summarize>(.*?)</summarize>", text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for think action."""
        return "Thinking about next steps"

    @staticmethod
    def interface() -> dict:
        """Pause to reflect and get a concise assessment before choosing next action.

        Use this action when the next operation is unclear and you need a brief
        assessment before proceeding.

        Guidelines:
        - Use sparingly, only when genuinely uncertain about direction.
        - The reflection will provide actionable next steps.

        Returns:
            Empty dict (no arguments needed).
        """
        return {}

    @staticmethod
    def do(
        think_model: BaseModelBackend,
        main_agent: ChatAgent,
        question: str,
        tracker: InteractionTracker = None,
    ) -> StateResult:
        """Reflect on current status and suggest next steps.

        Args:
            think_model: Model for think agent.
            main_agent: Main agent for context.
            question: Research question.
            tracker: Interaction tracker.

        Returns:
            StateResult with reflection.
        """
        think_agent = ChatAgent(
            system_message="You are a reflection agent that suggests next steps.",
            model=think_model
        )
        context_records = main_agent.memory.retrieve()[1:]
        forward_records = context_records_to_memory_records(context_records)
        think_agent.memory.write_records(forward_records)

        response = think_agent.step(THINK_PROMPT.format(question=question))
        response.msg.content
        record_interaction(tracker, think_agent.chat_history, llm_id=-2)

        thought = ThinkAction._parse(response.msg.content) or response.msg.content
        return StateResult(
            feedback=thought,
            records=[
                {"role": "user", "content": "Think about next steps"},
                {"role": "assistant", "content": thought},
            ],
            state="think",
            chat_history=context_records_to_memory_records(think_agent.memory.retrieve()),
        )


# =============================================================================
# UNSUPPORTED ACTIONS
# These actions are defined but not yet implemented. They raise NotImplementedError
# when their do() method is called.
# =============================================================================


@register_action("parallel_execute", supported=False)
class ParallelExecuteAction(ActionClass):
    """Run multiple tasks in parallel.

    NOT SUPPORTED: Parallel execution requires thread pool management and
    result aggregation logic. Use sequential ExecuteAction calls instead.
    """

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for parallel execute action."""
        tasks = data.get("tasks", [])
        return f"Executing {len(tasks)} tasks in parallel"

    @staticmethod
    def interface(tasks: List[str]) -> dict:
        """Run independent tasks in parallel using multiple worker agents.

        Use when multiple tasks are independent and can be executed concurrently.
        Results will be collected and presented as multi-round conversation records.

        Guidelines:
        - Use only when tasks are truly independent of each other.
        - Each task must be self-contained with all context needed.
        - Subagents cannot see prior conversation or each other's results.

        Args:
            tasks: List of independent task descriptions.

        Returns:
            Dict with tasks echoed back.
        """
        return {"tasks": tasks}

    @staticmethod
    def do(*args, **kwargs) -> StateResult:
        """NOT IMPLEMENTED: Parallel execution not yet supported.

        Reason: Requires ThreadPoolExecutor management, result aggregation,
        and handling of concurrent tracker updates. Use sequential execute calls.
        """
        raise NotImplementedError(
            "ParallelExecuteAction.do() is not supported. "
            "Parallel execution requires thread pool management and result aggregation. "
            "Use sequential ExecuteAction calls instead."
        )


@register_action("rewind", supported=False)
class RewindAction(ActionClass):
    """Backtrack to a previous state.

    NOT SUPPORTED: Rewind requires history analysis, turn identification,
    and state rollback logic that is not yet implemented.
    """

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for rewind action."""
        return "Analyzing rewind point"

    @staticmethod
    def interface() -> dict:
        """Backtrack when similar tasks fail repeatedly or exploring wrong direction.

        Use when similar tasks fail repeatedly (similar query variations, similar
        dead ends) or when exploring the wrong entity or topic.

        Guidelines:
        - Use when multiple similar approaches have failed.
        - After rewinding, you must switch to a different approach.
        - Rewind analyzes history to find the best backtrack point.

        Returns:
            Empty dict (no arguments needed).
        """
        return {}

    @staticmethod
    def do(rewind_model: BaseModelBackend, messages: List[dict]) -> StateResult:
        """NOT IMPLEMENTED: Rewind not yet supported.

        Reason: Requires parsing numbered history turns, identifying optimal
        backtrack points, and rolling back FlowContext state (tasks, history).
        """
        raise NotImplementedError(
            "RewindAction.do() is not supported. "
            "Rewind requires history analysis, turn identification, and state rollback. "
            "This feature is planned for future implementation."
        )
        history_str = RewindAction.format_history_numbered(messages)
        rewind_agent = ChatAgent(
            system_message="You analyze research history to find rewind points.",
            model=rewind_model
        )
        response = rewind_agent.step(REWIND_PROMPT.format(history=history_str))
        rewind_turn, summary = RewindAction.parse_rewind(response.msg.content)

        return StateResult(
            feedback=f"Rewinding to turn {rewind_turn or 0}",
            records=[],  # Rewind doesn't add records directly
            state="rewind",
            chat_history=context_records_to_memory_records(rewind_agent.memory.retrieve()),
            kwargs={"rewind_to_step": rewind_turn or 0, "summary": summary or ""},
        )

    @staticmethod
    def update(ctx: "ContextData", result: StateResult) -> None:
        """Update context after rewind action.

        Rolls back history and tasks to the target step.
        Does NOT increment step (rewind resets it).
        """
        rewind_to_step = result.kwargs.get("rewind_to_step", 0)
        summary = result.kwargs.get("summary", "")

        # Find the target round from step
        target_round = ctx._step_to_round.get(rewind_to_step)
        if target_round is not None:
            snapshot = ctx._snapshots.get(target_round, {})
            # Restore task lists
            ctx.pending = list(snapshot.get("pending", []))
            ctx.current = list(snapshot.get("current", []))
            ctx.finished = list(snapshot.get("finished", []))
            ctx.step = snapshot.get("step", rewind_to_step)
            # Rollback history to snapshot point
            history_len = snapshot.get("history_len", 0)
            ctx.history_records = ctx.history_records[:history_len]
            # Clear step_to_round entries after current step
            ctx._step_to_round = {s: r for s, r in ctx._step_to_round.items() if s <= ctx.step}

        # Add summary as assistant message if provided
        if summary:
            ctx.history_records.append({"role": "assistant", "content": summary})


@register_action("exam", supported=False)
class ExamAction(ActionClass):
    """Verify a previous step's result.

    NOT SUPPORTED: Exam requires accessing step history, formatting subagent
    chat logs, and parsing verification verdicts.
    """

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for exam action."""
        step = data.get("step", 0)
        return f"Examining step {step}"

    @staticmethod
    def interface(step: int) -> dict:
        """Verify a previous step's result for correctness.

        Use this action to verify a specific prior step when its result is
        suspicious, inconsistent, or high-impact.

        Guidelines:
        - Use when a result seems suspicious or inconsistent.
        - Use for high-impact results that affect the final answer.
        - Provide the 1-indexed step number to examine.

        Args:
            step: The 1-indexed step number to examine.

        Returns:
            Dict with step echoed back.
        """
        return {"step": step}

    @staticmethod
    def do(*args, **kwargs) -> StateResult:
        """NOT IMPLEMENTED: Exam not yet supported.

        Reason: Requires accessing StateResult history by step index, formatting
        subagent chat histories for verification, and parsing exam verdicts.
        """
        raise NotImplementedError(
            "ExamAction.do() is not supported. "
            "Exam requires step history access and verification verdict parsing. "
            "This feature is planned for future implementation."
        )

def get_action_tools(actions: List[str]) -> List:
    """Get interface functions for specified action names.

    Args:
        actions: List of action names to include.

    Returns:
        List of FunctionTool objects with action names as tool names.
    """
    tools = []
    for name in actions:
        if name not in ACTIONS:
            continue
        action_class = ACTIONS[name]
        # Create wrapper with correct name (so FunctionTool uses action name, not "interface")
        @functools.wraps(action_class.interface)
        def wrapper(*args, _interface=action_class.interface, **kwargs):
            return _interface(*args, **kwargs)
        wrapper.__name__ = name
        tools.append(FunctionTool(wrapper))
    return tools

def is_action_supported(action_name: str) -> bool:
    """Check if an action's do() method is implemented.

    Args:
        action_name: Name of the action to check.

    Returns:
        True if the action is fully supported, False otherwise.
    """
    return action_name in SUPPORTED_ACTIONS
