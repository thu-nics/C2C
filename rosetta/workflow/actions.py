"""Action classes for workflow orchestration.

Each action class contains:
- interface: Static method that LLMs see as a tool. Docstring becomes tool description.
- do: Static method that executes the action. Called externally with full context.
- display: Static method returning status description for UI.
- update: Static method to update ContextData after action execution.

Tree-flow specific (XML-based prompts):
- format_template: XML format template string.
- tree_description: Short description for tree prompts.
- guidelines: Usage guidelines list.
- with_param: Whether action requires parameter re-prompting in focused mode.
- parse: Static method to extract action data from XML response.

Supported actions: execute, plan, think, answer, continue, break
Extended actions (ext_actions.py): parallel_execute, rewind, exam
"""

import re
import functools
import json
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

    Attributes:
        pending: Tasks not yet started.
        current: Current task being worked on (0 or 1 item).
        finished: Tasks fully completed.
        history_records: Accumulated message records for agent memory sync.
        round: Iteration counter (always increases).
        step: Execution step counter (resets on rewind).
        main_agent: Optional reference to main agent (for treeflow compatibility).
    """

    def __init__(self, main_agent: "ChatAgent" = None):
        """Initialize context data.

        Args:
            main_agent: Optional ChatAgent reference (used by treeflow for memory ops).
        """
        # Optional main agent reference (for treeflow)
        self.main_agent = main_agent
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
    def progress(self) -> int:
        """Number of finished tasks."""
        return len(self.finished)

    @property
    def total(self) -> int:
        """Total number of tasks."""
        return len(self.pending) + len(self.current) + len(self.finished)

    @property
    def stateResults_sequence(self) -> List[StateResult]:
        """All state results in round order."""
        return [self._snapshots[r]["result"] for r in sorted(self._snapshots.keys())]

    @property
    def action_sequence(self) -> List[str]:
        """All actions in round order."""
        return [self._snapshots[r]["action"] for r in sorted(self._snapshots.keys())]

    # Aliases for treeflow compatibility
    @property
    def state_results(self) -> List[StateResult]:
        """Alias for stateResults_sequence (treeflow compatibility)."""
        return self.stateResults_sequence

    @property
    def state_sequence(self) -> List[str]:
        """Alias for action_sequence (treeflow compatibility)."""
        return self.action_sequence

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

    def get_snapshot(self, round_idx: int) -> Optional[dict]:
        """Get snapshot for a specific round.

        Args:
            round_idx: Round index to retrieve.

        Returns:
            Snapshot dict or None if not found.
        """
        return self._snapshots.get(round_idx)

    def get_round_for_step(self, step_idx: int) -> Optional[int]:
        """Get round index for a given step.

        Args:
            step_idx: Step index to look up.

        Returns:
            Round index or None if not found.
        """
        return self._step_to_round.get(step_idx)

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

    Tree-flow properties (for XML-based prompts):
    - format_template: XML format template string.
    - tree_description: Short description for tree prompts.
    - guidelines: Usage guidelines list.
    - with_param: Whether action requires parameter re-prompting in focused mode.
    - parse: Static method to extract action data from XML response.

    Attributes:
        name: Action name (set by @register_action decorator).
    """

    name: str = ""  # Set by @register_action decorator

    # =========================================================================
    # Tool-flow interface (function calling)
    # =========================================================================

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

    # =========================================================================
    # Tree-flow properties (XML-based prompts)
    # =========================================================================

    format_template: str = ""
    """XML format template for this action (used in tree prompts)."""

    tree_description: str = ""
    """Short description for tree-flow prompt building."""

    guidelines: List[str] = []
    """Usage guidelines for this action."""

    with_param: bool = False
    """Whether action requires parameter re-prompting in focused mode."""

    @staticmethod
    def parse(text: str) -> dict:
        """Parse XML response to extract action-specific data.

        Args:
            text: Response text containing XML tags.

        Returns:
            Dict with extracted action parameters.
        """
        return {}


"""
ACTION CLASSES
"""

@register_action("execute", supported=True)
class ExecuteAction(ActionClass):
    """Execute a subtask using a worker agent."""

    # Tree-flow properties
    format_template = """<action>execute</action>
<task>Self-contained subtask with necessary context</task>"""

    tree_description = "Execute - work on the current task"

    guidelines = [
        "[execute] In <task>, include all and only the context the subagent needs; assume it cannot see any prior conversation.",
        "[execute] Include what was already found; focus only on the remaining information needed.",
    ]

    with_param = True

    @staticmethod
    def parse(text: str) -> dict:
        """Extract task from <task>...</task> format."""
        match = re.search(r"<task>(.*?)</task>", text, re.DOTALL)
        return {"task": match.group(1).strip() if match else ""}

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
        """Execute - work on the current task.

        This function uses another agent to work on the current task, with addtional tools, such as internet search.

        Guidelines:
        - In `task`, include all and only the context the subagent needs; assume it cannot see any prior conversation.
        - Include what was already found; focus only on the remaining information needed.

        Args:
            task: Self-contained subtask with necessary context.

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
            max_iteration=max_iteration,
            summarize_threshold=None
        )

        if num_fewshot > 0:
            fewshot = FewShotManager()
            fewshot.add_to_agent_memory(worker_agent, n=num_fewshot, selection="first", add_separator=True)

        if tracker is not None:
            tracker.register_tools(llm_id=step_idx + 1, tools=worker_tools)

        try:
            response = worker_agent.step(task)
            response_text = response.msg.content
            tool_msgs_id = [i for i, msg in enumerate(worker_agent.chat_history) if msg['role'] == 'tool']

            # Check completion status
            check_prompt = EXECUTE_CHECK_PROMPT.format(task=task)
            check_response = worker_agent.step(check_prompt)
            completion_status, completion_note = ExecuteAction._parse_status(check_response.msg.content)
        except Exception as e:
            error_str = str(e)
            # Check if it's a context length error
            if "too long" in error_str.lower() or "context length" in error_str.lower() or "maximum context" in error_str.lower():
                completion_status = "fail"
                completion_note = (
                    "The task generates too much context, which is too long for the subagent to process. "
                    "Please break this task down into smaller, more focused subtasks. "
                    "Each subtask should be self-contained and require less steps."
                )
                response_text = (
                    "The subtask is too complicated to manage within one subagent execution. "
                    "Please use the 'plan' action to break down this task into smaller, more manageable subtasks."
                )
                tool_msgs_id = []
            else:
                # Re-raise other exceptions
                raise

        feedback = f"| {completion_status.capitalize()} | Sub-round {len(tool_msgs_id)} : {response_text}"
        if completion_status != "success" and completion_note:
            feedback += f"\n{completion_note}"

        # Get chat history and tools used (may be empty if error occurred early)
        try:
            chat_history = context_records_to_memory_records(worker_agent.memory.retrieve())
        except:
            chat_history = []

        # Extract tool names
        tools_used = []
        if hasattr(worker_agent, 'chat_history'):
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

    # Tree-flow properties
    format_template = """<action>plan</action>
<tasks>
<task>Revised subtask 1</task>
<task>Revised subtask 2</task>
</tasks>"""

    tree_description = "Plan tasks - replace current and pending tasks with a new task list"

    guidelines = [
        "[plan] Replaces all current and pending tasks with new tasks. Finished tasks are preserved.",
        "[plan] You may split one subtask into multiple smaller subtasks when helpful.",
        "[plan] Each subtask must be narrowly scoped, achievable within a few searches, and have a clear desired result.",
        "[plan] For failed/partial tasks, change the approach rather than repeating the same task.",
    ]

    with_param = True

    @staticmethod
    def parse(text: str) -> dict:
        """Extract tasks from <tasks><task>...</task></tasks> format."""
        matches = re.findall(r"<task>(.*?)</task>", text, re.DOTALL)
        return {"tasks": [m.strip() for m in matches if m.strip()]}

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for plan action."""
        return "Planning tasks"

    @staticmethod
    def interface(tasks: List[str]) -> dict:
        """Plan tasks - replaces all current and pending tasks with a list of new tasks. Finished tasks are preserved.

        Guidelines:
        - You may split one task into multiple smaller subtasks when helpful.
        - Each task must be narrowly scoped, achievable within a few searches, and have a clear desired result.
        - For failed/partial tasks, change the approach rather than repeating the same task.

        Args:
            tasks: List of revised task descriptions.

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
    """Provide final answer when sufficient information is gathered."""

    # Tree-flow properties
    format_template = """<action>answer</action>
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>"""

    tree_description = "Answer - final response when you have enough verified information"

    guidelines = [
        "[answer] Use this action only when you have enough verified information to respond conclusively.",
        "[answer] Keep the justification to 1-2 short sentences.",
        '[answer] Put the final answer span in the "answer" field and the brief rationale in "justification".',
    ]

    with_param = True

    @staticmethod
    def parse(text: str) -> dict:
        """Extract answer from <answer>...</answer> format."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"answer": content}
        return {}

    @staticmethod
    def interface(answer: str, justification: str = "") -> dict:
        """Answer - final response when you have enough verified information.

        Args:
            answer: Final answer span.
            justification: 1-2 short sentences of rationale.

        Returns:
            Dict with answer and justification echoed back.
        """
        return {"answer": answer, "justification": justification}

    @staticmethod
    def do(answer: str, justification: str = "") -> StateResult:
        """Return the final answer.

        Args:
            answer: The final answer.
            justification: Brief rationale.

        Returns:
            StateResult with answer.
        """
        payload = {"answer": answer, "justification": justification}
        payload_str = json.dumps(payload, ensure_ascii=False)
        return StateResult(
            feedback=answer,
            records=[
                {"role": "user", "content": "Based on the research above, answer the question."},
                {"role": "assistant", "content": payload_str},
            ],
            state="answer",
            kwargs=payload,
        )



@register_action("continue", supported=True)
class ContinueAction(ActionClass):
    """Continue without tool call - simply feed back the last response."""

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for continue action."""
        return "Continuing..."

    @staticmethod
    def interface() -> dict:
        """Continue - placeholder interface (not called by LLM)."""
        return {}

    @staticmethod
    def do(recent_messages: List[Dict[str, str]]) -> StateResult:
        """Continue without tool call.

        Args:
            recent_messages: The last two messages (user prompt and assistant response).

        Returns:
            StateResult with feedback and records.
        """
        last_response = recent_messages[-1].get("content", "") if recent_messages else ""
        return StateResult(
            feedback=last_response,
            records=recent_messages,
            state="continue",
        )

    @staticmethod
    def update(ctx: "ContextData", result: StateResult) -> None:
        """Update context after continue action.

        Only extends history records, does not increment step.
        """
        ctx.history_records.extend(result.records)


@register_action("break", supported=True)
class BreakAction(ActionClass):
    """Break out of the main loop when no actions are available."""

    @staticmethod
    def display(data: dict) -> str:
        """Return status description for break action."""
        return "Breaking - no actions available"

    @staticmethod
    def interface() -> dict:
        """Break - exit the workflow loop."""
        return {}

    @staticmethod
    def do() -> StateResult:
        """Break out of the loop.

        Returns:
            StateResult indicating break.
        """
        return StateResult(
            feedback="No actions available, breaking out of loop",
            records=[],
            state="break",
        )

    @staticmethod
    def update(ctx: "ContextData", result: StateResult) -> None:
        """No context update needed for break."""
        pass


@register_action("think", supported=True)
class ThinkAction(ActionClass):
    """Reflect on current status."""

    # Tree-flow properties
    format_template = """<action>think</action>"""

    tree_description = "Think - pause to reflect and get a concise assessment before choosing next action"

    guidelines = [
        "[think] Use this action when the next operation is unclear and you need a brief assessment before proceeding.",
    ]

    with_param = False

    @staticmethod
    def parse(text: str) -> dict:
        """Think action has no parameters to parse."""
        return {}

    @staticmethod
    def _parse_thought(text: str) -> Optional[str]:
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
        """Think - pause to reflect and get a concise assessment before choosing next action.

        Use this action when the next operation is unclear and you need a brief assessment before proceeding.

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
            model=think_model,
            summarize_threshold=None
        )
        context_records = main_agent.memory.retrieve()[1:]
        forward_records = context_records_to_memory_records(context_records)
        think_agent.memory.write_records(forward_records)

        response = think_agent.step(THINK_PROMPT.format(question=question))
        response.msg.content
        record_interaction(tracker, think_agent.chat_history, llm_id=-2)

        thought = ThinkAction._parse_thought(response.msg.content) or response.msg.content
        return StateResult(
            feedback=thought,
            records=[
                {"role": "user", "content": "Think about next steps"},
                {"role": "assistant", "content": thought},
            ],
            state="think",
            chat_history=context_records_to_memory_records(think_agent.memory.retrieve()),
        )
