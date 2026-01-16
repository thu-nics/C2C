from rosetta.workflow.actions import register_action, ActionClass, StateResult, ContextData
from typing import List
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool
import functools

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
        """Parallel Execute - work on multiple independent tasks simultaneously.

        Use when multiple tasks are independent and can be executed concurrently.
        Results will be collected and presented as multi-round conversation records.

        Guidelines:
        - Each task must be self-contained with all context needed; subagents cannot see prior conversation.

        Args:
            tasks: List of self-contained subtasks with necessary context.

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
        """Rewind - backtrack when similar tasks fail repeatedly.

        Guidelines:
        - Use when similar tasks fail repeatedly (similar query variations, similar dead ends).
        - Use when exploring the wrong entity or topic.
        - After rewinding, switch to a different approach.

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
        """Exam - verify a previous step's result if it seems suspicious or critical.

        Use this action to verify a specific prior step when its result is suspicious, inconsistent, or high-impact.

        Guidelines:
        - Provide the 1-indexed step you want to re-check.

        Args:
            step: Step index to examine (1-indexed).

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