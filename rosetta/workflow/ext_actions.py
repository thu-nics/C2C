"""Extended action classes for workflow orchestration.

These actions extend the base actions with more complex execution patterns:
- parallel_execute: Run multiple tasks concurrently
- rewind: Backtrack to a previous state
- exam: Verify a previous step's result
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool
from camel.agents import ChatAgent

from rosetta.workflow.actions import (
    register_action,
    ActionClass,
    StateResult,
    ContextData,
    ExecuteAction,
)
from rosetta.workflow.camel_utils import context_records_to_memory_records
from rosetta.workflow.track import InteractionTracker, record_interaction
from rosetta.workflow.tree_prompt import REWIND_PROMPT, EXAM_PROMPT


# =============================================================================
# PARALLEL EXECUTE ACTION
# =============================================================================


@register_action("parallel_execute", supported=True)
class ParallelExecuteAction(ActionClass):
    """Run multiple tasks in parallel using worker agents."""

    # Tree-flow properties
    format_template = """<action>parallel_execute</action>
<tasks>
<task>Self-contained subtask 1 with necessary context</task>
<task>Self-contained subtask 2 with necessary context</task>
</tasks>"""

    tree_description = "Parallel Execute - work on multiple independent tasks simultaneously"

    guidelines = [
        "[parallel_execute] Use when multiple tasks are independent and can be executed concurrently.",
        "[parallel_execute] Each <task> must be self-contained with all context needed; subagents cannot see prior conversation.",
        "[parallel_execute] Results will be collected and presented as multi-round conversation records.",
    ]

    with_param = True

    @staticmethod
    def parse(text: str) -> dict:
        """Extract tasks from <tasks><task>...</task></tasks> format."""
        matches = re.findall(r"<task>(.*?)</task>", text, re.DOTALL)
        return {"tasks": [m.strip() for m in matches if m.strip()]}

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
    def do(
        tasks: List[str],
        worker_model: BaseModelBackend,
        worker_tools: List[FunctionTool],
        tracker: InteractionTracker = None,
        step_idx: int = 0,
        max_iteration: Optional[int] = None,
        num_fewshot: int = 0,
    ) -> StateResult:
        """Execute multiple tasks in parallel using worker agents.

        Args:
            tasks: List of subtasks to execute in parallel.
            worker_model: Model for worker agents.
            worker_tools: Tools available to workers.
            tracker: Interaction tracker.
            step_idx: Current step index.
            max_iteration: Maximum number of iterations per task.
            num_fewshot: Number of few-shot examples to add.

        Returns:
            StateResult with aggregated feedback, records, and results.
        """
        results: List[Tuple[int, StateResult]] = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(
                    ExecuteAction.do,
                    task,
                    worker_model,
                    worker_tools,
                    tracker,
                    step_idx + idx,
                    max_iteration,
                    num_fewshot,
                ): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))

        # Sort by original task index to maintain order
        results.sort(key=lambda x: x[0])

        # Aggregate results
        records = []
        all_chat_histories = []
        all_tools_used = []
        total_tool_calls = 0
        feedbacks = []
        completion_statuses = []

        for task_idx, result in results:
            records.extend(result.records)
            all_chat_histories.append(result.chat_history)
            all_tools_used.extend(result.kwargs.get("tools_used", []))
            total_tool_calls += result.kwargs.get("num_tool_calls", 0)
            status = result.kwargs.get("completion_status", "success")
            completion_statuses.append(status)
            feedbacks.append(f"[Task {task_idx + 1}] {result.feedback}")

        # Determine overall status
        if all(s == "success" for s in completion_statuses):
            overall_status = "success"
        elif all(s == "fail" for s in completion_statuses):
            overall_status = "fail"
        else:
            overall_status = "partial"

        feedback = f"| Parallel {overall_status.capitalize()} | {len(tasks)} tasks, {total_tool_calls} tool calls\n" + "\n".join(feedbacks)

        return StateResult(
            feedback=feedback,
            records=records,
            state="parallel_execute",
            chat_history=all_chat_histories[0] if all_chat_histories else None,
            kwargs={
                "num_tool_calls": total_tool_calls,
                "completion_status": overall_status,
                "tools_used": list(dict.fromkeys(all_tools_used)),
                "all_chat_histories": all_chat_histories,
                "task_count": len(tasks),
            },
        )

    @staticmethod
    def update(ctx: "ContextData", result: StateResult) -> None:
        """Update context after parallel execute action.

        On success, moves current task to finished.
        """
        ctx.step += 1
        ctx.history_records.extend(result.records)
        status = result.kwargs.get("completion_status", "success")
        if status == "success" and ctx.current:
            ctx.finished.append(ctx.current.pop(0))
        ctx.promote_task()


# =============================================================================
# REWIND ACTION
# =============================================================================


@register_action("rewind", supported=True)
class RewindAction(ActionClass):
    """Backtrack to a previous state when the current approach isn't working."""

    # Tree-flow properties
    format_template = """<action>rewind</action>"""

    tree_description = "Rewind - backtrack when similar tasks fail repeatedly"

    guidelines = [
        "[rewind] Use when similar tasks fail repeatedly (similar query variations, similar dead ends).",
        "[rewind] Use when exploring the wrong entity or topic.",
        "[rewind] After rewinding, switch to a different approach.",
    ]

    with_param = False

    @staticmethod
    def parse(text: str) -> dict:
        """Rewind action has no parameters to parse."""
        return {}

    @staticmethod
    def _parse_rewind_response(text: str) -> Tuple[Optional[int], Optional[str]]:
        """Extract rewind index and summary from rewind agent response.

        Args:
            text: Response text containing <rewind_to> and <summary> tags.

        Returns:
            Tuple of (rewind_to_index, summary).
        """
        idx_match = re.search(r"<rewind_to>(\d+)</rewind_to>", text)
        sum_match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        idx = int(idx_match.group(1)) if idx_match else None
        summary = sum_match.group(1).strip() if sum_match else None
        return idx, summary

    @staticmethod
    def format_history_numbered(messages: List[dict]) -> str:
        """Format chat history messages as numbered turns for rewind agent.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Formatted string with numbered turns.
        """
        lines = []
        turn = 1  # 1-indexed turns
        start_idx = 1 if messages and messages[0].get('role') == 'system' else 0
        for i in range(start_idx, len(messages), 2):
            subtask = messages[i].get('content', '') if i < len(messages) else ""
            feedback = messages[i + 1].get('content', '') if i + 1 < len(messages) else ""
            lines.append(f"Turn {turn}:\n[Subtask] {subtask}\n[Feedback] {feedback}")
            turn += 1
        return "\n\n".join(lines)

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
        """Analyze history and determine rewind point.

        Args:
            rewind_model: Model for rewind agent.
            messages: Chat history messages (standard format).

        Returns:
            StateResult with rewind_to (turn index, 1-indexed) and summary in kwargs.
        """
        history_str = RewindAction.format_history_numbered(messages)
        rewind_agent = ChatAgent(
            system_message="You analyze research history to find rewind points.",
            model=rewind_model,
            summarize_threshold=None,
        )
        response = rewind_agent.step(REWIND_PROMPT.format(history=history_str))
        rewind_turn, summary = RewindAction._parse_rewind_response(response.msg.content)

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


# =============================================================================
# EXAM ACTION
# =============================================================================


@register_action("exam", supported=True)
class ExamAction(ActionClass):
    """Verify a previous step's result for correctness."""

    # Tree-flow properties
    format_template = """<action>exam</action>
<step>step_index to examine (1-indexed)</step>"""

    tree_description = "Exam - verify a previous step's result if it seems suspicious or critical"

    guidelines = [
        "[exam] Use this action to verify a specific prior step when its result is suspicious, inconsistent, or high-impact.",
        "[exam] Provide the 1-indexed <step> you want to re-check.",
    ]

    with_param = True

    @staticmethod
    def parse(text: str) -> dict:
        """Extract step index from <step>...</step> format."""
        match = re.search(r"<step>(\d+)</step>", text)
        return {"step": int(match.group(1)) if match else 0}

    @staticmethod
    def _parse_exam_result(text: str) -> Tuple[str, str, str]:
        """Extract verdict, reason, and correction from exam agent response.

        Args:
            text: Response text containing <verdict>, <reason>, <correction> tags.

        Returns:
            Tuple of (verdict, reason, correction).
        """
        verdict_match = re.search(r"<verdict>(.*?)</verdict>", text, re.DOTALL)
        reason_match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
        correction_match = re.search(r"<correction>(.*?)</correction>", text, re.DOTALL)
        verdict = verdict_match.group(1).strip().lower() if verdict_match else "unknown"
        reason = reason_match.group(1).strip() if reason_match else ""
        correction = correction_match.group(1).strip() if correction_match else ""
        return verdict, reason, correction

    @staticmethod
    def format_memory_records(records: Optional[List[Any]]) -> str:
        """Format MemoryRecord list into a readable chat history string.

        Args:
            records: List of MemoryRecord objects.

        Returns:
            Formatted string representation.
        """
        if not records:
            return "(no chat history available)"
        lines = []
        for idx, record in enumerate(records, 1):
            role = getattr(record, "role_at_backend", None)
            role_name = role.name.lower() if role is not None else "unknown"
            message = getattr(record, "message", None)
            content = getattr(message, "content", "") if message is not None else ""
            if not content and message is not None:
                # Try to format tool calls
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    content = f"[tool_calls] {tool_calls}"
                else:
                    func_name = getattr(message, "func_name", None)
                    args = getattr(message, "args", None)
                    result = getattr(message, "result", None)
                    if func_name or args is not None or result is not None:
                        parts = []
                        if func_name:
                            parts.append(f"name={func_name}")
                        if args is not None:
                            parts.append(f"args={args}")
                        if result is not None:
                            parts.append(f"result={result}")
                        content = "[tool_call] " + " ".join(parts)
            if not content:
                content = "(empty)"
            lines.append(f"[{idx}] {role_name}:\n{content}")
        return "\n\n".join(lines)

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
    def do(
        exam_model: BaseModelBackend,
        main_agent: ChatAgent,
        state_results: List[StateResult],
        step_idx: int,
        question: str,
        tracker: InteractionTracker = None,
    ) -> StateResult:
        """Examine a step's result for correctness.

        Args:
            exam_model: Model for exam agent.
            main_agent: Main agent whose history is provided as context.
            state_results: Sequence of StateResults for locating the step's subagent history.
            step_idx: Step index to examine (0-indexed).
            question: Original question.
            tracker: Interaction tracker.

        Returns:
            StateResult with exam verdict and records.
        """
        exam_agent = ChatAgent(
            system_message="You are a verification agent that checks for errors in research results.",
            model=exam_model,
            summarize_threshold=None,
        )
        main_context_records = main_agent.memory.retrieve()[1:-2]
        exam_agent.memory.write_records(context_records_to_memory_records(main_context_records))

        target_result = state_results[step_idx] if 0 <= step_idx < len(state_results) else None
        step_state = target_result.state if target_result is not None else "unknown"
        subagent_history = ExamAction.format_memory_records(
            target_result.chat_history if target_result is not None else None
        )

        prompt = EXAM_PROMPT.format(
            question=question,
            step_idx=step_idx,
            step_state=step_state,
            subagent_history=subagent_history,
        )
        response = exam_agent.step(prompt)
        response.msg.content  # Ensure lazy consumption
        record_interaction(tracker, exam_agent.chat_history, llm_id=-1)

        verdict, reason, correction = ExamAction._parse_exam_result(response.msg.content)

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
            state="exam",
            chat_history=context_records_to_memory_records(exam_agent.memory.retrieve()),
            kwargs={"verdict": verdict, "reason": reason, "correction": correction},
        )
