"""Tree-based research workflow with XML-based action selection.

Uses XML-formatted prompts for action selection. The main agent outputs
<action>name</action> tags to indicate which action to take.

Supported actions: execute, plan, think, answer, continue, break
Extended actions: parallel_execute, rewind, exam

Action classes and utilities are imported from:
- actions.py: ExecuteAction, PlanAction, ThinkAction, AnswerAction, ContinueAction, BreakAction
- ext_actions.py: ParallelExecuteAction, RewindAction, ExamAction
- tree_prompt.py: build_action_prompt, build_decision_prompt, parse_decision
- rules.py: ActionRuleEnforcer, TREEFLOW_ENFORCER
"""

from typing import Tuple, Optional, List, Dict
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit

from rosetta.workflow.actions import (
    ACTIONS,
    StateResult,
    ContextData,
    ExecuteAction,
    PlanAction,
    ThinkAction,
    AnswerAction,
    ContinueAction,
    BreakAction,
)
from rosetta.workflow.ext_actions import (
    RewindAction,
    ExamAction,
    ParallelExecuteAction,
)
from rosetta.workflow.tree_prompt import (
    build_action_prompt,
    build_decision_prompt,
    parse_decision,
)
from rosetta.workflow.tool_prompt import FORCE_ANSWER_PROMPT, AnswerFormat
from rosetta.workflow.track import InteractionTracker, record_interaction, TreeTracker
from rosetta.workflow.display import StatusLogger
from rosetta.workflow.camel_utils import messages_to_memoryRecords
from rosetta.workflow.rules import ActionRuleEnforcer, TREEFLOW_ENFORCER


# =============================================================================
# TREEFLOW-SPECIFIC CONTEXT UPDATES
# =============================================================================


def _update_agent_history(main_agent: ChatAgent, ctx: ContextData, result: StateResult) -> None:
    """Update main agent memory based on state result.

    This is treeflow-specific as it operates directly on agent memory.
    For toolflow, the action's update() method modifies ctx.history_records.

    Args:
        main_agent: Main agent whose memory to update.
        ctx: Context data for snapshot lookup during rewind.
        result: The state result to process.
    """
    if result.state == "rewind":
        rewind_to_turn = result.kwargs.get("rewind_to_step", 0)
        summary = result.kwargs.get("summary", "")
        # Rollback history to rewind point
        history = main_agent.chat_history
        start_idx = 1 if history and history[0].get("role") == "system" else 0
        num_messages_to_keep = max(0, rewind_to_turn) * 2
        keep_end = start_idx + num_messages_to_keep
        keep_end = min(max(keep_end, start_idx), len(history))
        drop_count = len(history) - keep_end - 1
        if drop_count:
            main_agent.memory.pop_records(drop_count)
        if summary:
            main_agent.memory.write_records(
                messages_to_memoryRecords([{"role": "assistant", "content": summary}])
            )
    elif result.state in ("continue", "break"):
        # For continue/break, don't pop the original messages
        pass
    else:
        main_agent.memory.pop_records(2)
        if result.records:
            main_agent.memory.write_records(
                messages_to_memoryRecords(result.records)
            )


def _update_tasks(ctx: ContextData, result: StateResult) -> None:
    """Update task lists and step based on state result.

    This is treeflow-specific task update that handles all action types.

    Args:
        ctx: Context data to update.
        result: The state result to process.
    """
    if result.state == "rewind":
        rewind_to_turn = result.kwargs.get("rewind_to_step", 0)
        target_step = rewind_to_turn
        target_round = ctx.get_round_for_step(target_step)
        if target_round is not None:
            snapshot = ctx.get_snapshot(target_round) or {}
            ctx.pending = list(snapshot.get("pending", []))
            ctx.current = list(snapshot.get("current", []))
            ctx.finished = list(snapshot.get("finished", []))
            ctx.step = snapshot.get("step", target_step)
            ctx._step_to_round = {s: r for s, r in ctx._step_to_round.items() if s <= ctx.step}
    elif result.state in ("continue", "break"):
        # continue/break don't increment step
        pass
    else:
        ctx.step += 1
        if result.state == "execute":
            status = result.kwargs.get("completion_status", "success")
            if status == "success" and ctx.current:
                ctx.finished.append(ctx.current.pop(0))
        elif result.state == "parallel_execute":
            status = result.kwargs.get("completion_status", "success")
            if status == "success" and ctx.current:
                ctx.finished.append(ctx.current.pop(0))
        elif result.state == "plan":
            ctx.current = []
            ctx.pending = result.tasks if result.tasks else []

    # Promote first pending to current (except for continue/break)
    if result.state not in ("continue", "break"):
        if not ctx.current and ctx.pending:
            ctx.current.append(ctx.pending.pop(0))


# =============================================================================
# DECISION FUNCTIONS
# =============================================================================


def main_decision(
    state: str,
    main_agent: ChatAgent,
    ctx: ContextData,
    question: str,
    state_rule_actions: Optional[List[str]] = None,
    action_rule_enforcer: ActionRuleEnforcer = TREEFLOW_ENFORCER,
    tracker: InteractionTracker = None,
) -> Tuple[str, dict, List[Dict[str, str]]]:
    """Master decision state: prompt main agent and determine next state.

    This is the central hub of the state machine. After each state handler completes,
    control returns here to decide the next transition.

    Args:
        state: Current workflow state.
        main_agent: Main agent for decision making.
        ctx: Context data containing tasks and history.
        question: Research question.
        state_rule_actions: Full list of allowed action names.
        action_rule_enforcer: Enforcer that filters actions based on rules.
        tracker: Interaction tracker.

    Returns:
        Tuple of (next_state, data, conversation).
    """
    available_actions = action_rule_enforcer.update_actions(state, ctx.state_results, state_rule_actions)

    # Return break if no actions available
    if not available_actions:
        return "break", {}, []

    prompt = build_action_prompt(
        available_actions, question, ctx.pending, ctx.current, ctx.finished,
        single_action=len(available_actions) == 1
    )
    response = main_agent.step(prompt)
    response.msg.content  # Ensure lazy consumption
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    next_state, data = parse_decision(response.msg.content)
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.msg.content},
    ]

    # If parse_decision returns "unknown", treat as continue
    if next_state == "unknown":
        return "continue", {"recent_messages": conversation}, conversation

    return next_state, data, conversation


def focused_substep(
    main_agent: ChatAgent,
    action: str,
    question: str,
    ctx: ContextData,
    tracker: InteractionTracker = None,
) -> Tuple[dict, List[Dict[str, str]]]:
    """Get focused parameters for an action by re-stepping with single-action prompt.

    When an action requires parameters (with_param=True), this function pops the
    original decision prompt/response and re-steps with a focused single-action prompt
    to get the specific parameters.

    Args:
        main_agent: Main agent for decision making.
        action: The chosen action name.
        question: Research question.
        ctx: Context data containing tasks.
        tracker: Interaction tracker.

    Returns:
        Tuple of (parsed_data, conversation).
    """
    main_agent.memory.pop_records(2)
    prompt = build_action_prompt(
        [action], question, ctx.pending, ctx.current, ctx.finished, single_action=True
    )
    response = main_agent.step(prompt)
    response.msg.content  # Ensure lazy consumption
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    _, data = parse_decision(response.msg.content)
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.msg.content},
    ]
    return data, conversation


# =============================================================================
# MAIN RESEARCH FUNCTION
# =============================================================================


def do_tree_research(
    question: str,
    main_model: BaseModelBackend,
    worker_model: BaseModelBackend,
    rewind_model: BaseModelBackend = None,
    exam_model: BaseModelBackend = None,
    think_model: BaseModelBackend = None,
    state_rule_actions: Optional[List[str]] = ["execute", "parallel_execute", "plan", "think", "rewind", "answer", "exam"],
    action_rule_enforcer: ActionRuleEnforcer = TREEFLOW_ENFORCER,
    tracker: InteractionTracker = None,
    tree_tracker: TreeTracker = None,
    worker_tools: List[FunctionTool] = None,
    max_rounds: int = 10,
    show_status: bool = True,
    focused: bool = False,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Tree-based research with XML-based action selection.

    State Machine Structure:
        - main_decision: Central hub that prompts main agent for next action
        - execute: Execute subtask via worker agent (ExecuteAction.do)
        - parallel_execute: Execute multiple tasks concurrently (ParallelExecuteAction.do)
        - plan: Update task list (PlanAction.do)
        - think: Reflect on current status (ThinkAction.do)
        - rewind: Analyze history and backtrack (RewindAction.do)
        - exam: Verify a previous step's result (ExamAction.do)
        - answer: Terminal state, return result
        - continue: No valid action parsed, feed back the response
        - break: No actions available, exit loop

    Args:
        question: The question to answer.
        main_model: Model for main decision-making agent.
        worker_model: Model for worker agent.
        rewind_model: Model for rewind agent. Defaults to worker_model.
        exam_model: Model for exam agent. Defaults to worker_model.
        think_model: Model for think agent. Defaults to worker_model.
        state_rule_actions: List of allowed action names.
        action_rule_enforcer: Enforcer that filters actions based on rules.
        tracker: Interaction tracker.
        tree_tracker: Tree structure tracker.
        worker_tools: List of tools for worker. Defaults to [Google search].
        max_rounds: Maximum iterations.
        show_status: Show spinner status.
        focused: If True, use focused_substep for actions with with_param=True.

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
    if think_model is None:
        think_model = worker_model

    main_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=main_model,
        summarize_threshold=None,
    )

    ctx = ContextData(main_agent)
    state = "initial"

    for _ in range(max_rounds):
        # Master decision: determine next state
        next_state, data, conversation = main_decision(
            state, main_agent, ctx, question, state_rule_actions, action_rule_enforcer, tracker
        )

        # If focused mode and action requires parameters, re-prompt with single-action prompt
        if focused and next_state in ACTIONS and ACTIONS[next_state].with_param:
            data, conversation = focused_substep(main_agent, next_state, question, ctx, tracker)

        if tree_tracker is not None:
            parent_id = tree_tracker.get_current_parent()
            tree_tracker.add_node(ctx.round, parent_id, next_state, data)

        # Terminal state: answer
        if next_state == "answer":
            if tracker is not None:
                tracker.state_sequence = ctx.state_sequence + ["answer"]
                tracker.state_results = ctx.state_results
            return data.get("answer", ""), tracker

        # Get action class and status description
        action_cls = ACTIONS.get(next_state)
        if action_cls is None:
            break  # Unknown state
        status_desc = action_cls.display(data)

        with logger.status(next_state, ctx.round, ctx.step, status_desc, ctx.finished, ctx.current, ctx.pending) as update_status:
            # Dispatch to action class do() methods
            if next_state == "execute":
                if tree_tracker is not None:
                    tree_tracker.register_step(ctx.step, ctx.round)
                result = ExecuteAction.do(
                    data.get("task", ""), worker_model, worker_tools, tracker, ctx.step
                )

            elif next_state == "parallel_execute":
                if tree_tracker is not None:
                    tree_tracker.register_step(ctx.step, ctx.round)
                result = ParallelExecuteAction.do(
                    data.get("tasks", []), worker_model, worker_tools, tracker, ctx.step
                )

            elif next_state == "plan":
                result = PlanAction.do(data.get("tasks", []), conversation, main_agent, question)

            elif next_state == "think":
                result = ThinkAction.do(think_model, main_agent, question, tracker)

            elif next_state == "rewind":
                result = RewindAction.do(rewind_model, main_agent.chat_history[:-2])

            elif next_state == "exam":
                result = ExamAction.do(
                    exam_model, main_agent, ctx.state_results, data.get("step", 0), question, tracker
                )

            elif next_state == "continue":
                recent_messages = data.get("recent_messages", [])
                result = ContinueAction.do(recent_messages)

            elif next_state == "break":
                result = BreakAction.do()

            else:
                break  # Unknown state

            # Update main agent history and tasks
            _update_agent_history(main_agent, ctx, result)
            _update_tasks(ctx, result)

            # Snapshot AFTER action
            ctx.snapshot(next_state, result)

            # Update status display
            tools_used = result.kwargs.get("tools_used")
            update_status(result.feedback, tools_used=tools_used,
                          tasks=(ctx.finished, ctx.current, ctx.pending))

            # Record tools used for tree tracker
            if tree_tracker is not None:
                tree_tracker.record_tools_used(tools_used or [])
                if result.state == "rewind":
                    tree_tracker.mark_rewind(
                        result.kwargs.get("rewind_to_step", 0),
                        result.kwargs.get("summary", "")
                    )

        state = next_state

        # Exit if break state
        if next_state == "break":
            break

    # Fallback: force answer using structured output
    # Reuse existing main_agent (which has the conversation history)
    prompt = FORCE_ANSWER_PROMPT.format(question=question)
    response = main_agent.step(prompt, response_format=AnswerFormat)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    answer = response.msg.content

    if tracker is not None:
        tracker.state_sequence = ctx.state_sequence + ["forceAnswer"]
        tracker.state_results = ctx.state_results
    return answer, tracker
