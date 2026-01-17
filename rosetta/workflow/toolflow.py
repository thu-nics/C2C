"""Tool-based research workflow with function calling.

Uses external tools for action selection - the main agent calls a tool to indicate
which action to take, and the tool arguments contain the action parameters.

Supported actions: execute, plan, think, answer, continue, break
Extended actions: parallel_execute, rewind, exam
"""

from typing import Tuple, Optional, List, Dict
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit

from rosetta.workflow.tool_prompt import build_task_state_prompt, MAIN_AGENT_SYSTEM_MESSAGE, FORCE_ANSWER_PROMPT, AnswerFormat
from rosetta.workflow.actions import (
    ACTIONS,
    ActionClass,
    get_action_tools,
    StateResult,
    ExecuteAction,
    PlanAction,
    ThinkAction,
    AnswerAction,
    ContinueAction,
    BreakAction,
    ContextData,
)
from rosetta.workflow.ext_actions import (
    RewindAction,
    ExamAction,
    ParallelExecuteAction,
)
from rosetta.workflow.track import InteractionTracker, record_interaction, TreeTracker
from rosetta.workflow.display import StatusLogger
from rosetta.workflow.camel_utils import messages_to_memoryRecords, add_tool_requests_to_chat_history
from rosetta.workflow.rules import ActionRuleEnforcer, DEFAULT_ENFORCER

def main_decision(
    state: str,
    main_model: BaseModelBackend,
    ctx: ContextData,
    question: str,
    state_rule_actions: Optional[List[str]] = None,
    action_rule_enforcer: ActionRuleEnforcer = DEFAULT_ENFORCER,
    tracker: InteractionTracker = None,
) -> Tuple[str, dict, List[Dict[str, str]], ChatAgent]:
    """Master decision state: prompt main agent and determine next state.

    This is the central hub of the state machine. After each state handler completes,
    control returns here to decide the next transition.

    Uses external tools for action selection - the main agent calls a tool to indicate
    which action to take, and the tool arguments contain the action parameters.

    Args:
        state: Current workflow state.
        main_model: Model for main decision agent.
        ctx: FlowContext containing tasks and history.
        question: Research question.
        state_rule_actions: Full list of allowed action names.
        action_rule_enforcer: Enforcer that filters actions based on rules.
        tracker: Interaction tracker.

    Returns:
        Tuple of (next_state, data, conversation, main_agent).
    """
    # Get allowed actions based on current state
    available_actions = action_rule_enforcer.update_actions(state, ctx.stateResults_sequence, state_rule_actions)
    action_tools = get_action_tools(available_actions)

    # Create main agent with external tools
    main_agent = ChatAgent(
        system_message=MAIN_AGENT_SYSTEM_MESSAGE,
        model=main_model,
        external_tools=action_tools,
        summarize_threshold=None,
    )

    # Sync history from previous rounds
    if ctx.history_records:
        main_agent.memory.write_records(messages_to_memoryRecords(ctx.history_records))

    # Return break if no actions available
    if not action_tools:
        return "break", {}, [], main_agent

    # Build task state prompt
    prompt = build_task_state_prompt(ctx.pending, ctx.current, ctx.finished, question=question)
    response = main_agent.step(prompt)
    response.msg.content  # call to ensure lazy consumption of the response
    chat_history = main_agent.chat_history
    
    # Check for external tool request
    tool_requests = response.info.get("external_tool_call_requests", [])
    
    # Add tool_calls to the last assistant message before recording
    if tool_requests and len(chat_history) > 0:
        chat_history = add_tool_requests_to_chat_history(chat_history, tool_requests[0])
    
    if tracker is not None:
        record_interaction(tracker, chat_history, llm_id=0)
        tracker.register_tools(llm_id=0, tools=action_tools)
    
    if not tool_requests:
        # No tool called - return continue state with recent messages
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.msg.content},
        ]
        return "continue", {"recent_messages": conversation}, conversation, main_agent

    request = tool_requests[0]
    next_state = request.tool_name
    data = request.args or {}

    # Build conversation records for history
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.msg.content},
    ]

    return next_state, data, conversation, main_agent


def do_tool_research(
    question: str,
    main_model: BaseModelBackend,
    worker_model: BaseModelBackend = None,
    rewind_model: BaseModelBackend = None,
    exam_model: BaseModelBackend = None,
    think_model: BaseModelBackend = None,
    state_rule_actions: Optional[List[str]] = ["execute", "parallel_execute", "plan", "think", "rewind", "answer", "exam"],
    action_rule_enforcer: ActionRuleEnforcer = DEFAULT_ENFORCER,
    tracker: InteractionTracker = None,
    tree_tracker: TreeTracker = None,
    worker_tools: List[FunctionTool] = None,
    max_rounds: int = 10,
    show_status: bool = True,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Tool-based research with external tool-based action selection.

    The main agent selects actions via external tools. Tool execution is handled
    externally in this function, not by the tools themselves.

    Supported actions: execute, plan, think, answer, continue, break
    Extended actions: parallel_execute, rewind, exam

    Args:
        question: The question to answer.
        main_model: Model for main decision agent.
        worker_model: Model for worker agent. Defaults to main_model.
        rewind_model: Model for rewind agent. Defaults to worker_model.
        exam_model: Model for exam agent. Defaults to worker_model.
        think_model: Model for think agent. Defaults to worker_model.
        state_rule_actions: List of allowed action names. Defaults to supported actions.
        action_rule_enforcer: Enforcer that filters actions based on rules. Defaults to DEFAULT_ENFORCER.
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
    if worker_model is None:
        worker_model = main_model
    if rewind_model is None:
        rewind_model = worker_model
    if exam_model is None:
        exam_model = worker_model
    if think_model is None:
        think_model = worker_model

    ctx = ContextData()
    state = "initial"

    for _ in range(max_rounds):
        # Master decision: determine next state
        next_state, data, conversation, main_agent = main_decision(
            state, main_model, ctx, question, state_rule_actions, action_rule_enforcer, tracker
        )


        if tree_tracker is not None:
            parent_id = tree_tracker.get_current_parent()
            tree_tracker.add_node(ctx.round, parent_id, next_state, data)

        # Get status description from action class
        action_class: ActionClass = ACTIONS.get(next_state)
        if action_class is None:
            raise ValueError(f"Unknown or unsupported action: {next_state}")
        status_desc = action_class.display(data)

        with logger.status(next_state, ctx.round, ctx.step, status_desc, ctx.finished, ctx.current, ctx.pending) as update_status:
            # Dispatch to state handler (external execution via action classes)
            if next_state == "execute":
                if tree_tracker is not None:
                    tree_tracker.register_step(ctx.step, ctx.round)
                task = data.get("task", "")
                result = ExecuteAction.do(task, worker_model, worker_tools, tracker, ctx.step, num_fewshot=0)

            elif next_state == "parallel_execute":
                raise NotImplementedError("Parallel execute is not supported")
                if tree_tracker is not None:
                    tree_tracker.register_step(ctx.step, ctx.round)
                tasks = data.get("tasks", [])
                result = ParallelExecuteAction.do(
                    tasks, worker_model, worker_tools, tracker, ctx.step, num_fewshot=0
                )

            elif next_state == "plan":
                tasks = data.get("tasks", [])
                result = PlanAction.do(tasks, conversation, main_agent, question)

            elif next_state == "think":
                result = ThinkAction.do(think_model, main_agent, question, tracker)

            elif next_state == "rewind":
                result = RewindAction.do(rewind_model, ctx.history_records)

            elif next_state == "exam":
                result = ExamAction.do(
                    exam_model, main_agent, ctx.stateResults_sequence, data.get("step", 0), question, tracker
                )
            elif next_state == "answer":
                answer = data.get("answer", "")
                justification = data.get("justification", "")
                result = AnswerAction.do(answer, justification)

            elif next_state == "break":
                result = BreakAction.do()

            elif next_state == "continue":
                recent_messages = data.get("recent_messages", [])
                result = ContinueAction.do(recent_messages)
            else:
                break  # Unknown state

            # Update context via action's update method
            action_class.update(ctx, result)

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

        # Exit if answer state
        if next_state == "answer":
            return result.feedback, tracker
        elif next_state == "break":
            break

    # Fallback: answer directly using chat history
    main_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=main_model
    )
    main_agent.memory.write_records(messages_to_memoryRecords(ctx.history_records))

    prompt = FORCE_ANSWER_PROMPT.format(question=question)
    response = main_agent.step(prompt, response_format=AnswerFormat)
    record_interaction(tracker, main_agent.chat_history, llm_id=0)

    answer = response.msg.content

    if tracker is not None:
        tracker.state_sequence = ctx.action_sequence + ["forceAnswer"]
        tracker.state_results = ctx.stateResults_sequence
    return answer, tracker

