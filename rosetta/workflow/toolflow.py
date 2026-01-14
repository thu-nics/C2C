"""Tree-based research workflow with rewind capability.

Currently supported actions: execute, plan, think, answer
Unsupported actions (raise NotImplementedError): parallel_execute, rewind, exam
"""

from typing import Tuple, Optional, List, Any, Dict
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit

from rosetta.workflow.tool_prompt import build_task_state_prompt, MAIN_AGENT_SYSTEM_MESSAGE, FORCE_ANSWER_PROMPT, AnswerFormat
from rosetta.workflow.actions import (
    ACTIONS,
    ActionClass,
    get_action_tools,
    is_action_supported,
    StateResult,
    ExecuteAction,
    PlanAction,
    ThinkAction,
    AnswerAction,
    RewindAction,
    ExamAction,
    ParallelExecuteAction,
    ContextData,
)
from rosetta.workflow.track import InteractionTracker, record_interaction, TreeTracker
from rosetta.workflow.display import StatusLogger
from rosetta.workflow.camel_utils import messages_to_memoryRecords, add_tool_requests_to_chat_history

def state_rule(
    _current_state: str,
    state_sequence: Optional[List[StateResult]] = None,
    full_state_rule_actions: Optional[List[str]] = None,
) -> List[str]:
    """Define available next states based on current state.

    Currently restricts the initial state to plan/think, otherwise returns all actions.
    Override this function to implement custom transitions.

    Args:
        _current_state: Current workflow state.
        state_sequence: Full sequence of prior StateResult objects (unused for now).
        full_state_rule_actions: Full action list to filter from.

    Returns:
        List of available action names for next transition.
    """
    actions = full_state_rule_actions
    if _current_state == "initial":
        # Always think or plan first
        filtered = [action for action in actions if action in ("plan", "think")]
        return filtered if filtered else list(actions)
    else:
        pass
        # If not initial, don't think
        # _current_state = [action for action in actions if action not in ("think")]
        # actions = _current_state

    if not state_sequence or not any(result.state == "plan" for result in state_sequence):
        # If no plan, don't execute or parallel_execute
        actions = [action for action in actions if action not in ("execute", "parallel_execute")]
    if state_sequence:
        last_stateResult = state_sequence[-1]
        if last_stateResult.state == "rewind":
            # If rewind, must plan again
            filtered = [action for action in actions if action in ("plan")]
            return filtered if filtered else list(actions)

        # if last_stateResult.state == "execute":
        #     # If execute, must plan again
        #     filtered = [action for action in actions if action in ("plan")]
        #     return filtered if filtered else list(actions)

        # Count consecutive execute failures (fail/partial) without success
        consecutive_failures = 0
        for result in reversed(state_sequence):
            if result.state in ("execute", "parallel_execute"):
                status = result.kwargs.get("completion_status", "success")
                if status in ("fail", "partial"):
                    consecutive_failures += 1
                else:  # success resets the count
                    break
        # if consecutive_failures >= 4 and "rewind" in actions:
        #     return ["rewind"]

    return actions

def main_decision(
    state: str,
    main_model: BaseModelBackend,
    ctx: ContextData,
    question: str,
    state_rule_actions: Optional[List[str]] = None,
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
        tracker: Interaction tracker.

    Returns:
        Tuple of (next_state, data, conversation, main_agent).
    """
    # Get allowed actions based on current state
    available_actions = state_rule(state, ctx.stateResults_sequence, state_rule_actions)
    action_tools = get_action_tools(available_actions)

    # Create main agent with external tools
    main_agent = ChatAgent(
        system_message=MAIN_AGENT_SYSTEM_MESSAGE,
        model=main_model,
        external_tools=action_tools,
    )

    # Sync history from previous rounds
    if ctx.history_records:
        main_agent.memory.write_records(messages_to_memoryRecords(ctx.history_records))

    # Build task state prompt
    prompt = build_task_state_prompt(ctx.pending, ctx.current, ctx.finished, question=question)
    response = main_agent.step(prompt)
    response.info  # call to ensure lazy consumption of the response
    
    chat_history = main_agent.chat_history
    
    # Check for external tool request
    tool_requests = response.info.get("external_tool_call_requests", [])
    
    # Add tool_calls to the last assistant message before recording
    if tool_requests and len(chat_history) > 0:
        chat_history = add_tool_requests_to_chat_history(chat_history, tool_requests[0])
    
    record_interaction(tracker, chat_history, llm_id=0)
    
    if not tool_requests:
        # No tool called - return unknown state
        return "unknown", {}, [], main_agent

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
    tracker: InteractionTracker = None,
    tree_tracker: TreeTracker = None,
    worker_tools: List[FunctionTool] = None,
    max_rounds: int = 10,
    show_status: bool = True,
) -> Tuple[str, Optional[InteractionTracker]]:
    """Tree-based research with external tool-based action selection.

    The main agent selects actions via external tools. Tool execution is handled
    externally in this function, not by the tools themselves.

    Currently supported actions: execute, plan, think, answer

    Args:
        question: The question to answer.
        main_model: Model for main decision agent.
        worker_model: Model for worker agent. Defaults to main_model.
        rewind_model: Model for rewind agent. Defaults to worker_model.
        exam_model: Model for exam agent. Defaults to worker_model.
        think_model: Model for think agent. Defaults to worker_model.
        state_rule_actions: List of allowed action names. Defaults to supported actions.
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
            state, main_model, ctx, question, state_rule_actions, tracker
        )

        if next_state == "unknown":
            # No tool called - shouldn't happen, but handle that by entering answer state
            print("No tool called - entering answer state")
            print(main_agent.chat_history)
            break

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
                    # Handle answer - terminal state
            elif next_state == "answer":
                answer = data.get("answer", "")
                justification = data.get("justification", "")
                result = AnswerAction.do(answer, justification)
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
            return result.kwargs.get("answer", ""), tracker

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
        tracker.state_sequence = ctx.action_sequence + ["answer"]
        tracker.state_results = ctx.stateResults_sequence
    return answer, tracker

