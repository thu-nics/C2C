"""Prompts for tree-based research workflow.

Action properties (format_template, tree_description, guidelines, with_param, parse)
are now defined in the action classes themselves (actions.py, ext_actions.py).
Use build_decision_prompt() and build_action_prompt() to construct prompts dynamically.
"""

from typing import List

# Import action registry (populated by action class decorators)
from rosetta.workflow.actions import ACTIONS

# =============================================================================
# PROMPT BUILDING UTILITIES
# =============================================================================


def build_decision_prompt(
    action_names: List[str],
    question_var: str = "{question}",
    pending_var: str = "{pending_tasks}",
    current_var: str = "{current_task}",
    finished_var: str = "{finished_tasks}",
    include_tasks: bool = True,
    single_action: bool = False,
) -> str:
    """Build a decision prompt from action classes.

    Args:
        action_names: List of action names to include (e.g., ["execute", "plan", "answer"]).
        question_var: Variable placeholder for question (default: "{question}").
        pending_var: Variable placeholder for pending tasks.
        current_var: Variable placeholder for current task.
        finished_var: Variable placeholder for finished tasks.
        include_tasks: Whether to include task sections (default: True).
        single_action: If True, simplify prompt for single action (no "Choose ONE action").

    Returns:
        Formatted decision prompt string.

    Raises:
        ValueError: If an unknown action name is provided.
    """
    if single_action:
        assert len(action_names) == 1, f"single_action=True requires exactly 1 action, got {len(action_names)}"

    prompt_lines = []

    if not single_action:
        prompt_lines.extend([
            "Decide the next action based on current progress.",
            "",
        ])

    prompt_lines.extend([
        f"Question: {question_var}",
        "",
    ])

    if include_tasks:
        prompt_lines.extend([
            "Finished tasks:",
            finished_var,
            "",
            "Current task:",
            current_var,
            "",
            "Pending tasks:",
            pending_var,
            "",
        ])

    if single_action:
        prompt_lines.extend([
            "Provide the details in this format:",
            ""
        ])
    else:
        prompt_lines.extend([
            "Choose ONE action and output only its block:",
            ""
        ])

    # Add action options from action classes
    for i, action_name in enumerate(action_names, 1):
        if action_name not in ACTIONS:
            raise ValueError(f"Unknown action: {action_name}")

        action_cls = ACTIONS[action_name]
        if single_action:
            prompt_lines.append(action_cls.format_template)
        else:
            prompt_lines.append(f"{i}. {action_cls.tree_description}:")
            prompt_lines.append(action_cls.format_template)
        prompt_lines.append("")

    # Collect all guidelines
    all_guidelines = []
    for action_name in action_names:
        all_guidelines.extend(ACTIONS[action_name].guidelines)

    if all_guidelines:
        prompt_lines.append("Guidelines:")
        for guideline in all_guidelines:
            prompt_lines.append(f"- {guideline}")

    return "\n".join(prompt_lines)


def build_action_prompt(
    action_names: List[str],
    question: str,
    pending_tasks: List[str],
    current_task: List[str] = None,
    finished_tasks: List[str] = None,
    single_action: bool = False,
) -> str:
    """Build prompt with an explicit set of actions and task lists.

    Args:
        action_names: List of action names to include in the prompt.
        question: Research question.
        pending_tasks: Tasks not yet started.
        current_task: The current in-progress task (typically 0 or 1 item).
        finished_tasks: Tasks fully completed.
        single_action: If True, simplify prompt for single action.

    Returns:
        Formatted decision prompt with task values filled in.
    """
    current_task = current_task or []
    finished_tasks = finished_tasks or []
    pending_str = "\n".join(f"- {t}" for t in pending_tasks) if pending_tasks else "(none)"
    current_str = "\n".join(f"- {t}" for t in current_task) if current_task else "(none)"
    finished_str = "\n".join(f"- {t}" for t in finished_tasks) if finished_tasks else "(none)"

    prompt_template = build_decision_prompt(action_names, include_tasks=True, single_action=single_action)
    return prompt_template.format(
        question=question,
        pending_tasks=pending_str,
        current_task=current_str,
        finished_tasks=finished_str,
    )


def parse_action_from_response(text: str) -> str:
    """Extract action name from <action>...</action> format.

    Args:
        text: Response text containing action tag.

    Returns:
        Action name (lowercase) or "unknown" if not found.
    """
    import re
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    return match.group(1).strip().lower() if match else "unknown"


def parse_decision(text: str) -> tuple:
    """Parse main agent response to determine next state.

    Args:
        text: Response text from main agent.

    Returns:
        Tuple of (action_name, data_dict).
    """
    action_name = parse_action_from_response(text)
    if action_name in ACTIONS:
        data = ACTIONS[action_name].parse(text)
        return action_name, data
    return "unknown", {}


# =============================================================================
# STATIC PROMPTS
# =============================================================================

INIT_PROMPT = """As a Task Decomposer Agent, your objective is to analyze the given task and decompose it into subtasks if the task requires multiple searches.

You have been provided with the following objective:
{question}

Please format the subtasks as `plan` action and a numbered list within <tasks> tags, as demonstrated below:
<action>plan</action>
<tasks>
<task>Subtask 1</task>
<task>Subtask 2</task>
</tasks>

Each subtask should be concise, concrete, and achievable.
Ensure that the task plan is created without asking any questions.
Be specific and clear.
"""

WORKER_PROMPT = """You are a helpful search agent. Given a subtask, complete it by either:
- Searching the internet if external information is needed.
- Answering directly if you already know the answer.

Provide a concise summary of your findings or answer."""

EXAM_PROMPT = """You are a verification agent. A suspicious output has been observed, so verify correctness step by step.

Original question: {question}

Step {step_idx} ({step_state}) subagent chat history:
{subagent_history}

Verify each step:
1. Does the result actually answer the task?
2. Is the information consistent with the original question and main agent context?
3. Are there any obvious errors or hallucinations?

Output:
<verdict>correct OR incorrect</verdict>
<reason>brief explanation</reason>
<correction>if incorrect, provide the correct information or suggest what to search instead</correction>
"""

REWIND_PROMPT = """As a Rewind Agent, your objective is to analyze the failed research path and decide where to rewind.

You have been provided with the following history:

{history}

Please format your output within tags, as demonstrated below:
<rewind_to>turn_index</rewind_to>
<summary>Summary text</summary>

`rewind_to = N` means rewind to just after turn N (1-indexed). For example, if you output 2, turns 3, 4, 5, ... are removed; turns 1, 2 are kept.

The summary replaces the assistant feedback for turn N. Write it as a concise assistant reply:
- Start with "This approach will not work." then suggest a specific alternative (e.g., different search terms, verify spelling, try a related query).
- Keep it actionable so the main agent knows what to try next.
"""

# THINK_PROMPT = """You are a reflection agent. Carefully review the conversation history above.

# Question: {question}

# First think and analyze freely. Then provide concise, actionable suggestions for what to do next wrapped in <thought>...</thought>."""

THINK_PROMPT = """You are a reflection agent. Review the conversation history above and assess the current situation.

Question: {question}

Instructions:
1) Write your internal reasoning in <think>...</think>. Keep it brief and focused.
2) Then write a concise, actionable assessment in <summarize>...</summarize>:
   - What is the user trying to achieve?
   - What information is missing or ambiguous (if any)?
   - What specific next steps should be taken?

Constraints:
- Do NOT solve the question or provide the final answer.
- Output ONLY the two tagged blocks, in the order shown.

Output format:
<think>
...
</think>
<summarize>
...
</summarize>
"""



ANSWER_PROMPT = """Based on the research above, answer the question now.

Question: {question}

Output:
<action>answer</action>
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>
"""

SELECT_PROMPT = """You are a selection agent. Given a question and multiple responses from different tools, select the one that best answers the question.

Question: {question}

Task: {task}

Responses:
{responses}

Select the response that best answers the question. Output the response number (1-indexed):
<select>response_number</select>
"""

EXECUTE_CHECK_PROMPT = """Assess your completion status for this task:

Task: {task}

<status>success|partial|fail</status>
<note>Brief explanation</note>

- success: Task fully completed, needed information found
- partial: Made progress but some aspects remain incomplete
- fail: Could not complete (search failed, info not found, etc.)
"""
