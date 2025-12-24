"""Prompts for tree-based research workflow."""

INIT_PROMPT = """As a Task Decomposer Agent, your objective is to analyze the given task and decompose it into subtasks if the task requires multiple searches.

You have been provided with the following objective:
{question}

Please format the subtasks as `revise` action and a numbered list within <tasks> tags, as demonstrated below:
<action>revise</action>
<tasks>
<task>Subtask 1</task>
<task>Subtask 2</task>
</tasks>

Each subtask should be concise, concrete, and achievable.
Ensure that the task plan is created without asking any questions.
Be specific and clear.
"""

# Tree workflow actions - modular action definitions
TREE_ACTIONS = {
    "execute": {
        "description": "Execute next task - proceed with the next subtask",
        "format": """<action>execute</action>
<task>Self-contained subtask with necessary context</task>""",
        "guidelines": [
            "When executing, include all needed context in the <task>",
            "Frame the task to be self-contained - the subagent cannot see prior conversation",
            "Include relevant context (e.g., candidate names, criteria) in the task itself",
        ]
    },
    "revise": {
        "description": "Revise tasks - update the remaining task list based on new findings",
        "format": """<action>revise</action>
<tasks>
<task>Revised subtask 1</task>
<task>Revised subtask 2</task>
</tasks>""",
        "guidelines": [
            "Build on previous findings - if candidates were found, verify them against remaining criteria",
            "If a subtask failed, try a different approach (e.g., search for specific candidate + criterion)",
            "Avoid repetition - don't repeat failed searches with identical queries",
        ]
    },
    "exam": {
        "description": "Exam - verify a previous step's result if it seems suspicious or critical",
        "format": """<action>exam</action>
<step>step_index to examine (0-indexed)</step>""",
        "guidelines": [
            "Subagent results may be wrong - do not blindly trust them",
            "If a result seems suspicious or inconsistent, use exam to verify before proceeding",
            "Verify critical steps that affect the final answer",
        ]
    },
    "rewind": {
        "description": "Rewind - if the current path repeatedly fails and you need to backtrack to restart",
        "format": """<action>rewind</action>""",
        "guidelines": [
            "Only use rewind when the entire workflow direction is fundamentally wrong",
            "Use when you need to backtrack to a very early step to restart",
            "Try different approaches after rewinding",
            "Use rewind only if you must restart from an early step due to a major workflow error",
        ]
    },
    "answer": {
        "description": "Answer - final response when you have enough verified information",
        "format": """<action>answer</action>
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>""",
        "guidelines": [
            "Provide the final answer when you have enough verified information",
            "Ensure justification is concise (1-2 sentences)",
            "Answer should be the final answer span only",
        ]
    },
}


def build_decision_prompt(actions: list[str], question_var: str = "{question}", tasks_var: str = "{tasks}", include_tasks: bool = True) -> str:
    """Build a decision prompt from selected actions.

    Args:
        actions: List of action names to include (e.g., ["execute", "revise", "answer"])
        question_var: Variable name for question (default: "{question}")
        tasks_var: Variable name for tasks (default: "{tasks}")
        include_tasks: Whether to include the "Remaining tasks:" section (default: True)

    Returns:
        Formatted decision prompt string
    """
    prompt_lines = [
        "Decide the next action based on current progress.",
        "",
        f"Question: {question_var}",
        "",
    ]

    if include_tasks:
        prompt_lines.extend([
            f"Remaining tasks:",
            tasks_var,
            "",
        ])

    prompt_lines.extend([
        "Choose ONE action and output only its block:",
        ""
    ])

    # Add action options
    for i, action_name in enumerate(actions, 1):
        if action_name not in TREE_ACTIONS:
            raise ValueError(f"Unknown action: {action_name}")

        action = TREE_ACTIONS[action_name]
        prompt_lines.append(f"{i}. {action['description']}:")
        prompt_lines.append(action["format"])
        prompt_lines.append("")

    # Collect all guidelines
    all_guidelines = []
    for action_name in actions:
        all_guidelines.extend(TREE_ACTIONS[action_name]["guidelines"])

    if all_guidelines:
        prompt_lines.append("Guidelines:")
        for guideline in all_guidelines:
            prompt_lines.append(f"- {guideline}")

    return "\n".join(prompt_lines)


# Pre-built decision prompts for convenience
DECISION_PROMPT = build_decision_prompt(["execute", "revise", "rewind", "answer"])

WORKER_PROMPT = """You are a helpful search agent. Given a subtask, complete it by either:
- Searching the internet if external information is needed.
- Answering directly if you already know the answer.

Provide a concise summary of your findings or answer."""

EXAM_PROMPT = """You are a verification agent. Examine if the subagent's result is correct.

Original question: {question}

Main agent context (previous steps):
{main_context}

Step {step_idx} to verify:
[Task] {task}
[Subagent Result] {result}

Verify:
1. Does the result actually answer the task?
2. Is the information consistent with the original question?
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
<rewind_to>step_index</rewind_to>
<summary>Summary text</summary>

`rewind_to = N` means rewind to just after step N. For example, if you output 2, steps 3, 4, 5, ... are removed; steps 0, 1, 2 are kept.

The summary replaces the assistant feedback for step N. Write it as a concise assistant reply:
- Start with "This approach will not work." then suggest a specific alternative (e.g., different search terms, verify spelling, try a related query).
- Keep it actionable so the main agent knows what to try next.
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
