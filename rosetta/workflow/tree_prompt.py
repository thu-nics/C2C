"""Prompts for tree-based research workflow."""

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

# Tree workflow actions - modular action definitions
TREE_ACTIONS = {
    "execute": {
        "description": "Execute - work on the current task",
        "format": """<action>execute</action>
<task>Self-contained subtask with necessary context</task>""",
        "guidelines": [
            "[execute] In <task>, include all and only the context the subagent needs; assume it cannot see any prior conversation.",
            "[execute] Include what was already found; focus only on the remaining information needed.",
        ],
        "with_param": True,
    },
    "plan": {
        "description": "Plan tasks - replace current and pending tasks with a new task list",
        "format": """<action>plan</action>
<tasks>
<task>Revised subtask 1</task>
<task>Revised subtask 2</task>
</tasks>""",
        "guidelines": [
            "[plan] Replaces all current and pending tasks with new tasks. Finished tasks are preserved.",
            "[plan] You may split one subtask into multiple smaller subtasks when helpful.",
            "[plan] Each subtask must be narrowly scoped, achievable within a few searches, and have a clear desired result.",
            "[plan] For failed/partial tasks, change the approach rather than repeating the same task.",
        ],
        "with_param": True,
    },
    "exam": {
        "description": "Exam - verify a previous step's result if it seems suspicious or critical",
        "format": """<action>exam</action>
<step>step_index to examine (1-indexed)</step>""",
        "guidelines": [
            "[exam] Use this action to verify a specific prior step when its result is suspicious, inconsistent, or high-impact.",
            "[exam] Provide the 1-indexed <step> you want to re-check.",
        ],
        "with_param": True,
    },
    "think": {
        "description": "Think - pause to reflect and get a concise assessment before choosing next action",
        "format": """<action>think</action>""",
        "guidelines": [
            "[think] Use this action when the next operation is unclear and you need a brief assessment before proceeding.",
        ],
        "with_param": False,
    },
    "rewind": {
        "description": "Rewind - backtrack when similar tasks fail repeatedly",
        "format": """<action>rewind</action>""",
        "guidelines": [
            "[rewind] Use when similar tasks fail repeatedly (similar query variations, similar dead ends).",
            "[rewind] Use when exploring the wrong entity or topic.",
            "[rewind] After rewinding, switch to a different approach.",
        ],
        "with_param": False,
    },
    "answer": {
        "description": "Answer - final response when you have enough verified information",
        "format": """<action>answer</action>
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>""",
        "guidelines": [
            "[answer] Use this action only when you have enough verified information to respond conclusively.",
            "[answer] Keep the justification to 1–2 short sentences.",
            "[answer] Put the final answer span in the \"answer\" field and the brief rationale in \"justification\".",
        ],
        "with_param": True,
    },
    "parallel_execute": {
        "description": "Parallel Execute - work on multiple independent tasks simultaneously",
        "format": """<action>parallel_execute</action>
<tasks>
<task>Self-contained subtask 1 with necessary context</task>
<task>Self-contained subtask 2 with necessary context</task>
</tasks>""",
        "guidelines": [
            "[parallel_execute] Use when multiple tasks are independent and can be executed concurrently.",
            "[parallel_execute] Each <task> must be self-contained with all context needed; subagents cannot see prior conversation.",
            "[parallel_execute] Results will be collected and presented as multi-round conversation records.",
        ],
        "with_param": True,
    },
}

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
