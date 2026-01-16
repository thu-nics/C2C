"""Prompts for tree-based research workflow."""

from typing import List
from pydantic import Field, BaseModel

MAIN_AGENT_SYSTEM_MESSAGE = """You are a helpful assistant."""

FORCE_ANSWER_PROMPT = """Based on the research above, answer the question now.

Question: {question}
"""

class AnswerFormat(BaseModel):
    justification: str = Field(description="1-2 short sentences explaining the answer")
    answer: str = Field(description="The final answer span")

def build_task_state_prompt(
    pending: List[str],
    current: List[str],
    finished: List[str],
    question: str = None,
    question_var: str = "{question}",
    include_tasks: bool = True,
) -> str:
    """Build task state prompt for main agent.

    Args:
        pending: Pending tasks.
        current: Current task (0 or 1 item).
        finished: Completed tasks.
        question: Actual question text (if provided, substitutes question_var).
        question_var: Variable placeholder for question.
        include_tasks: Whether to include task sections.

    Returns:
        Formatted prompt string.
    """
    def format_list(tasks: List[str]) -> str:
        return "\n".join(f"- {t}" for t in tasks) if tasks else "(none)"

    q = question if question else question_var

    lines = [
        "Decide the next action based on current progress.",
        "",
        f"Question: {q}",
        "",
    ]

    if include_tasks:
        lines.extend([
            "Finished tasks:",
            format_list(finished),
            "",
            "Current task:",
            format_list(current),
            "",
            "Pending tasks:",
            format_list(pending),
            "",
        ])

    lines.append("Now, call the appropriate tool to proceed whenever possible. You may directly respond (not recommended) ONLY when you have gathered sufficient information to deterministically answer the question.")

    return "\n".join(lines)

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
