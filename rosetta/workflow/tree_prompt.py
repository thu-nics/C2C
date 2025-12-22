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

DECISION_PROMPT = """Decide the next action based on current progress.

Question: {question}

Remaining tasks:
{tasks}

Choose ONE action and output only its block:

1. Execute next task (run one subtask through the worker):
<action>execute</action>
<task>Self-contained subtask with necessary context</task>

2. Revise tasks (update the remaining task list based on new findings):
<action>revise</action>
<tasks>
<task>Revised subtask 1</task>
<task>Revised subtask 2</task>
</tasks>

3. Rewind (only if the entire path is failing and cannot be fixed with revision):
<action>rewind</action>

4. Answer (final response when you have enough verified information):
<action>answer</action>
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>

Guidelines:
- When executing, include all needed context in the <task>.
- Build on prior findings and avoid repetition.
- Use rewind only if you must restart from an early step due to a major workflow error.
- Try different approaches after rewinding.
"""

DECISION_PROMPT_3 = """Decide the next action based on current progress.

Question: {question}

Remaining tasks:
{tasks}

Choose ONE action and output only its block:

1. Execute next task (run one subtask through the worker):
<action>execute</action>
<task>Self-contained subtask with necessary context</task>

2. Revise tasks (update the remaining task list based on new findings):
<action>revise</action>
<tasks>
<task>Revised subtask 1</task>
<task>Revised subtask 2</task>
</tasks>

3. Answer (final response when you have enough verified information):
<action>answer</action>
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>

Guidelines:
- When executing, include all needed context in the <task>.
- Build on prior findings and avoid repetition.
"""

DECISION_PROMPT_5 = """Based on the current progress, decide your next action.

Question: {question}

Remaining tasks:
{tasks}

Choose one action:

1. Execute next task - proceed with the next subtask:
<action>execute</action>
<task>Subtask and its necessary context to execute</task>

2. Revise tasks - modify the task list based on new findings:
<action>revise</action>
<tasks>
<task>Revised subtask 1</task>
<task>Revised subtask 2</task>
</tasks>

3. Exam - verify a previous step's result if it seems suspicious or critical:
<action>exam</action>
<step>step_index to examine (0-indexed)</step>

4. Rewind - if the current path repeatedly fails and you need to backtrack to a very early step to restart:
<action>rewind</action>

5. Answer - if you have enough information to answer:
<action>answer</action>
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>

Guidelines:
- Subagent results may be wrong. Do not blindly trust them. If a result seems suspicious, use "exam" to verify before proceeding.
- When executing, frame the task to be self-contained. The subagent cannot see prior conversation, so include relevant context (e.g., candidate names, criteria) in the task itself.
- Build on previous findings. If candidates were found, verify them against remaining criteria.
- If a subtask result seems wrong or inconsistent, use "exam" to verify it or "revise" to try a different approach.
- Only use rewind when the entire workflow direction is fundamentally wrong and you need to backtrack to a very early step to restart.
- Provide the final answer when you have enough verified information.
"""

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
