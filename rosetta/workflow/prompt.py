SEARCH_TASK_DECOMPOSE_PROMPT = """As a Task Decomposer Agent, your objective is to analyze the given task and decompose it into subtasks if the task requires multiple searches.

You have been provided with the following objective:

{content}

Please format the subtasks as a numbered list within <tasks> tags, as demonstrated below:
<tasks>
<task>Subtask 1</task>
<task>Subtask 2</task>
</tasks>

Each subtask should be concise, concrete, and achievable.
Ensure that the task plan is created without asking any questions.
Be specific and clear.
"""

TASK_REVISE_PROMPT = """As a Task Reviser Agent, your objective is to update the task list based on the search result provided above.

Original question: {question}

Based on the search result, you should:
- Remove tasks that have been completed successfully.
- Modify or simplify tasks that failed or returned insufficient information.
- Provide the final answer if you have gathered enough information.

If tasks remain, output the revised list within <tasks> tags:
<tasks>
<task>Revised subtask 1</task>
<task>Revised subtask 2</task>
</tasks>

If you can answer the original question, output within <answer> tags:
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>

Constraints:
- Each subtask should be concise, concrete, and achievable.
- For yes/no questions, answer must be exactly "Yes" or "No".
- Do not include any extra text outside the tags.
"""

FORCE_ANSWER_PROMPT = """Answer the question based on collected information.

Question: {question}

Collected information:
{info}

Output:
<answer>{{"justification":"...", "answer":"..."}}</answer>
"""

RESPONSE_PROMPT_DIRECT = """Answer the question based on the Search Results.

Question: {question}

Search Results:
{search_info}

Return ONLY a JSON object on a single line (no markdown, no code fences), exactly with these keys:
- justification: string (1-2 short sentences)
- answer: string (the final answer span only)

Constraints:
- Do not include any extra keys.
- Do not include any extra text before/after the JSON.
- For yes/no questions, answer must be exactly "Yes" or "No".

Example (format only):
{{"justification":"Both Scott Derrickson and Ed Wood were American.", "answer":"Yes"}}"""

SEARCH_AGENT_PROMPT = """You are a helpful search agent. Given a subtask, complete it by either:
- Searching the internet if external information is needed.
- Answering directly if you already know the answer.

Provide a concise summary of your findings or answer."""