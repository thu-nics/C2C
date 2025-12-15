SEARCH_TASK_DECOMPOSE_PROMPT = """As a Task Decomposer Agent, your objective is to analyze the given task and decompose it into search subtasks if the task requires multiple searches.

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

SEARCH_AGENT_PROMPT = """You are a helpful search agent. Given a question, you will write a query to search the internet for information and summarize the results."""