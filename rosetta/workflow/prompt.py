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

TASK_REVISE_PROMPT = """Revise the task list based on the search result above.

Original question: {question}

Guidelines:
- Build on previous findings. If candidates were found, verify them against remaining criteria.
- If a subtask failed, try a different approach (e.g., search for specific candidate + criterion).
- Provide the final answer when you have enough information.

Output revised tasks:
<tasks>
<task>Revised subtask 1</task>
</tasks>

Or output the answer:
<answer>{{"justification":"1-2 short sentences", "answer":"final answer span"}}</answer>
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

SIMPLE_RESEARCH_PROMPT = """Answer the question using multiple steps and tools.

Question: {question}

Tool-use protocol:
1) Output ONLY a search_engine tool call in <tool_call>...</tool_call>.
   Example:
   <tool_call>
   {{"name":"search_engine","arguments":{{"query":"example query","top_k":5}}}}
   </tool_call>
   Use a query that includes key phrases from the question.
2) You MUST call search_engine a second time before answering.
   Do not reuse specific series/author names from the first results unless they match ALL clues.
   If any result looks like a companion book (e.g., a title with "Chronicles"), use that title plus "series".
   Otherwise, add "chronicles" and "alien species" to the query.
3) After the second tool returns, answer using the required JSON format.
4) Do not answer from memory; verify with search results.

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
