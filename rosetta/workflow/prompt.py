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

SIMPLE_RESEARCH_PROMPT = """Analyze the task and decompose it into subtasks if the task requires multiple searches.

Objective:
{question}

List the subtasks as a numbered list (no tags). Each subtask should be concise, concrete, and achievable.
Ensure that the task plan is created without asking any questions. Be specific and clear.

Given a subtask, complete it by either searching the internet if external information is needed, or answering directly if you already know the answer. Provide a concise summary of your findings or answer.

Revise the task list based on the search result above.
Original question: {question}

Guidelines:
- Build on previous findings. If candidates were found, verify them against remaining criteria.
- If a subtask failed, try a different approach (e.g., search for specific candidate + criterion).
- Provide the final answer when you have enough information.

After the final tool response, return ONLY a JSON object on a single line (no markdown, no code fences), exactly with these keys:
- justification: string (1-2 short sentences)
- answer: string (the final answer span only)

Constraints:
- Do not include any extra keys.
- Do not include any extra text before/after the JSON.
- For yes/no questions, answer must be exactly "Yes" or "No".

Example (format only):
{{"justification":"Both Scott Derrickson and Ed Wood were American.", "answer":"Yes"}}"""

SEARCH_AGENT_PROMPT_MIRO = """You are a helpful search agent. Given a subtask, complete it by breaking it down into clear steps and working through them methodically.

# Guidelines:
- You only have access to the tools provided. You can only use one tool per message, and will receive the result of that tool in the next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use.
- **IMPORTANT: Each step must involve exactly ONE tool call only, unless the task is already solved.**
- If you have known the answer, provide a concise summary of your findings or answer."""

SEARCH_AGENT_PROMPT = """You are a helpful search agent. Given a subtask, complete it by either:
- Searching the internet if external information is needed.
- Answering directly if you already know the answer.

Provide a concise summary of your findings or answer."""


# Error categorization for workflow evaluation
ERROR_CATEGORIES = {
    "Search Strategy Failure": "Poor/vague search queries, gave up too early, didn't try alternative search terms",
    "Information Not Retrieved": "Searched appropriately but needed information not found by search engine",
    "Retrieved Wrong Information": "Search engine returned incorrect/misleading facts that model trusted",
    "Retrieved Related But Wrong Entity": "Found similar/related entity instead of the correct one",
    "Information Retrieved But Ignored": "Had correct information in chat history but didn't use it in final answer",
    "Multi-Hop Reasoning Failure": "Failed to properly connect information across multiple search hops",
    "Question Misinterpretation": "Fundamentally misunderstood what the question was asking",
    "Answer Extraction Error": "Had correct understanding but formatted/extracted answer incorrectly (too verbose, too brief, wrong specificity)",
    "Premature Conclusion": "Concluded without sufficient verification or additional needed searches",
}

ERROR_CATEGORIZATION_PROMPT = """Analyze this multi-agent research workflow that produced an incorrect answer.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {pred_answer}

FULL CHAT HISTORY:
{chat_history}

Categorize the PRIMARY reason for failure into ONE category:

{category_list}

Respond with ONLY the category name."""

LLM_JUDGE_PROMPT = """You are comparing a predicted answer to a gold (reference) answer. Your task is to determine if they convey the SAME meaning.

IMPORTANT: Do NOT use your own knowledge to verify factual correctness. Only compare whether the two answers match each other semantically.

Question (for context only): {question}

Gold Answer: {gold_answer}

Predicted Answer: {pred_answer}

Judge if the Predicted Answer matches the Gold Answer by checking:
- Do they refer to the same entity, person, place, or concept?
- Are they numerically equivalent (e.g., "1,000" vs "1000", "50%" vs "half")?
- Are they paraphrases expressing the same meaning?
- For yes/no answers: "yes" only matches "yes", and "no" only matches "no" - opposites are INCORRECT

Ignore differences in:
- Capitalization, punctuation, or minor formatting
- Extra context or explanation (if core answer matches)
- Phrasing variations that preserve meaning

Reply 'CORRECT' if the Predicted Answer conveys the same meaning as the Gold Answer.
Reply 'INCORRECT' if they refer to different things, give conflicting information, or are opposites.

Your response must be ONLY 'CORRECT' or 'INCORRECT'."""