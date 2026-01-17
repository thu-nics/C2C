SUMMARIZE_PROMPT = """Summarize this 2-message exchange while preserving all useful information.
Keep the same roles and format, but make the content more concise.

Examples:

Input:
Role: assistant
Content: [Tool call: search_wiki] {{"entity": "Tokyo"}}
Role: tool
Content: Tokyo is the capital of Japan with a population of approximately 13.96 million as of 2024. The Greater Tokyo Area has over 37 million people, making it the most populous metropolitan area in the world.

Output:
Role: assistant
Content: Searching Wikipedia for Tokyo.
Role: tool
Content: Tokyo is Japan's capital with 13.96 million people (2024). Greater Tokyo Area has 37+ million.

Input:
Role: assistant
Content: [Tool call: search_google] {{"query": "Badly Drawn Boy instruments"}}
Role: tool
Content: Damon Gough, known as Badly Drawn Boy, is an English singer-songwriter and multi-instrumentalist who plays guitar, keyboards, drums and various other instruments on his recordings.

Output:
Role: assistant
Content: Searching Google for Badly Drawn Boy instruments.
Role: tool
Content: Badly Drawn Boy (Damon Gough) is a multi-instrumentalist who plays guitar, keyboards, drums and more.

Now summarize this exchange:

Role: {role1}
Content: {content1}
Role: {role2}
Content: {content2}

Output the summarized messages in the exact format:
Role: {role1}
Content: <summarized content>
Role: {role2}
Content: <summarized content>"""


CONTRACT_PROMPT = """Merge these 4 messages (2 rounds) into 2 messages (1 round).
Show the starting intent and final result, hiding intermediate steps.

Examples:

Input:
Role: user
Content: Find out who directed the movie Inception and what other famous movies they directed.
Role: assistant
Content: I'll search for the director of Inception first.
Role: user
Content: Now find their other movies.
Role: assistant
Content: Christopher Nolan directed Inception. He also directed The Dark Knight trilogy, Interstellar, Dunkirk, Tenet, and Oppenheimer.

Output:
Role: user
Content: Find out who directed Inception and what other famous movies they directed.
Role: assistant
Content: Christopher Nolan directed Inception. He also directed The Dark Knight trilogy, Interstellar, Dunkirk, Tenet, and Oppenheimer.

Input:
Role: user
Content: What's the capital of France?
Role: assistant
Content: The capital of France is Paris.
Role: user
Content: What's its population?
Role: assistant
Content: Paris has a population of approximately 2.1 million in the city proper, and over 12 million in the metropolitan area.

Output:
Role: user
Content: What's the capital of France and its population?
Role: assistant
Content: The capital of France is Paris, with approximately 2.1 million people in the city proper and over 12 million in the metropolitan area.

Now merge these 4 messages:

Role: {role1}
Content: {content1}
Role: {role2}
Content: {content2}
Role: {role3}
Content: {content3}
Role: {role4}
Content: {content4}

Output exactly 2 messages:
Role: {role1}
Content: <merged content>
Role: {role4}
Content: <merged content>"""


SUMMARIZE_TOOL_RESP_PROMPT = """Summarize the tool response, keeping ONLY information relevant to the original query.
Discard unrelated search results or tangential information.

Examples:

Input:
Tool call: search_wiki({{"entity": "Tokyo"}})
Tool response: [Result 1] Tokyo is the capital of Japan with 13.96 million people. [Result 2] Kyoto was the former capital. [Result 3] Tokyo Tower is a famous landmark built in 1958.

Output:
Tokyo is Japan's capital with 13.96 million people.

Input:
Tool call: search_google({{"query": "Badly Drawn Boy instruments"}})
Tool response: [Result 1] Badly Drawn Boy (Damon Gough) plays guitar, keyboards, drums. [Result 2] The band was formed in 1995. [Result 3] His real name is Damon Michael Gough, born October 1969.

Output:
Badly Drawn Boy (Damon Gough) plays guitar, keyboards, drums.

Input:
Tool call: search_wiki({{"entity": "Christopher Nolan filmography"}})
Tool response: [Result 1] Christopher Nolan directed Inception, The Dark Knight, Interstellar. [Result 2] He was born in London. [Result 3] His brother Jonathan also writes screenplays.

Output:
Christopher Nolan directed Inception, The Dark Knight, Interstellar.

Now summarize this tool response based on the call:

Tool call: {tool_call}
Tool response: {tool_content}

Output only the relevant summarized content, nothing else."""

SMART_SUMMARIZE_TOOL_RESP_PROMPT = """You are a smart search tool that condenses raw tool outputs.

Input:
- Tool call: {tool_call}
- Tool response: {tool_content}

Task:
- Return ONLY the useful results relevant to the tool call/query.
- If the tool response is a JSON list of items, output a JSON list of ONLY the useful items.
  - Keep existing keys like "docid" and "score" if present.
  - Rewrite "snippet" to be much shorter using "..." to skip unimportant text.
  - Include ONLY facts that appear in the original snippet/tool response (no new info).
  - Add a brief parenthetical comment inside the snippet describing limitations/uncertainty (e.g., missing constraints, examples-only).
- Drop irrelevant/duplicative items.
- If nothing is useful, DO NOT output []. Instead output a single-item JSON list like:
  [{{"snippet": "... (no relevant information found in the tool response for this query.)"}}]

Output ONLY the condensed results (no extra prose)."""