"""Context management utilities for compressing conversation history."""

from typing import List, Dict
from camel.models import BaseModelBackend


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


def _get_content(msg: Dict) -> str:
    """Extract content from message, converting tool_calls to text if needed."""
    content = msg.get("content", "")
    if content:
        return content
    # If content is empty but has tool_calls, describe the tool call
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        args = func.get("arguments", "{}")
        return f"[Tool call: {name}] {args}"
    return ""


def _parse_output(text: str, roles: List[str]) -> List[Dict[str, str]]:
    """Parse LLM output into message dicts."""
    messages = []
    lines = text.strip().split("\n")
    current_role = None
    current_content = []

    for line in lines:
        if line.startswith("Role:"):
            if current_role is not None:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            current_role = line[5:].strip().lower()
            current_content = []
        elif line.startswith("Content:"):
            current_content.append(line[8:].strip())
        elif current_role is not None:
            current_content.append(line)

    if current_role is not None:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })

    # Ensure correct roles if parsing fails
    if len(messages) != len(roles):
        return [{"role": r, "content": "..."} for r in roles]

    for i, role in enumerate(roles):
        messages[i]["role"] = role

    return messages


def summarize(
    messages: List[Dict[str, str]],
    model: BaseModelBackend,
) -> List[Dict[str, str]]:
    """Summarize a 2-message exchange while preserving useful information.

    Args:
        messages: List of 2 message dicts with "role" and "content" keys.
        model: CAMEL model backend for summarization.

    Returns:
        List of 2 summarized message dicts with same roles.
    """
    assert len(messages) == 2, "summarize requires exactly 2 messages"

    prompt = SUMMARIZE_PROMPT.format(
        role1=messages[0]["role"],
        content1=_get_content(messages[0]),
        role2=messages[1]["role"],
        content2=_get_content(messages[1]),
    )

    response = model.run([
        {"role": "system", "content": "You summarize conversations concisely. Output only the requested format."},
        {"role": "user", "content": prompt},
    ])
    roles = [messages[0]["role"], messages[1]["role"]]
    return _parse_output(response.choices[0].message.content, roles)


def contract(
    messages: List[Dict[str, str]],
    model: BaseModelBackend,
) -> List[Dict[str, str]]:
    """Contract 4 messages (2 rounds) into 2 messages (1 round).

    Args:
        messages: List of 4 message dicts (2 conversation rounds).
        model: CAMEL model backend for contraction.

    Returns:
        List of 2 message dicts showing start intent and final result.
    """
    assert len(messages) == 4, "contract requires exactly 4 messages"

    prompt = CONTRACT_PROMPT.format(
        role1=messages[0]["role"],
        content1=_get_content(messages[0]),
        role2=messages[1]["role"],
        content2=_get_content(messages[1]),
        role3=messages[2]["role"],
        content3=_get_content(messages[2]),
        role4=messages[3]["role"],
        content4=_get_content(messages[3]),
    )

    response = model.run([
        {"role": "system", "content": "You merge conversation rounds concisely. Output only the requested format."},
        {"role": "user", "content": prompt},
    ])
    roles = [messages[0]["role"], messages[3]["role"]]
    return _parse_output(response.choices[0].message.content, roles)
