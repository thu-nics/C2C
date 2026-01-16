import json
from typing import Dict, Any, List

def msg_system(content: str) -> Dict[str, Any]:
    return {"role": "system", "content": content}

def msg_user(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}

def msg_assistant(content: str, tool_call=None) -> Dict[str, Any]:
    msg = {"role": "assistant", "content": content or ""}
    if tool_call:
        msg["tool_calls"] = [{
            "id": tool_call.id,
            "type": "function",
            "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
        }]
    return msg

def msg_tool(tool_call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

def execute_tool(tool_map: Dict, name: str, args: Dict) -> str:
    tool = tool_map.get(name)
    if tool is None:
        return f"Error: Unknown tool '{name}'"
    try:
        result = tool.func(**args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"

def _clean_for_api(messages: List[dict]) -> List[dict]:
    """Remove internal keys (starting with _) before sending to API."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]