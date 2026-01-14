"""Utility functions for CAMEL message conversion."""

import os
from typing import List, Optional, Any
from dotenv import load_dotenv, find_dotenv

from camel.messages import BaseMessage, FunctionCallingMessage
from camel.memories import MemoryRecord, ContextRecord
from camel.types import OpenAIBackendRole, RoleType, ModelPlatformType, ModelType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig

def setup_env():
    """Setup environment variables."""
    load_dotenv(find_dotenv())

def create_model(
    provider: str,
    model_type: Optional[str] = None,
    model_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 32768,
    chat_template_kwargs: Optional[dict] = None,
    **kwargs: Any,
):
    """Create a model based on the provider.

    Args:
        provider: Model provider, one of "local", "openai", "gemini".
        model_type: Model type/name string. If None, uses provider defaults:
            - local: "local"
            - openai: GPT_4O_MINI
            - gemini: "gemini-3-flash-preview"
        model_url: API URL for local/compatible models.
        api_key: API key (uses env var if not provided).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        chat_template_kwargs: Custom chat template kwargs for local models.
            Defaults to {"enable_thinking": False} if not provided.
        **kwargs: Additional model config parameters.

    Returns:
        Configured CAMEL model instance.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider == "local":
        model_type = model_type or "local"
        # Use provided chat_template_kwargs or default
        effective_chat_template_kwargs = chat_template_kwargs if chat_template_kwargs is not None else {"enable_thinking": False}
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": {"chat_template_kwargs": effective_chat_template_kwargs},
                **kwargs,
            },
            api_key=api_key or "not-needed",
            url=model_url or "http://localhost:30000/v1",
        )
    elif provider == "openai":
        model_type = model_type or ModelType.GPT_5_MINI
        config = ChatGPTConfig(max_tokens=max_tokens, temperature=temperature)
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_type,
            model_config_dict=config.as_dict(),
            api_key=api_key,
        )
    elif provider == "gemini":
        model_type = model_type or "gemini-3-flash-preview"
        config = ChatGPTConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=kwargs.get("reasoning_effort", "medium"),
        )
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict=config.as_dict(),
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif provider == "fireworks":
        model_type = model_type or "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
        config = ChatGPTConfig(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict=config.as_dict(),
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            url="https://api.fireworks.ai/inference/v1",
        )
    else:
        raise ValueError(
            f"Unsupported model provider: {provider}. "
            f"Choose from: local, openai, gemini"
        )

def context_records_to_memory_records(
    records: List[ContextRecord]
) -> List[MemoryRecord]:
    """Convert ContextRecord list to standard message format.
    
    Args:
        records: List of MemoryRecord objects.
    """
    return [ctx_record.memory_record for ctx_record in records]

def memoryRecord_flip_role(message: MemoryRecord) -> MemoryRecord:
    """Flip the role of a message."""
    if message.message.role_type == RoleType.USER:
        message.message.role_type = RoleType.ASSISTANT
    elif message.message.role_type == RoleType.ASSISTANT:
        message.message.role_type = RoleType.USER
    elif message.message.role_type == RoleType.SYSTEM:
        message.message.role_type = RoleType.SYSTEM
    elif message.message.role_type == RoleType.FUNCTION:
        message.message.role_type = RoleType.FUNCTION
    elif message.message.role_type == RoleType.TOOL:
        message.message.role_type = RoleType.TOOL
    else:
        raise ValueError(f"Unsupported role type: {message.message.role_type}.")
    return message

def messages_to_memoryRecords(
    chat_history: List[dict],
    skip_system: bool = False
) -> List[MemoryRecord]:
    """Convert standard message format to CAMEL MemoryRecord list.

    Args:
        chat_history: List of dictionaries with 'role' and 'content' keys.
                     Roles can be 'user', 'assistant', 'system', 'function',
                     'tool', or 'developer'.
        skip_system: Whether to skip system messages. Default is True.

    Returns:
        List of MemoryRecord objects suitable for CAMEL agents.

    Example:
        >>> chat_history = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': 'Hello'},
        ...     {'role': 'assistant', 'content': 'Hi there!'}
        ... ]
        >>> message_list = convert_to_camel_messages(chat_history)
        >>> len(message_list)  # System message skipped by default
        2
    """
    message_list = []

    # Build a mapping of tool_call_id -> function_name for tool messages
    # that don't have func_name specified
    tool_call_map = {}
    for msg in chat_history:
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                tool_call_map[tc['id']] = tc['function']['name']
    
    for message in chat_history:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage.make_user_message(
                        role_name="user", 
                        content=content
                    ),
                    role_at_backend=OpenAIBackendRole.USER
                )
            )
        elif role == 'assistant':
            # Check if this assistant message has tool_calls
            tool_calls = message.get('tool_calls')
            if tool_calls:
                # Use FunctionCallingMessage for assistant messages with tool calls
                # Extract function name and arguments from first tool_call
                first_call = tool_calls[0]
                func_name = first_call.get('function', {}).get('name')
                args_str = first_call.get('function', {}).get('arguments', '{}')
                import json
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                base_msg = FunctionCallingMessage(
                    role_name="assistant",
                    role_type=RoleType.ASSISTANT,
                    content=content,
                    meta_dict={'tool_calls': tool_calls},
                    func_name=func_name,
                    args=args,
                    tool_call_id=first_call.get('id')
                )
            else:
                base_msg = BaseMessage.make_assistant_message(
                    role_name="assistant",
                    content=content
                )
            message_list.append(
                MemoryRecord(
                    message=base_msg,
                    role_at_backend=OpenAIBackendRole.ASSISTANT
                )
            )
        elif role == 'system':
            if not skip_system:
                message_list.append(
                    MemoryRecord(
                        message=BaseMessage.make_system_message(
                            role_name="System",
                            content=content
                        ),
                        role_at_backend=OpenAIBackendRole.SYSTEM
                    )
                )
        elif role == 'function':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="function",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None
                    ),
                    role_at_backend=OpenAIBackendRole.FUNCTION
                )
            )
        elif role == 'tool':
            # Tool messages use FunctionCallingMessage with FUNCTION role
            tool_call_id = message.get('tool_call_id')
            func_name = message.get('func_name')

            # If func_name not provided, try to look it up from tool_call_map
            if not func_name and tool_call_id:
                func_name = tool_call_map.get(tool_call_id)

            message_list.append(
                MemoryRecord(
                    message=FunctionCallingMessage(
                        role_name="tool",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None,
                        result=content,
                        tool_call_id=tool_call_id,
                        func_name=func_name
                    ),
                    role_at_backend=OpenAIBackendRole.FUNCTION
                )
            )
        elif role == 'developer':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="developer",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None
                    ),
                    role_at_backend=OpenAIBackendRole.DEVELOPER
                )
            )
        else:
            raise ValueError(f"Unsupported role: {role}.")
    
    return message_list



def memoryRecords_to_messages(
    records: List[MemoryRecord]
) -> List[dict]:
    """Convert MemoryRecord list to standard message format.
    
    Args:
        records: List of MemoryRecord objects.
    """
    return [record.to_openai_message() for record in records]

def add_tool_requests_to_chat_history(
    chat_history: List[dict],
    tool_request,
) -> List[dict]:
    """Add tool requests to chat history."""
    last_msg = chat_history[-1]
    if last_msg.get("role") == "assistant":
        # Format tool_calls according to what record_interaction expects
        last_msg["tool_calls"] = [
            {
                "id": tool_request.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_request.tool_name,
                    "arguments": tool_request.args or {},
                },
            }
        ]
    return chat_history