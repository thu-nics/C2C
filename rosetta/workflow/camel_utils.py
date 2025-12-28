"""Utility functions for CAMEL message conversion."""

import os
from typing import List, Optional, Any

from camel.messages import BaseMessage
from camel.memories import MemoryRecord, ContextRecord
from camel.types import OpenAIBackendRole, RoleType, ModelPlatformType, ModelType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig


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

def MemoryRecord_flip_role(message: MemoryRecord) -> MemoryRecord:
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
            message_list.append(
                MemoryRecord(
                    message=BaseMessage.make_assistant_message(
                        role_name="assistant",
                        content=content
                    ),
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
                        content=content
                    ),
                    role_at_backend=OpenAIBackendRole.FUNCTION
                )
            )
        elif role == 'tool':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="tool",
                        role_type=RoleType.DEFAULT,
                        content=content
                    ),
                    role_at_backend=OpenAIBackendRole.TOOL
                )
            )
        elif role == 'developer':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="developer",
                        role_type=RoleType.DEFAULT,
                        content=content
                    ),
                    role_at_backend=OpenAIBackendRole.DEVELOPER
                )
            )
        else:
            raise ValueError(f"Unsupported role: {role}.")
    
    return message_list

