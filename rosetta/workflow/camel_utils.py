"""Utility functions for CAMEL message conversion."""

from typing import List

from camel.messages import BaseMessage
from camel.memories import MemoryRecord, ContextRecord
from camel.types import OpenAIBackendRole, RoleType

def records_to_camel_messages(
    records: List[ContextRecord]
) -> List[MemoryRecord]:
    """Convert ContextRecord list to standard message format.
    
    Args:
        records: List of MemoryRecord objects.
    """
    return [ctx_record.memory_record for ctx_record in records]


def messages_to_camel_messages(
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

