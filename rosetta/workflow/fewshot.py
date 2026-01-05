"""Few-shot example manager for worker agents."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any


class FewShotManager:
    """Manages few-shot examples for worker agents from JSON file.

    Loads from best.json produced by FeedbackAgent.best_json(), which contains
    full message objects sorted by delta (highest quality first).
    """

    def __init__(self, json_path: str = "local/data/fewshot/best.json"):
        """Initialize the few-shot manager.

        Args:
            json_path: Path to the JSON file containing few-shot examples.
                      Defaults to best.json produced by FeedbackAgent.
        """
        self.json_path = Path(json_path)
        self.examples: List[Dict[str, Any]] = []
        self._load_examples()

    def _load_examples(self) -> None:
        """Load examples from JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Few-shot JSON file not found: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        for record in records:
            self.examples.append({
                'task': record.get('task', ''),
                'step_idx': record.get('step_idx', 0),
                'tool_idx': record.get('tool_idx', 0),
                # Store full message objects
                'user': record.get('user', {}),
                'query': record.get('query', {}),
                'tool': record.get('tool', {}),
                'next': record.get('next', {}),
                # Store scores for reference
                'scores': record.get('scores', {}),
                'model_tag': record.get('model_tag', ''),
                'score_sum': record.get('score_sum', 0),
                'delta': record.get('delta', 0),
            })

    def get_examples(
        self,
        n: int = 3,
        selection: str = "first",
        seed: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get n few-shot examples.

        Args:
            n: Number of examples to return.
            selection: Selection strategy - "first", "random", or "last".
            seed: Random seed for reproducible random selection.

        Returns:
            List of example dicts with task, user_content, query_content,
            tool_content, and next_content fields.
        """
        if n <= 0:
            return []

        n = min(n, len(self.examples))

        if selection == "first":
            return self.examples[:n]
        elif selection == "last":
            return self.examples[-n:]
        elif selection == "random":
            if seed is not None:
                random.seed(seed)
            return random.sample(self.examples, n)
        else:
            raise ValueError(f"Unknown selection strategy: {selection}")

    def format_as_chat_history(
        self,
        n: int = 3,
        selection: str = "first",
        seed: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Format few-shot examples as chat history messages.

        Args:
            n: Number of examples to include.
            selection: Selection strategy - "first", "random", or "last".
            seed: Random seed for reproducible random selection.

        Returns:
            List of message dicts with 'role' and 'content' keys, suitable
            for adding to ChatAgent chat_history.
        """
        examples = self.get_examples(n, selection, seed)
        messages = []

        for example in examples:
            # Extract full message objects
            user_msg = example['user']
            query_msg = example['query']
            tool_msg = example['tool']
            next_msg = example['next']

            # Get tool_call_id and func_name from query message
            tool_calls = query_msg.get('tool_calls', [])
            if tool_calls:
                tool_call_id = tool_calls[0].get('id')
                func_name = tool_calls[0].get('function', {}).get('name')
            else:
                tool_call_id = None
                func_name = None

            # Add messages in sequence
            messages.append(user_msg)
            messages.append(query_msg)

            # Add func_name to tool message for proper conversion
            tool_msg_with_func = dict(tool_msg)
            if func_name and 'func_name' not in tool_msg_with_func:
                tool_msg_with_func['func_name'] = func_name
            messages.append(tool_msg_with_func)

            messages.append(next_msg)

        return messages

    def add_to_agent_memory(
        self,
        agent,
        n: int = 3,
        selection: str = "first",
        seed: Optional[int] = None
    ) -> None:
        """Add few-shot examples to a ChatAgent's memory.

        Args:
            agent: ChatAgent instance to add examples to.
            n: Number of examples to include.
            selection: Selection strategy - "first", "random", or "last".
            seed: Random seed for reproducible random selection.
        """
        from rosetta.workflow.camel_utils import messages_to_memoryRecords

        messages = self.format_as_chat_history(n, selection, seed)
        memory_records = messages_to_memoryRecords(messages)
        agent.memory.write_records(memory_records)

    @property
    def num_examples(self) -> int:
        """Get total number of available examples."""
        return len(self.examples)
