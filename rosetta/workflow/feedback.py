"""Feedback agent for evaluating search agent tool call triplets."""

import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.storages.key_value_storages import JsonStorage


# Default path for feedback storage
DEFAULT_FEEDBACK_PATH = Path("local/data/fewshot/feedback.json")


@dataclass
class TripletScore:
    """Score for user-query-tool-next interaction.

    Evaluates 3 pairs:
    - user → query: How well assistant formulated the tool call from user task
    - query → tool: Quality of tool call and its result
    - tool → next: How well assistant interpreted the result
    """
    query_score: int   # 1-5: user → query (formulation quality)
    tool_score: int    # 1-5: query → tool (result relevance)
    next_score: int    # 1-5: tool → next (interpretation quality)
    reasoning: str


FEEDBACK_SYSTEM_PROMPT = """You evaluate search agent interactions by scoring three transitions:

1. **query_score** (User → Query): How well did the assistant formulate the tool call based on the user's task?
   - 5: Highly specific query, directly addresses the task
   - 3: Adequate but could be more targeted
   - 1: Vague, off-topic, or poorly formulated

2. **tool_score** (Query → Tool Result): How useful/relevant is the tool result for the query?
   - 5: Comprehensive, accurate, directly answers the query
   - 3: Partial information, somewhat relevant
   - 1: Irrelevant, error, or no useful information

3. **next_score** (Tool Result → Next Step): How well did the assistant interpret and use the result?
   - 5: Correctly interprets, no hallucination, proper synthesis
   - 3: Mostly correct but minor misinterpretation
   - 1: Hallucination, ignores results, or wrong conclusions"""


FEEDBACK_PROMPT = """Evaluate this search agent interaction:

[User Task]
{user_content}

[Assistant Tool Call]
{query_content}

[Tool Result]
{tool_content}

[Assistant Response]
{next_content}

Score each transition (1-5):
- query_score: User → Query (formulation quality)
- tool_score: Query → Tool Result (result relevance)
- next_score: Tool Result → Response (interpretation quality)

Return JSON only:
{{"query_score": <1-5>, "tool_score": <1-5>, "next_score": <1-5>, "reasoning": "<brief explanation>"}}"""


class FeedbackAgent:
    """Agent for evaluating search tool call triplets with persistent storage."""

    def __init__(
        self,
        model: BaseModelBackend,
        storage_path: Optional[Path] = None,
    ):
        """Initialize feedback agent.

        Args:
            model: Model backend for evaluation.
            storage_path: Path for JSON storage. Defaults to local/data/fewshot/feedback.json.
        """
        self.model = model
        self.storage_path = storage_path or DEFAULT_FEEDBACK_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage = JsonStorage(self.storage_path)

    def evaluate_triplet(
        self,
        triplet: Dict[str, Any],
    ) -> TripletScore:
        """Evaluate a single triplet with one LLM call.

        Args:
            triplet: Dict with keys 'user', 'query', 'tool', 'next'.

        Returns:
            TripletScore object.
        """
        user_content = self._format_message(triplet["user"])
        query_content = self._format_message(triplet["query"])
        tool_content = self._format_message(triplet["tool"])
        next_content = self._format_message(triplet["next"])

        prompt = FEEDBACK_PROMPT.format(
            user_content=user_content,
            query_content=query_content,
            tool_content=tool_content,
            next_content=next_content,
        )

        # Create fresh agent for each evaluation
        agent = ChatAgent(
            system_message=FEEDBACK_SYSTEM_PROMPT,
            model=self.model,
        )
        response = agent.step(prompt)

        return self._parse_single_score(response.msg.content)

    @staticmethod
    def extract_triplets(raw_chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract user-query-tool-next triplets from raw chat history.

        For each tool message at index i, extracts:
        - user: message at i-2 (user task)
        - query: message at i-1 (assistant tool call)
        - tool: message at i (tool result)
        - next: message at i+1 (assistant response)

        Args:
            raw_chat_history: List of chat message dicts with 'role' key.

        Returns:
            List of triplet dicts with keys 'user', 'query', 'tool', 'next', 'tool_idx'.
            tool_idx is a sequential counter (0, 1, 2, ...) for tool calls in this history.
        """
        tool_ids = [i for i, m in enumerate(raw_chat_history) if m.get("role") == "tool"]
        triplets = []
        for seq_idx, i in enumerate(tool_ids):
            # Need 2 messages before (user, query) and 1 after (next)
            if i >= 2 and i + 1 < len(raw_chat_history):
                triplets.append({
                    "user": raw_chat_history[i - 2],
                    "query": raw_chat_history[i - 1],
                    "tool": raw_chat_history[i],
                    "next": raw_chat_history[i + 1],
                    "tool_idx": seq_idx,  # Sequential counter, not raw index
                })
        return triplets

    def evaluate(
        self,
        triplets: List[Dict[str, Any]],
        task: str,
        step_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """Evaluate triplets and return scored records.

        Args:
            triplets: List of triplet dicts with keys 'user', 'query', 'tool', 'next', 'tool_idx'.
            task: Original subtask for context (used for storage key).
            step_idx: Current step index.

        Returns:
            List of scored record dicts.
        """
        if not triplets:
            return []

        scored_records = []
        for triplet in triplets:
            score = self.evaluate_triplet(triplet)
            scored_records.append({
                "step_idx": step_idx,
                "tool_idx": triplet.get("tool_idx", 0),
                "task": task,
                "user": triplet["user"],
                "query": triplet["query"],
                "tool": triplet["tool"],
                "next": triplet["next"],
                "scores": asdict(score),
            })

        return scored_records

    def store(self, records: List[Dict[str, Any]]) -> None:
        """Store scored records to JSON, appending only new ones.

        Args:
            records: List of scored record dicts from evaluate().
        """
        if not records:
            return

        existing_records = self._storage.load() or []
        existing_keys = {
            (r.get("task"), r.get("step_idx"), r.get("tool_idx"))
            for r in existing_records
        }

        new_records = []
        for new_record in records:
            key = (
                new_record.get("task"),
                new_record.get("step_idx"),
                new_record.get("tool_idx"),
            )
            if key not in existing_keys:
                new_records.append(new_record)
                existing_keys.add(key)

        if new_records:
            self._storage.save(new_records)

    def _format_message(self, msg: Dict[str, Any]) -> str:
        """Format a chat message for display."""
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle tool_calls in assistant messages
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            calls_str = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                calls_str.append(f"  {name}({args})")
            content = "Tool calls:\n" + "\n".join(calls_str)
            if msg.get("content"):
                content = msg["content"] + "\n" + content

        # Truncate long content
        max_len = 2000
        if len(content) > max_len:
            content = content[:max_len] + "... [truncated]"

        return content

    def _parse_single_score(self, response: str) -> TripletScore:
        """Parse LLM response into a single TripletScore object."""
        # Extract JSON object from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            return TripletScore(
                query_score=-1, tool_score=-1, next_score=-1, reasoning="Parse error"
            )

        try:
            s = json.loads(json_match.group())
        except json.JSONDecodeError:
            return TripletScore(
                query_score=-1, tool_score=-1, next_score=-1, reasoning="JSON parse error"
            )

        return TripletScore(
            query_score=self._clamp_score(s.get("query_score", -1)),
            tool_score=self._clamp_score(s.get("tool_score", -1)),
            next_score=self._clamp_score(s.get("next_score", -1)),
            reasoning=s.get("reasoning", ""),
        )

    @staticmethod
    def _clamp_score(score: Any) -> int:
        """Clamp score to 1-5 range, or -1 for errors."""
        try:
            val = int(score)
            if val == -1:
                return -1
            return max(1, min(5, val))
        except (ValueError, TypeError):
            return -1

    @staticmethod
    def to_csv(
        json_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Convert stored JSON records to CSV.

        Args:
            json_path: Input JSON path. Defaults to DEFAULT_FEEDBACK_PATH.
            output_path: Output CSV path. Defaults to same path as JSON with .csv extension.

        Returns:
            Path to the created CSV file.
        """
        if json_path is None:
            json_path = DEFAULT_FEEDBACK_PATH
        if output_path is None:
            output_path = json_path.with_suffix(".csv")

        storage = JsonStorage(json_path)
        records = storage.load() or []

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step_idx", "tool_idx", "task",
                "user_content", "query_content", "tool_content", "next_content",
                "query_score", "tool_score", "next_score", "reasoning"
            ])

            for record in records:
                scores = record.get("scores", {})
                writer.writerow([
                    record.get("step_idx", ""),
                    record.get("tool_idx", ""),
                    record.get("task", ""),
                    FeedbackAgent._extract_content(record.get("user", {})),
                    FeedbackAgent._extract_content(record.get("query", {})),
                    FeedbackAgent._extract_content(record.get("tool", {})),
                    FeedbackAgent._extract_content(record.get("next", {})),
                    scores.get("query_score", ""),
                    scores.get("tool_score", ""),
                    scores.get("next_score", ""),
                    scores.get("reasoning", ""),
                ])

        return output_path

    @staticmethod
    def best_csv(
        csv_paths: Optional[List[Path]] = None,
        output_path: Optional[Path] = None,
        key_fields: Tuple[str, ...] = ("task", "step_idx", "tool_idx"),
    ) -> Path:
        """Select best-scoring rows across multiple CSVs and write best.csv.

        Args:
            csv_paths: Input CSV paths. Defaults to all *.csv in fewshot dir,
                excluding best.csv.
            output_path: Output CSV path. Defaults to fewshot dir / best.csv.
            key_fields: Columns used to align rows across models.

        Returns:
            Path to the created CSV file.
        """
        if csv_paths is None:
            csv_dir = DEFAULT_FEEDBACK_PATH.parent
            csv_paths = sorted(
                p for p in csv_dir.glob("fewshot*.csv") if p.name != "best.csv"
            )
        if not csv_paths:
            raise ValueError("No CSV paths provided or found.")
        if output_path is None:
            output_path = DEFAULT_FEEDBACK_PATH.parent / "best.csv"

        def _safe_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1

        def _score_sum(row: Dict[str, Any]) -> int:
            return (
                _safe_int(row.get("query_score"))
                + _safe_int(row.get("tool_score"))
                + _safe_int(row.get("next_score"))
            )

        def _model_tag_from_path(path: Path) -> str:
            stem = path.stem
            if stem.startswith("fewshot_"):
                return stem[len("fewshot_") :]
            return stem

        best_by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        sums_by_key: Dict[Tuple[str, ...], List[int]] = {}
        order: List[Tuple[str, ...]] = []
        headers: Optional[List[str]] = None

        for path in csv_paths:
            with Path(path).open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if headers is None:
                    headers = reader.fieldnames or []
                for row in reader:
                    key = tuple((row.get(k) or "") for k in key_fields)
                    if key not in sums_by_key:
                        sums_by_key[key] = []
                        order.append(key)
                    score_sum = _score_sum(row)
                    sums_by_key[key].append(score_sum)
                    current = best_by_key.get(key)
                    if current is None or score_sum > current["score_sum"]:
                        best_by_key[key] = {
                            "row": row,
                            "score_sum": score_sum,
                            "source": Path(path).stem,
                            "model_tag": _model_tag_from_path(Path(path)),
                        }

        headers = headers or []
        model_tag_col = "model_tag"
        if model_tag_col in headers:
            model_tag_col = "best_model_tag"
        source_col = "source"
        if source_col in headers:
            source_col = "source_file"
        score_sum_col = "score_sum"
        if score_sum_col in headers:
            score_sum_col = "best_score_sum"
        delta_col = "delta"
        if delta_col in headers:
            delta_col = "best_minus_avg"

        output_headers = headers + [model_tag_col, source_col, score_sum_col, delta_col]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_headers)
            writer.writeheader()
            for key in order:
                best = best_by_key.get(key)
                if best is None:
                    continue
                row = dict(best["row"])
                best_sum = best["score_sum"]
                avg_sum = sum(sums_by_key[key]) / len(sums_by_key[key])
                delta = best_sum - avg_sum
                row[model_tag_col] = best["model_tag"]
                row[source_col] = best["source"]
                row[score_sum_col] = str(best_sum)
                row[delta_col] = f"{delta:.3f}"
                writer.writerow(row)

        return output_path

    @staticmethod
    def best_json(
        json_paths: Optional[List[Path]] = None,
        output_path: Optional[Path] = None,
        key_fields: Tuple[str, ...] = ("task", "step_idx", "tool_idx"),
    ) -> Path:
        """Select best-scoring entries across multiple JSON files and write best.json.

        Entries are sorted by delta (descending), so highest-scoring examples appear first.

        Args:
            json_paths: Input JSON paths. Defaults to all fewshot_*.json in fewshot dir.
            output_path: Output JSON path. Defaults to fewshot dir / best.json.
            key_fields: Fields used to align entries across models.

        Returns:
            Path to the created JSON file.
        """
        if json_paths is None:
            json_dir = DEFAULT_FEEDBACK_PATH.parent
            json_paths = sorted(
                p for p in json_dir.glob("fewshot_*.json") if p.name != "best.json"
            )
        if not json_paths:
            raise ValueError("No JSON paths provided or found.")
        if output_path is None:
            output_path = DEFAULT_FEEDBACK_PATH.parent / "best.json"

        def _safe_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1

        def _score_sum(record: Dict[str, Any]) -> int:
            scores = record.get("scores", {})
            return (
                _safe_int(scores.get("query_score"))
                + _safe_int(scores.get("tool_score"))
                + _safe_int(scores.get("next_score"))
            )

        def _model_tag_from_path(path: Path) -> str:
            stem = path.stem
            if stem.startswith("fewshot_"):
                return stem[len("fewshot_") :]
            return stem

        best_by_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        sums_by_key: Dict[Tuple[str, ...], List[int]] = {}

        for path in json_paths:
            storage = JsonStorage(path)
            records = storage.load() or []
            for record in records:
                key = tuple((record.get(k) or "") for k in key_fields)
                if key not in sums_by_key:
                    sums_by_key[key] = []
                score_sum = _score_sum(record)
                sums_by_key[key].append(score_sum)
                current = best_by_key.get(key)
                if current is None or score_sum > current["score_sum"]:
                    best_by_key[key] = {
                        "record": record,
                        "score_sum": score_sum,
                        "model_tag": _model_tag_from_path(path),
                    }

        # Build final list with delta and sort by delta descending
        best_records = []
        for key, best_data in best_by_key.items():
            record = best_data["record"]
            best_sum = best_data["score_sum"]
            avg_sum = sum(sums_by_key[key]) / len(sums_by_key[key])
            delta = best_sum - avg_sum

            # Add metadata
            record["model_tag"] = best_data["model_tag"]
            record["score_sum"] = best_sum
            record["delta"] = delta
            best_records.append(record)

        # Sort by delta descending (highest delta first)
        best_records.sort(key=lambda r: r.get("delta", 0), reverse=True)

        # Save to JSON
        storage = JsonStorage(output_path)
        storage.save(best_records)

        return output_path

    @staticmethod
    def _extract_content(msg: Dict[str, Any]) -> str:
        """Extract content string from a message dict."""
        if not msg:
            return ""
        content = msg.get("content", "")
        # For assistant messages with tool_calls, include function info
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            calls_str = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                calls_str.append(f"{name}({args})")
            if calls_str:
                content = (content + " " if content else "") + "; ".join(calls_str)
        return content
