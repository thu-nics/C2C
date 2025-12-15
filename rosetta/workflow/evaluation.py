"""
Evaluation utilities for workflow research tasks.

Provides functions for answer extraction, normalization, and exact match evaluation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_ANSWER_LINE_RE = re.compile(
    r'^\s*"answer"\s*:\s*"(?P<ans>.*?)"\s*(?:,|\})\s*$',
    flags=re.DOTALL,
)


def extract_answer(pred_raw: str) -> Optional[str]:
    """Extract answer from model output.

    Preferred format: a single-line JSON object:
      {"answer":"...","justification":"..."}

    Fallbacks:
    - parse JSON anywhere in text
    - parse a line containing: "answer": "..."
    - parse 'Final Answer: ...'
    """
    if not pred_raw:
        return None

    s = pred_raw.strip()

    # 1) Strict JSON object line (best case)
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                return obj["answer"].strip()
        except json.JSONDecodeError:
            pass

    # 2) First {...} blob in the text
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        blob = s[first : last + 1]
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                return obj["answer"].strip()
        except json.JSONDecodeError:
            pass

    # 3) Look for an "answer" field line
    for line in s.splitlines():
        m = _ANSWER_LINE_RE.match(line)
        if m:
            return m.group("ans").strip()

    # 4) Last resort: "Final Answer: ..."
    for line in reversed(s.splitlines()):
        if "final answer" in line.lower():
            _, _, tail = line.partition(":")
            tail = tail.strip()
            if tail:
                return tail

    return None


def _normalize_answer(s: str) -> str:
    """HotpotQA/SQuAD-style normalization for EM."""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> bool:
    """Check if predicted answer exactly matches gold answer after normalization."""
    return _normalize_answer(pred) == _normalize_answer(gold)


def load_done_ids(jsonl_path: Path) -> set[str]:
    """Load set of completed IDs from a JSONL file."""
    if not jsonl_path.exists():
        return set()
    done = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "hotpot_id" in obj:
                    done.add(str(obj["hotpot_id"]))
            except json.JSONDecodeError:
                continue
    return done

