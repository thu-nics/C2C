"""
Evaluation utilities for workflow research tasks.

Provides functions for answer extraction, normalization, exact match evaluation,
and LLM-based answer evaluation with error categorization.
"""

from __future__ import annotations

import json
import ast
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from camel.agents import ChatAgent

from rosetta.workflow.singleflow import single_research
from rosetta.workflow.prompt import (
    ERROR_CATEGORIES,
    ERROR_CATEGORIZATION_PROMPT,
    LLM_JUDGE_SYSTEM,
    LLM_JUDGE_PROMPT,
)
from rosetta.workflow.singletool import run_with_tools
from rosetta.workflow.contextManage import ContextManager
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from camel.agents import ChatAgent
    from camel.models import BaseModelBackend
    from camel.toolkits import FunctionTool
    from rosetta.workflow.track import InteractionTracker

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
        # Try Python dict literal (single quotes)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                return obj["answer"].strip()
        except Exception:
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
        try:
            obj = ast.literal_eval(blob)
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                return obj["answer"].strip()
        except Exception:
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


def _normalize_answer(s: Optional[str]) -> str:
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


def run_research(
    mode: str,
    *,
    question: str,
    main_model: "BaseModelBackend",
    tracker: Optional["InteractionTracker"] = None,
    search_model: Optional["BaseModelBackend"] = None,
    think_model: Optional["BaseModelBackend"] = None,
    search_tools: Optional[List["FunctionTool"]] = None,
    context_plan: Optional[dict] = None,
    show_status: bool = True,
    worker_model: Optional["BaseModelBackend"] = None,
    rewind_model: Optional["BaseModelBackend"] = None,
    exam_model: Optional["BaseModelBackend"] = None,
    worker_tools: Optional[List["FunctionTool"]] = None,
    tree_tracker: Optional[object] = None,
    max_rounds: int = 10,
    state_rule_actions: Optional[List[str]] = None,
    main_agent_tools: Optional[List["FunctionTool"]] = None,
    step_timeout: Optional[float] = None,
    tokenizer: Optional[object] = None,
) -> Tuple[str, Optional["InteractionTracker"]]:
    """Dispatch to the requested research workflow."""
    mode = mode.lower()
    if mode == "single":
        agent_kwargs = {
            "system_message": "You are a helpful assistant.",
            "model": main_model,
            "tools": main_agent_tools,
        }
        if step_timeout is not None:
            agent_kwargs["step_timeout"] = step_timeout
        main_agent = ChatAgent(**agent_kwargs)
        return single_research(
            question=question,
            main_agent=main_agent,
            tracker=tracker,
            context_plan=context_plan,
        )
    if mode == "oneflow":
        if search_model is None:
            raise ValueError("search_model is required for mode='oneflow'")
        from rosetta.workflow.oneflow import do_research

        agent_kwargs = {
            "system_message": "You are a helpful assistant.",
            "model": main_model,
        }
        if step_timeout is not None:
            agent_kwargs["step_timeout"] = step_timeout
        main_agent = ChatAgent(**agent_kwargs)
        return do_research(
            question=question,
            main_agent=main_agent,
            search_model=search_model,
            tracker=tracker,
            search_tools=search_tools,
            context_plan=context_plan,
            show_status=show_status,
        )
    if mode == "tree":
        from rosetta.workflow.treeflow import do_tree_research

        return do_tree_research(
            question=question,
            main_model=main_model,
            worker_model=worker_model,
            rewind_model=rewind_model,
            exam_model=exam_model,
            think_model=think_model,
            state_rule_actions=state_rule_actions,
            tracker=tracker,
            tree_tracker=tree_tracker,
            worker_tools=worker_tools,
            max_rounds=max_rounds,
            show_status=show_status,
        )
    if mode == "tool":
        from rosetta.workflow.toolflow import do_tool_research
        from rosetta.workflow.rules import ActionRuleEnforcer, ActionRule

        action_rule_enforcer = ActionRuleEnforcer(
            ActionRule.require_initial_plan_or_think,
            ActionRule.require_continue_before_answer,
            ActionRule.break_on_consecutive_continue,
        )

        return do_tool_research(
            question=question,
            main_model=main_model,
            worker_model=worker_model,
            rewind_model=rewind_model,
            exam_model=exam_model,
            think_model=think_model,
            state_rule_actions=state_rule_actions,
            action_rule_enforcer=action_rule_enforcer,
            tracker=tracker,
            tree_tracker=tree_tracker,
            worker_tools=worker_tools,
            max_rounds=max_rounds,
            show_status=show_status,
        )
    if mode == "singletool":
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(main_model.model_type))
        ctx_manager = None
        ctx_manager = ContextManager(main_model, tokenizer=tokenizer)

        return run_with_tools(
            question=question,
            model=main_model,
            tools=main_agent_tools or [],
            tracker=tracker,
            max_iterations=max_rounds,
            ctx_manager=ctx_manager,
        )
    raise ValueError(f"Unsupported mode: {mode}")


class LLMJudge:
    """LLM-based judge for answer correctness and error categorization.

    Args:
        model: CAMEL model backend (created via create_model).
        max_workers: Maximum number of parallel workers for batch operations.
    """

    def __init__(self, model: "BaseModelBackend", max_workers: int = 32):
        self._model = model
        self._max_workers = max_workers

    def _run(self, system: str, prompt: str) -> str:
        """Run a prompt through the agent and return the response content."""
        agent = ChatAgent(system_message=system, model=self._model)
        response = agent.step(prompt)
        return response.msgs[0].content

    def judge_answer(self, rec: dict) -> Tuple[dict, bool]:
        """Judge if predicted answer matches gold answer.

        Args:
            rec: Record with question, gold_answer, pred_answer fields.

        Returns:
            Tuple of (updated record with correct_llm field, is_correct).
        """
        prompt = LLM_JUDGE_PROMPT.format(
            question=rec["question"],
            gold_answer=rec["gold_answer"],
            pred_answer=rec["pred_answer"],
        )
        try:
            content = self._run(LLM_JUDGE_SYSTEM, prompt)
            # Strip <think>...</think> blocks and extract JSON
            if "</think>" in content:
                content = content.split("</think>")[-1]
            start, end = content.find("{"), content.rfind("}")
            result = json.loads(content[start : end + 1])
            is_correct = result.get("verdict", False)
            rec["correct_llm"] = is_correct
            rec["judge_confidence"] = result.get("confidence", "unknown")
            rec["judge_reason"] = result.get("brief_reason", "")
            return rec, is_correct
        except Exception:
            rec["correct_llm"] = False
            return rec, False

    def categorize_error(self, rec: dict) -> Tuple[dict, str]:
        """Categorize the error reason for an incorrect answer.

        Args:
            rec: Record with question, gold_answer, pred_answer, llm0_messages fields.

        Returns:
            Tuple of (updated record with error_category field, category string).
        """
        # Format chat history
        chat_history = ""
        if rec.get("llm0_messages"):
            for i, msg in enumerate(rec["llm0_messages"], 1):
                role = msg["role"].upper()
                content = msg["content"]
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"
                chat_history += f"[{i}] {role}:\n{content}\n\n"

        category_list = "\n".join([
            f"{i}. **{cat}** - {desc}"
            for i, (cat, desc) in enumerate(ERROR_CATEGORIES.items(), 1)
        ])

        prompt = ERROR_CATEGORIZATION_PROMPT.format(
            question=rec["question"],
            gold_answer=rec["gold_answer"],
            pred_answer=rec["pred_answer"],
            chat_history=chat_history or "No chat history available",
            category_list=category_list,
        )

        try:
            content = self._run(
                "You are an expert judge analyzing LLM research workflow failures.",
                prompt,
            )
            # Strip <think>...</think> blocks
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            category = "Unknown"
            for cat in ERROR_CATEGORIES.keys():
                if cat.lower() in content.lower():
                    category = cat
                    break
            rec["error_category"] = category
            return rec, category
        except Exception:
            rec["error_category"] = "Unknown"
            return rec, "Unknown"

    def judge_batch(
        self,
        records: List[dict],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[dict], int, int]:
        """Judge answer correctness for a batch of records in parallel.

        Args:
            records: List of record dicts to judge.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            Tuple of (updated records, total judged, correct count).
        """
        to_judge = [rec for rec in records if rec.get("error") is None]
        for rec in records:
            if rec.get("error") is not None:
                rec["correct_llm"] = False

        total, correct, completed = len(to_judge), 0, 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(self.judge_answer, rec): rec for rec in to_judge}
            for future in as_completed(futures):
                try:
                    _, is_correct = future.result()
                    correct += int(is_correct)
                except Exception:
                    pass
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return records, total, correct

    def categorize_batch(
        self,
        records: List[dict],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[dict], Dict[str, List[dict]]]:
        """Categorize errors for a batch of incorrect records in parallel.

        Args:
            records: List of records (should have correct_llm field set).
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            Tuple of (updated records, category_counts dict).
        """
        # Initialize categories
        for rec in records:
            if rec.get("error") is not None:
                rec["error_category"] = "System Error"
            elif rec.get("correct_llm", False):
                rec["error_category"] = "N/A"

        incorrect = [rec for rec in records if not rec.get("correct_llm", False) and rec.get("error") is None]
        category_counts: Dict[str, List[dict]] = {}
        total, completed = len(incorrect), 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(self.categorize_error, rec): rec for rec in incorrect}
            for future in as_completed(futures):
                try:
                    rec, category = future.result()
                    category_counts.setdefault(category, []).append(rec)
                except Exception:
                    pass
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return records, category_counts
