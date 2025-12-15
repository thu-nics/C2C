"""
Evaluate subagent research on HotpotQA validation questions.

Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from rosetta.context.track import InteractionTracker
from rosetta.workflow.research_flow import direct_subagent_research


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
    return _normalize_answer(pred) == _normalize_answer(gold)


@dataclass
class EvalRecord:
    idx: int
    hotpot_id: str
    question: str
    gold_answer: str
    pred_answer: str
    pred_raw: str
    llm0_messages: Optional[list[dict[str, Any]]]
    correct_em: bool
    seconds: float
    error: Optional[str] = None


def _load_done_ids(jsonl_path: Path) -> set[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="local/evaluation/direct/hotpotqa.jsonl")
    parser.add_argument(
        "--output-format",
        default="json",
        choices=["jsonl", "json"],
        help=(
            "Output format. 'jsonl' writes one compact JSON object per line (best for streaming/resume). "
            "'json' writes a single pretty-printed JSON array (more readable)."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip items already in output JSONL")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-type", default="contextual-model")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    args = parser.parse_args()

    # Environment variables (search tools)
    # NOTE: This project stores keys in a local file; keep behavior consistent with subagent_research.py.
    from rosetta.workflow.API import FIRECRAWL_API_KEY, GOOGLE_API_KEY, SEARCH_ENGINE_ID

    os.environ["FIRECRAWL_API_KEY"] = FIRECRAWL_API_KEY
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["SEARCH_ENGINE_ID"] = SEARCH_ENGINE_ID

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "jsonl":
        done_ids: set[str] = _load_done_ids(out_path) if args.resume else set()
        json_records: Optional[list[dict[str, Any]]] = None
    else:
        # JSON array mode: read existing file (if resume), append, then rewrite once at the end.
        json_records = []
        done_ids = set()
        if args.resume and out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    json_records = existing
                    for obj in existing:
                        if isinstance(obj, dict) and "hotpot_id" in obj:
                            done_ids.add(str(obj["hotpot_id"]))
            except json.JSONDecodeError:
                # If user passed resume but file isn't valid JSON, start fresh.
                json_records = []
                done_ids = set()

    # Load HF dataset slice
    split = args.split
    if args.limit is not None and args.limit > 0:
        split = f"{split}[:{args.limit}]"
    ds = load_dataset("hotpotqa/hotpot_qa", args.subset, split=split)

    # Model + agent
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=args.model_type,
        model_config_dict={
            "temperature": 0.0,
            "max_tokens": 32768,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        api_key="not-needed",
        url=args.model_url,
    )
    main_agent = ChatAgent(system_message="You are a helpful assistant.", model=model)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    total = 0
    correct = 0

    jsonl_f = out_path.open("a", encoding="utf-8") if args.output_format == "jsonl" else None
    try:
        for idx, ex in enumerate(tqdm(ds, desc="Evaluating HotpotQA")):
            hotpot_id = str(ex["id"])
            if hotpot_id in done_ids:
                continue

            question = ex["question"]
            gold = ex["answer"]

            tracker = InteractionTracker(tokenizer=tokenizer)
            t0 = time.time()
            pred_raw = ""
            pred = ""
            llm0_messages: Optional[list[dict[str, Any]]] = None
            err: Optional[str] = None
            try:
                pred_raw, tracker = direct_subagent_research(
                    question=question,
                    main_agent=main_agent,
                    search_model=model,
                    tracker=tracker
                )
                extracted = extract_answer(pred_raw)
                pred = extracted if extracted is not None else pred_raw.strip()
                try:
                    llm0_messages = tracker.get_messages(llm_id=0) if tracker is not None else None
                except Exception:  # noqa: BLE001
                    llm0_messages = None
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            seconds = time.time() - t0

            is_correct = exact_match(pred, gold) if err is None else False
            total += 1
            correct += int(is_correct)

            rec = EvalRecord(
                idx=idx,
                hotpot_id=hotpot_id,
                question=question,
                gold_answer=gold,
                pred_answer=pred,
                pred_raw=pred_raw,
                llm0_messages=llm0_messages,
                correct_em=is_correct,
                seconds=seconds,
                error=err,
            )
            rec_dict = asdict(rec)
            if args.output_format == "jsonl":
                assert jsonl_f is not None
                jsonl_f.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
                jsonl_f.flush()
            else:
                assert json_records is not None
                json_records.append(rec_dict)

    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    if args.output_format == "json":
        assert json_records is not None
        out_path.write_text(
            json.dumps(json_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    acc = (correct / total) if total else 0.0
    print(f"Done. evaluated={total} EM={correct} acc={acc:.3f} output={out_path}")


if __name__ == "__main__":
    main()

