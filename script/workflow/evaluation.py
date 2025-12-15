"""
Evaluate subagent research on HotpotQA validation questions.

Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa
"""

from __future__ import annotations

import argparse
import json
import os
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
from camel.toolkits import FunctionTool

from rosetta.context.track import InteractionTracker
from rosetta.workflow.research_flow import direct_subagent_research, extend_subagent_research
from rosetta.workflow.evaluation import extract_answer, exact_match, load_done_ids
from rosetta.workflow.retriever import search_engine


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
    parser.add_argument("--mode", default="direct", choices=["direct", "extended"])
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
        done_ids: set[str] = load_done_ids(out_path) if args.resume else set()
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    search_tool = FunctionTool(search_engine)


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
            # reset main agent memory
            main_agent = ChatAgent(system_message="You are a helpful assistant.", model=model)
            
            t0 = time.time()
            pred_raw = ""
            pred = ""
            llm0_messages: Optional[list[dict[str, Any]]] = None
            err: Optional[str] = None
            try:
                if args.mode == "direct":
                    research_func = direct_subagent_research
                elif args.mode == "extended":
                    research_func = extend_subagent_research
                else:
                    raise ValueError(f"Invalid mode: {args.mode}")
                pred_raw, tracker = research_func(
                    question=question,
                    main_agent=main_agent,
                    search_model=model, 
                    tracker=tracker,
                    search_tool=search_tool
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

