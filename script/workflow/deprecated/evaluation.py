"""
Evaluate subagent research on HotpotQA validation questions.

Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool

from rosetta.workflow.selector import ContextSelector
from rosetta.workflow.track import InteractionTracker
from rosetta.workflow.oneflow import do_research
from rosetta.workflow.research_flow import direct_subagent_research, extend_subagent_research, full_subagent_research, extend_sequential_subagent_research
from rosetta.workflow.evaluation import extract_answer, exact_match, load_done_ids
from rosetta.workflow.retriever import search_engine
from rosetta.workflow.hf_qwen_model import HFQwenModel, HFContextAttentionQwenModel


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


def _jsonl_sidecar_path(out_path: Path) -> Path:
    """Pick a JSONL path used for incremental writes during a run."""
    suf = out_path.suffix.lower()
    if suf == ".jsonl":
        return out_path
    if suf == ".json":
        return out_path.with_suffix(".jsonl")
    # Fallback: keep original name and add a .jsonl suffix
    return out_path.with_name(out_path.name + ".jsonl")


def _jsonl_to_json_array(jsonl_path: Path, json_path: Path) -> None:
    """Convert JSONL (one JSON object per line) into a pretty-printed JSON array."""
    first = True
    with jsonl_path.open("r", encoding="utf-8") as fin, json_path.open("w", encoding="utf-8") as fout:
        fout.write("[\n")
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not first:
                fout.write(",\n")
            first = False
            obj_str = json.dumps(obj, ensure_ascii=False, indent=2)
            # Indent objects by 2 spaces inside the array
            fout.write("\n".join("  " + s for s in obj_str.splitlines()))
        fout.write("\n]\n")


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
            "'json' streams to a sidecar JSONL during the run, then writes a single pretty-printed JSON array at the end."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip items already written (JSONL output, or the JSON sidecar JSONL when --output-format json).",
    )
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-type", default="contextual-model")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    parser.add_argument("--mode", default="direct", choices=["direct", "extend", "full", "extend_sequential", "oneflow"])
    args = parser.parse_args()  

    # Environment variables (search tools)
    load_dotenv(find_dotenv())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_jsonl_path = out_path if args.output_format == "jsonl" else _jsonl_sidecar_path(out_path)

    done_ids: set[str] = set()
    if args.resume:
        # Prefer the incremental JSONL (works for both output modes).
        done_ids = load_done_ids(run_jsonl_path)
        # Fallback: if user asked for JSON output and only the JSON exists, parse it once.
        if not done_ids and args.output_format == "json" and out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    for obj in existing:
                        if isinstance(obj, dict) and "hotpot_id" in obj:
                            done_ids.add(str(obj["hotpot_id"]))
            except json.JSONDecodeError:
                pass

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
    search_model = HFContextAttentionQwenModel(
        "Qwen/Qwen3-32B",
        model_config_dict={"temperature": 0.0, "max_tokens": 32768},
        enable_thinking=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    search_tool = FunctionTool(search_engine)


    total = 0
    correct = 0

    # Always stream records during the run to JSONL (output file or sidecar).
    jsonl_f = run_jsonl_path.open("a", encoding="utf-8")
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
                elif args.mode == "extend":
                    research_func = extend_subagent_research
                elif args.mode == "full":
                    research_func = full_subagent_research
                elif args.mode == "extend_sequential":
                    research_func = extend_sequential_subagent_research
                elif args.mode == "oneflow":
                    research_func = do_research
                else:
                    raise ValueError(f"Invalid mode: {args.mode}")

                context_plan = {
                    'search_to_main_selector': ContextSelector(
                        filter_fn=ContextSelector.filter_search_only,
                        select_fn=ContextSelector.select_query_response
                    ),
                    'main_to_search_selector': ContextSelector(
                        filter_fn=None,
                        # select_fn=ContextSelector.select_none
                        # select_fn=ContextSelector.select_skip_system
                        select_fn=ContextSelector.select_initial
                    )
                }
                pred_raw, tracker = research_func(
                    question=question,
                    main_agent=main_agent,
                    search_model=search_model,
                    tracker=tracker,
                    search_tools=[search_tool],
                    context_plan=context_plan,
                    **({"show_status": False} if args.mode == "oneflow" else {}),
                    drop=True,
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
            jsonl_f.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
            jsonl_f.flush()

    finally:
        jsonl_f.close()

    if args.output_format == "json":
        _jsonl_to_json_array(run_jsonl_path, out_path)

    acc = (correct / total) if total else 0.0
    summary_line = f"evaluated={total} EM={correct} acc={acc:.3f} output={out_path}"
    summary_path = out_path.parent / "summary.txt"
    summary_path.write_text(summary_line + "\n", encoding="utf-8")
    print(f"Done. {summary_line} summary={summary_path}")


if __name__ == "__main__":
    main()
