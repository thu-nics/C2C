"""
Evaluate subagent research on HotpotQA validation questions.

Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Callable

import requests
from datasets import load_dataset
from transformers import AutoTokenizer
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from dotenv import find_dotenv, load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool

from rosetta.workflow.selector import ContextSelector
from rosetta.workflow.track import InteractionTracker, TreeTracker
from rosetta.workflow.evaluation import extract_answer, exact_match, load_done_ids, run_research
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


@dataclass
class EvalConfig:
    subset: str
    split: str
    limit: int
    max_rounds: int
    output: Path
    output_format: str
    resume: bool
    model_url: str
    model_type: str
    tokenizer: str
    mode: str
    main_to_search: str
    search: str
    search_to_main: str
    main: str
    num_workers: int


def setup_env():
    """Setup environment variables."""
    load_dotenv(find_dotenv())


def create_context_plan(config: EvalConfig, use_single: bool, use_tree: bool) -> Optional[dict]:
    """Create context plan based on configuration."""
    main_to_search_select_fn: Callable = {
        "all": ContextSelector.select_skip_system,
        "initial": ContextSelector.select_initial_with_system,
        "none": ContextSelector.select_none,
    }[config.main_to_search]

    search_contextual_select_fn: Callable = {
        "all": ContextSelector.select_all,
        "initial": ContextSelector.select_initial,
        "none": ContextSelector.select_none,
    }[config.search]

    search_to_main_select_fn: Callable = {
        "all": ContextSelector.select_skip_system,
        "qr": ContextSelector.select_query_response_with_system,
    }[config.search_to_main]

    main_contextual_select_fn: Callable = {
        "all": ContextSelector.select_all,
        "qr": ContextSelector.select_query_response,
    }[config.main]

    if use_single:
        return {
            "main_contextual": ContextSelector(
                select_fn=ContextSelector.select_query_response
            ),
        }
    elif not use_tree:
        return {
            'search_to_main_selector': ContextSelector(
                filter_fn=ContextSelector.filter_search_only,
                select_fn=search_to_main_select_fn
            ),
            'main_to_search_selector': ContextSelector(
                filter_fn=None,
                select_fn=main_to_search_select_fn
            ),
            'search_contextual': ContextSelector(
                select_fn=search_contextual_select_fn
            ),
            'main_contextual': ContextSelector(
                select_fn=main_contextual_select_fn
            ),
        }
    else:
        return None


def evaluate_single(
    idx: int,
    ex: dict,
    config: EvalConfig,
    model,
    search_model,
    tokenizer,
    search_tool: FunctionTool,
) -> EvalRecord:
    """Evaluate a single example."""
    hotpot_id = str(ex["id"])
    question = ex["question"]
    gold = ex["answer"]

    use_single = config.mode == "single"
    use_tree = config.mode == "tree"

    tracker = InteractionTracker(tokenizer=tokenizer)
    tools = [search_tool] if use_single else None
    main_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        tools=tools
    )
    tree_tracker = TreeTracker() if use_tree else None
    context_plan = create_context_plan(config, use_single, use_tree)

    t0 = time.time()
    pred_raw = ""
    pred = ""
    llm0_messages: Optional[list[dict[str, Any]]] = None
    err: Optional[str] = None

    try:
        pred_raw, tracker = run_research(
            mode=config.mode,
            question=question,
            main_agent=main_agent,
            search_model=search_model if not use_single else None,
            tracker=tracker,
            search_tools=[search_tool] if not use_single and search_tool else None,
            context_plan=context_plan,
            show_status=False,
            max_rounds=config.max_rounds,
            worker_model=search_model if use_tree else None,
            rewind_model=search_model if use_tree else None,
            exam_model=search_model if use_tree else None,
            worker_tools=[search_tool] if use_tree and search_tool else None,
            tree_tracker=tree_tracker,
        )
        extracted = extract_answer(pred_raw)
        pred = extracted if extracted is not None else pred_raw.strip()
        try:
            llm0_messages = tracker.get_messages(llm_id=0) if tracker is not None else None
        except Exception:
            llm0_messages = None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    seconds = time.time() - t0
    is_correct = exact_match(pred, gold) if err is None else False

    return EvalRecord(
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


def worker_process(
    worker_id: int,
    examples: list[dict],
    config: EvalConfig,
    process_dir: Path,
) -> None:
    """Worker process that evaluates a chunk of examples."""
    setup_env()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=config.model_type,
        model_config_dict={
            "temperature": 0.0,
            "max_tokens": 32768,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        api_key="not-needed",
        url=config.model_url,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    search_tool = FunctionTool(search_engine)
    output_file = process_dir / f"worker_{worker_id}.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for ex in examples:
            rec = evaluate_single(
                idx=ex["_idx"],
                ex=ex,
                config=config,
                model=model,
                search_model=model,
                tokenizer=tokenizer,
                search_tool=search_tool,
            )
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            f.flush()


def aggregate_results(process_dir: Path, output_path: Path, output_format: str) -> tuple[int, int]:
    """Aggregate results from all worker files."""
    # Collect all worker files
    worker_files = sorted(process_dir.glob("worker_*.jsonl"))

    # Determine output paths
    if output_format == "jsonl":
        final_jsonl = output_path
    else:
        final_jsonl = output_path.with_suffix(".jsonl")

    # Aggregate into single JSONL
    total = 0
    correct = 0

    with final_jsonl.open("w", encoding="utf-8") as out_f:
        for worker_file in worker_files:
            with worker_file.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        total += 1
                        correct += int(obj.get("correct_em", False))
                    except json.JSONDecodeError:
                        continue

    return total, correct


def _evaluate_single_answer(rec: dict, api_url: str, model_type: str, prompt_template: str) -> tuple[dict, bool]:
    """Evaluate a single answer via SGLang API."""
    prompt = prompt_template.format(
        question=rec["question"],
        gold_answer=rec["gold_answer"],
        pred_answer=rec["pred_answer"],
    )

    request_data = {
        "model": model_type,
        "messages": [
            {"role": "system", "content": "You are an expert evaluator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    try:
        response = requests.post(api_url, json=request_data, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        judgment = content.strip().upper()
        is_correct = "YES" in judgment
        rec["correct_llm"] = is_correct
        return rec, is_correct
    except Exception as e:
        rec["correct_llm"] = False
        return rec, False


def llm_evaluate_answers(jsonl_path: Path, config: EvalConfig, max_workers: int = 32) -> tuple[int, int]:
    """Use LLM to evaluate answer correctness with concurrent requests to SGLang backend."""
    eval_prompt_template = """Question: {question}

Gold Answer: {gold_answer}

Predicted Answer: {pred_answer}

Are these answers semantically equivalent? Consider:
- Same factual information (even if worded differently)
- Numerical equivalence
- Paraphrasing

Answer with ONLY 'YES' or 'NO'."""

    # Read all records
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Prepare records for evaluation
    eval_records = [rec for rec in records if rec.get("error") is None]
    for rec in records:
        if rec.get("error") is not None:
            rec["correct_llm"] = False

    total_llm = len(eval_records)
    correct_llm = 0
    api_url = f"{config.model_url.rstrip('/')}/chat/completions"

    # Concurrent evaluation (SGLang handles batching internally)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_evaluate_single_answer, rec, api_url, config.model_type, eval_prompt_template): rec
            for rec in eval_records
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                rec, is_correct = future.result()
                correct_llm += int(is_correct)
                if i % 10 == 0 or i == total_llm:
                    print(f"LLM evaluation progress: {i}/{total_llm}")
            except Exception as e:
                print(f"Evaluation error: {e}")

    # Write updated records
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return total_llm, correct_llm


def write_output_files(jsonl_path: Path, output_path: Path, output_format: str) -> None:
    """Write final output files in requested format and CSV."""
    # Convert to JSON array if needed
    if output_format == "json":
        _jsonl_to_json_array(jsonl_path, output_path)

    # Write CSV
    csv_path = output_path.with_suffix(".csv")
    csv_fields = ["idx", "hotpot_id", "question", "gold_answer", "pred_answer", "pred_raw",
                  "correct_em", "correct_llm", "seconds", "error"]

    with jsonl_path.open("r", encoding="utf-8") as fin, csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=csv_fields)
        writer.writeheader()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                row = {k: obj.get(k) for k in csv_fields}
                writer.writerow(row)
            except json.JSONDecodeError:
                continue


def _jsonl_to_json_array(jsonl_path: Path, json_path: Path) -> None:
    """Convert JSONL into a pretty-printed JSON array."""
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
            fout.write("\n".join("  " + s for s in obj_str.splitlines()))
        fout.write("\n]\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--output", default="local/evaluation/direct/hotpotqa.jsonl")
    parser.add_argument("--output-format", default="json", choices=["jsonl", "json"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-type", default="contextual-model")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    parser.add_argument("--mode", default="oneflow", choices=["oneflow", "single", "tree"])
    parser.add_argument("--main_to_search", type=str, default="none", choices=["all", "initial", "none"])
    parser.add_argument("--search", type=str, default="none", choices=["all", "initial", "none"])
    parser.add_argument("--search_to_main", type=str, default="qr", choices=["all", "qr"])
    parser.add_argument("--main", type=str, default="all", choices=["all", "none", "qr"])
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    config = EvalConfig(
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        max_rounds=args.max_rounds,
        output=Path(args.output),
        output_format=args.output_format,
        resume=args.resume,
        model_url=args.model_url,
        model_type=args.model_type,
        tokenizer=args.tokenizer,
        mode=args.mode,
        main_to_search=args.main_to_search,
        search=args.search,
        search_to_main=args.search_to_main,
        main=args.main,
        num_workers=args.num_workers,
    )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    process_dir = config.output.parent / "process"
    process_dir.mkdir(exist_ok=True)

    # Load dataset
    split = config.split
    if config.limit is not None and config.limit > 0:
        split = f"{split}[:{config.limit}]"
    ds = load_dataset("hotpotqa/hotpot_qa", config.subset, split=split)

    # Skip already done (optional)
    done_ids: set[str] = set()
    if config.resume:
        # Check existing worker files
        for worker_file in process_dir.glob("worker_*.jsonl"):
            worker_done = load_done_ids(worker_file)
            done_ids.update(worker_done)

    # Filter dataset
    examples = []
    for idx, ex in enumerate(ds):
        if str(ex["id"]) not in done_ids:
            ex["_idx"] = idx
            examples.append(ex)

    if not examples:
        print("No new examples to evaluate.")
        return

    # Split examples into chunks
    num_workers = min(config.num_workers, len(examples))
    chunk_size = (len(examples) + num_workers - 1) // num_workers
    chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]

    # Start workers
    processes = []
    for worker_id, chunk in enumerate(chunks):
        p = mp.Process(target=worker_process, args=(worker_id, chunk, config, process_dir))
        p.start()
        processes.append(p)

    # Monitor with Rich Progress (file-based)
    worker_files = [process_dir / f"worker_{i}.jsonl" for i in range(len(chunks))]
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        # Create progress bars
        worker_tasks = [progress.add_task(f"Worker {i}", total=len(chunk)) for i, chunk in enumerate(chunks)]
        overall = progress.add_task("[bold]Overall", total=len(examples))

        # Poll worker files
        while any(p.is_alive() for p in processes):
            for i, wfile in enumerate(worker_files):
                if wfile.exists():
                    count = sum(1 for _ in wfile.open("r") if _.strip())
                    progress.update(worker_tasks[i], completed=count)
            total = sum(progress.tasks[t].completed for t in worker_tasks)
            progress.update(overall, completed=total)
            time.sleep(0.5)

        # Final update
        for i, wfile in enumerate(worker_files):
            if wfile.exists():
                count = sum(1 for _ in wfile.open("r") if _.strip())
                progress.update(worker_tasks[i], completed=count)
        progress.update(overall, completed=sum(progress.tasks[t].completed for t in worker_tasks))

    for p in processes:
        p.join()

    # Aggregate results
    total, correct_em = aggregate_results(process_dir, config.output, config.output_format)

    # Determine JSONL path
    if config.output_format == "jsonl":
        final_jsonl = config.output
    else:
        final_jsonl = config.output.with_suffix(".jsonl")

    # LLM-based evaluation
    print("\nRunning LLM-based evaluation...")
    total_llm, correct_llm = llm_evaluate_answers(final_jsonl, config)

    # Write final output files with updated records
    write_output_files(final_jsonl, config.output, config.output_format)

    # Write summary with configuration and prompts
    acc_em = (correct_em / total) if total else 0.0
    acc_llm = (correct_llm / total_llm) if total_llm else 0.0
    csv_path = config.output.with_suffix(".csv")

    summary_lines = [
        "=" * 80,
        "EVALUATION SUMMARY",
        "=" * 80,
        f"Results: evaluated={total}",
        f"  Exact Match: EM={correct_em} acc={acc_em:.3f}",
        f"  LLM Eval: correct={correct_llm} acc={acc_llm:.3f}",
        f"Output: {config.output}",
        f"CSV: {csv_path}",
        "",
        "Arguments:",
        f"  --subset {config.subset}",
        f"  --split {config.split}",
        f"  --limit {config.limit}",
        f"  --max-rounds {config.max_rounds}",
        f"  --output {config.output}",
        f"  --output-format {config.output_format}",
        f"  --resume {config.resume}",
        f"  --model-url {config.model_url}",
        f"  --model-type {config.model_type}",
        f"  --tokenizer {config.tokenizer}",
        f"  --mode {config.mode}",
        f"  --main_to_search {config.main_to_search}",
        f"  --search {config.search}",
        f"  --search_to_main {config.search_to_main}",
        f"  --main {config.main}",
        f"  --num-workers {config.num_workers}",
        "",
    ]

    # Prompt definitions per mode
    mode_prompts = {}
    if config.mode == "tree":
        from rosetta.workflow.tree_prompt import INIT_PROMPT, DECISION_PROMPT, WORKER_PROMPT, REWIND_PROMPT
        mode_prompts["tree"] = [
            ("INIT_PROMPT", INIT_PROMPT),
            ("DECISION_PROMPT", DECISION_PROMPT),
            ("WORKER_PROMPT", WORKER_PROMPT),
            ("REWIND_PROMPT", REWIND_PROMPT),
        ]
    elif config.mode == "oneflow":
        from rosetta.workflow.prompt import (
            SEARCH_TASK_DECOMPOSE_PROMPT,
            TASK_REVISE_PROMPT,
            FORCE_ANSWER_PROMPT,
            SEARCH_AGENT_PROMPT,
        )
        mode_prompts["oneflow"] = [
            ("SEARCH_TASK_DECOMPOSE_PROMPT", SEARCH_TASK_DECOMPOSE_PROMPT),
            ("TASK_REVISE_PROMPT", TASK_REVISE_PROMPT),
            ("FORCE_ANSWER_PROMPT", FORCE_ANSWER_PROMPT),
            ("SEARCH_AGENT_PROMPT", SEARCH_AGENT_PROMPT),
        ]

    # Add prompts to summary
    if config.mode in mode_prompts:
        summary_lines.append(f"Prompts ({config.mode.capitalize()} Mode):")
        summary_lines.append("")
        for prompt_name, prompt_text in mode_prompts[config.mode]:
            summary_lines.extend([
                f"{prompt_name}:",
                "-" * 80,
                prompt_text,
                "",
            ])

    summary_lines.append("=" * 80)

    summary_path = config.output.parent / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nDone. evaluated={total}")
    print(f"  Exact Match: EM={correct_em} acc={acc_em:.3f}")
    print(f"  LLM Eval: correct={correct_llm} acc={acc_llm:.3f}")
    print(f"Output: {config.output}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
