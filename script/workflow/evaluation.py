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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Callable

from datasets import load_dataset
from transformers import AutoTokenizer
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table

from camel.agents import ChatAgent
from camel.toolkits import FunctionTool, SearchToolkit

from rosetta.workflow.retriever import search_engine
from rosetta.workflow.selector import ContextSelector
from rosetta.workflow.track import InteractionTracker, TreeTracker
from rosetta.workflow.evaluation import (
    extract_answer,
    exact_match,
    load_done_ids,
    run_research,
    LLMJudge,
)
from rosetta.workflow.camel_utils import create_model, setup_env


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
    tools_used: Optional[list[list[str]]] = None
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
    model_type: Optional[str]  # None uses provider defaults
    model_provider: str  # Model provider: local, openai, gemini
    tokenizer: str
    mode: str
    main_to_search: str
    search: str
    search_to_main: str
    main: str
    tools: list[str]
    num_workers: int
    judge_model_provider: str = "local"  # Model provider for LLM judge (default: local)
    judge_api_url: Optional[str] = None  # API URL for LLM judge (defaults to model_url)
    judge_model_type: Optional[str] = None  # Model type for LLM judge
    step_timeout: Optional[float] = None  # Timeout in seconds for agent step calls
    enable_thinking: bool = False  # Enable thinking mode for main agent (local models only)

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
    tools: list[FunctionTool],
) -> EvalRecord:
    """Evaluate a single example."""
    hotpot_id = str(ex["id"])
    question = ex["question"]
    gold = ex["answer"]

    use_single = config.mode == "single"
    use_tree = config.mode == "tree"

    tracker = InteractionTracker(tokenizer=tokenizer)
    agent_tools = tools if use_single else None
    agent_kwargs = {
        "system_message": "You are a helpful assistant.",
        "model": model,
        "tools": agent_tools,
    }
    if config.step_timeout is not None:
        agent_kwargs["step_timeout"] = config.step_timeout
    main_agent = ChatAgent(**agent_kwargs)
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
            search_tools=tools if not use_single else None,
            context_plan=context_plan,
            show_status=False,
            max_rounds=config.max_rounds,
            worker_model=search_model if use_tree else None,
            rewind_model=search_model if use_tree else None,
            exam_model=search_model if use_tree else None,
            worker_tools=tools if use_tree else None,
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

    # Get tools used per round from tree_tracker
    tools_used_per_round: Optional[list[list[str]]] = None
    if tree_tracker is not None:
        try:
            tools_used_per_round = tree_tracker.get_tools_per_round()
        except Exception:
            tools_used_per_round = None

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
        tools_used=tools_used_per_round,
        error=err,
    )


def worker_process(
    worker_id: int,
    examples: list[dict],
    config: EvalConfig,
    process_dir: Path,
    tool_funcs: list[Callable],
) -> None:
    """Worker process that evaluates a chunk of examples."""
    setup_env()

    # Create main model with custom chat_template_kwargs for thinking mode
    main_chat_template_kwargs = {"enable_thinking": config.enable_thinking} if config.model_provider == "local" else None
    main_model = create_model(
        provider=config.model_provider,
        model_type=config.model_type,
        model_url=config.model_url,
        chat_template_kwargs=main_chat_template_kwargs,
    )

    # Create search model (always disable thinking for search)
    search_model = create_model(
        provider=config.model_provider,
        model_type=config.model_type,
        model_url=config.model_url,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    tools = [FunctionTool(func) for func in tool_funcs]
    output_file = process_dir / f"worker_{worker_id}.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for ex in examples:
            rec = evaluate_single(
                idx=ex["_idx"],
                ex=ex,
                config=config,
                model=main_model,
                search_model=search_model,
                tokenizer=tokenizer,
                tools=tools,
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


JUDGE_FIELDS = ["idx", "hotpot_id", "question", "gold_answer", "pred_answer",
                "correct_em", "correct_llm", "judge_confidence", "judge_reason", "error_category"]


def read_jsonl(path: Path) -> list[dict]:
    """Read records from a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_jsonl_path(config: EvalConfig) -> Path:
    """Get JSONL path from config output path."""
    return config.output.with_suffix(".jsonl") if config.output.suffix == ".json" else config.output


def run_llm_judge(jsonl_path: Path, config: EvalConfig, max_workers: int = 32) -> tuple[int, int, dict[str, list]]:
    """Run LLM judge for answer correctness and error categorization."""
    records = read_jsonl(jsonl_path)

    # Create judge model (local uses thinking by default)
    judge_model = create_model(
        provider=config.judge_model_provider,
        model_type=config.judge_model_type or config.model_type,
        model_url=config.judge_api_url or config.model_url,
        chat_template_kwargs={"enable_thinking": True} if config.judge_model_provider == "local" else None,
    )
    judge = LLMJudge(judge_model, max_workers=max_workers)

    # Run judge with progress
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Judging answers", total=1)
        records, total_llm, correct_llm = judge.judge_batch(
            records, progress_callback=lambda c, t: progress.update(task, completed=c, total=t))

        task = progress.add_task("Categorizing errors", total=1)
        records, category_counts = judge.categorize_batch(
            records, progress_callback=lambda c, t: progress.update(task, completed=c, total=t))

    # Write results
    write_jsonl(jsonl_path, records)

    # Write judge-only results
    judge_path = jsonl_path.with_name(jsonl_path.stem + "_judge.jsonl")
    write_jsonl(judge_path, [{k: r.get(k) for k in JUDGE_FIELDS} for r in records])
    print(f"Judge results: {judge_path}")

    return total_llm, correct_llm, category_counts


def print_summary(total: int, correct_em: int, total_llm: int, correct_llm: int, category_counts: dict) -> None:
    """Print evaluation summary."""
    acc_em = (correct_em / total) if total else 0.0
    acc_llm = (correct_llm / total_llm) if total_llm else 0.0
    print(f"\nDone. evaluated={total}")
    print(f"  Exact Match: EM={correct_em} acc={acc_em:.3f}")
    print(f"  LLM Judge: correct={correct_llm} acc={acc_llm:.3f}")
    _print_error_table(category_counts)


def write_output_files(jsonl_path: Path, output_path: Path, output_format: str) -> None:
    """Write final output files in requested format and CSV."""
    # Convert to JSON array if needed
    if output_format == "json":
        _jsonl_to_json_array(jsonl_path, output_path)

    # Write CSV
    csv_path = output_path.with_suffix(".csv")
    csv_fields = ["idx", "hotpot_id", "question", "gold_answer", "pred_answer", "pred_raw",
                  "correct_em", "correct_llm", "judge_confidence", "judge_reason",
                  "error_category", "tools_used", "seconds", "error"]

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
                # Convert tools_used list to JSON string for CSV
                if row.get("tools_used") is not None:
                    row["tools_used"] = json.dumps(row["tools_used"])
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


def _print_error_table(category_counts: dict[str, list]) -> None:
    """Print error category statistics with rich table."""
    console = Console()
    console.print("\n[bold]ERROR CATEGORY ANALYSIS[/bold]\n")

    total_incorrect = sum(len(examples) for examples in category_counts.values())

    table = Table(show_header=True, header_style="bold cyan", title="Error Distribution")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Category", style="magenta", min_width=30)
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    sorted_categories = sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True)

    for rank, (category, examples) in enumerate(sorted_categories, 1):
        count = len(examples)
        percentage = (count / total_incorrect * 100) if total_incorrect > 0 else 0
        table.add_row(f"{rank}", category, f"{count}/{total_incorrect}", f"{percentage:.1f}%")

    console.print(table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--output", default="local/evaluation/direct/hotpotqa.jsonl")
    parser.add_argument("--output-format", default="json", choices=["jsonl", "json"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-type", default=None,
                        help="Model type (default: local='local', openai='gpt-4o-mini', gemini='gemini-3-flash-preview')")
    parser.add_argument("--model-provider", default="local", choices=["local", "openai", "gemini"],
                        help="Model provider: local (OpenAI-compatible), openai, or gemini")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    parser.add_argument("--mode", default="oneflow", choices=["oneflow", "single", "tree"])
    parser.add_argument("--main_to_search", type=str, default="none", choices=["all", "initial", "none"])
    parser.add_argument("--search", type=str, default="none", choices=["all", "initial", "none"])
    parser.add_argument("--search_to_main", type=str, default="qr", choices=["all", "qr"])
    parser.add_argument("--main", type=str, default="all", choices=["all", "none", "qr"])
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["search_engine"],
        choices=["search_engine", "search_wiki", "search_google", "search_tavily"],
        help="Tools to enable (default: search_engine)",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--judge-only", action="store_true",
                        help="Only run LLM judge on existing data (use with --output to specify input file)")
    parser.add_argument("--judge-model-provider", type=str, default="local", choices=["local", "openai", "gemini"],
                        help="Model provider for LLM judge (default: local)")
    parser.add_argument("--judge-api-url", type=str, default=None, help="API URL for LLM judge (defaults to model-url)")
    parser.add_argument("--judge-model-type", type=str, default=None, help="Model type for LLM judge")
    parser.add_argument("--step-timeout", type=float, default=None, help="Timeout in seconds for agent step calls")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for main agent (local models only)")
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
        model_provider=args.model_provider,
        tokenizer=args.tokenizer,
        mode=args.mode,
        main_to_search=args.main_to_search,
        search=args.search,
        search_to_main=args.search_to_main,
        main=args.main,
        tools=args.tools,
        num_workers=args.num_workers,
        judge_model_provider=args.judge_model_provider,
        judge_api_url=args.judge_api_url,
        judge_model_type=args.judge_model_type,
        step_timeout=args.step_timeout,
        enable_thinking=args.enable_thinking,
    )

    config.output.parent.mkdir(parents=True, exist_ok=True)

    # Judge-only mode: skip research, just run LLM judge on existing data
    if args.judge_only:
        jsonl_path = get_jsonl_path(config)
        if not jsonl_path.exists():
            print(f"Error: Input file not found: {jsonl_path}")
            return

        print(f"Running judge-only mode on: {jsonl_path}")
        total_llm, correct_llm, category_counts = run_llm_judge(
            jsonl_path,
            config,
            max_workers=config.num_workers,
        )

        records = read_jsonl(jsonl_path)
        total = len(records)
        correct_em = sum(1 for r in records if r.get("correct_em", False))

        write_output_files(jsonl_path, config.output, config.output_format)
        print_summary(total, correct_em, total_llm, correct_llm, category_counts)
        return

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

    # Define tools as list of callables
    tool_funcs: list[Callable] = []
    search_toolkit: Optional[SearchToolkit] = None
    for tool_name in config.tools:
        if tool_name == "search_engine":
            tool_funcs.append(search_engine)
            continue
        if search_toolkit is None:
            search_toolkit = SearchToolkit()
        if tool_name == "search_wiki":
            tool_funcs.append(search_toolkit.search_wiki)
        elif tool_name == "search_google":
            tool_funcs.append(search_toolkit.search_google)
        elif tool_name == "search_tavily":
            tool_funcs.append(search_toolkit.search_tavily)
        else:
            raise ValueError(f"Unsupported tool: {tool_name}")

    # Split examples into chunks
    num_workers = min(config.num_workers, len(examples))
    chunk_size = (len(examples) + num_workers - 1) // num_workers
    chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]

    # Start workers
    processes = []
    for worker_id, chunk in enumerate(chunks):
        p = mp.Process(target=worker_process, args=(worker_id, chunk, config, process_dir, tool_funcs))
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
    jsonl_path = get_jsonl_path(config)

    # LLM judge for answer correctness and error categorization
    print("\nRunning LLM judge...")
    total_llm, correct_llm, category_counts = run_llm_judge(
        jsonl_path,
        config,
        max_workers=config.num_workers,
    )

    # Write final output files
    write_output_files(jsonl_path, config.output, config.output_format)

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
        f"  --model-provider {config.model_provider}",
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

    # Add error category statistics to summary
    summary_lines.append("Error Category Analysis:")
    summary_lines.append("")
    summary_lines.append(f"{'Rank':<6} {'Category':<35} {'Count':<15} {'Percentage':<10}")
    summary_lines.append("-" * 80)

    total_incorrect = sum(len(examples) for examples in category_counts.values())
    sorted_categories = sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True)

    for rank, (category, examples) in enumerate(sorted_categories, 1):
        count = len(examples)
        percentage = (count / total_incorrect * 100) if total_incorrect > 0 else 0
        summary_lines.append(f"{rank:<6} {category:<35} {count}/{total_incorrect:<13} {percentage:.1f}%")

    summary_lines.append("")
    summary_lines.append("=" * 80)

    summary_path = config.output.parent / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Output: {config.output}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print_summary(total, correct_em, total_llm, correct_llm, category_counts)


if __name__ == "__main__":
    main()
