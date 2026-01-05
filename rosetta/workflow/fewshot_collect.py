"""Rerun fewshot tasks with a different model and collect best rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, List, Optional, Tuple
from tqdm import tqdm

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from rosetta.workflow.camel_utils import create_model, setup_env
from rosetta.workflow.feedback import FeedbackAgent, DEFAULT_FEEDBACK_PATH
from rosetta.workflow.prompt import SEARCH_AGENT_PROMPT as WORKER_PROMPT
from rosetta.workflow.retriever import search_engine

DEFAULT_CSV_PATH = Path("local/data/fewshot/fewshot.csv")
DEFAULT_OUTPUT_DIR = DEFAULT_FEEDBACK_PATH.parent

# Model config (match script/workflow/subagent_tree.py). Edit as needed.
WORKER_URL = "http://localhost:30000/v1"
WORKER_MODEL_TYPE = "contextual-model"
WORKER_TEMPERATURE = 0.0
WORKER_MAX_TOKENS = 32768
WORKER_ENABLE_THINKING = False


def replay_execute_from_csv(
    csv_path: Path,
    worker_model: BaseModelBackend,
    worker_tools: List[FunctionTool],
    scorer_model: BaseModelBackend,
    model_tag: str,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Replay single-step execution for each CSV row and store scores."""

    def _run_task_once(task: str) -> List[dict]:
        worker_agent = ChatAgent(
            system_message=WORKER_PROMPT,
            model=worker_model,
            tools=worker_tools,
            max_iteration=2
        )
        worker_agent.step(task)
        return worker_agent.chat_history

    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"fewshot_{model_tag}.json"
    output_csv = output_dir / f"fewshot_{model_tag}.csv"

    feedback_agent = FeedbackAgent(model=scorer_model, storage_path=output_json)

    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]

    for offset, row in enumerate(tqdm(rows)):
        task = (row.get("task") or "").strip()
        if not task:
            continue
        fallback_step = offset
        step_idx = _safe_int(row.get("step_idx"), fallback_step)

        chat_history = _run_task_once(task)
        triplets = FeedbackAgent.extract_triplets(chat_history)
        if not triplets:
            continue
        scores = feedback_agent.evaluate([triplets[0]], task, step_idx)
        feedback_agent.store(scores)

    FeedbackAgent.to_csv(json_path=output_json, output_path=output_csv)
    return output_json, output_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun fewshot tasks with a different model and build best.csv."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "rerun", "best"],
        default="all",
        help="Run mode: rerun tasks, build best.csv, or both.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Input fewshot CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max rows to process (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    do_rerun = args.mode in ("all", "rerun")
    do_best = args.mode in ("all", "best")

    # Create worker model
    model_tag = "gemini"
    worker_model = create_model(
        provider="gemini"
    )

    if do_rerun:
        tools = [FunctionTool(search_engine)]

        scorer_model = create_model(
            provider="local",
            model_type="contextual-model",
            model_url="http://localhost:30000/v1",
            temperature=0.0,
            max_tokens=32768,
            chat_template_kwargs={"enable_thinking": False},
        )
        output_json, output_csv = replay_execute_from_csv(
            csv_path=args.csv,
            worker_model=worker_model,
            worker_tools=tools,
            scorer_model=scorer_model,
            model_tag=model_tag,
            output_dir=DEFAULT_OUTPUT_DIR,
            limit=args.limit,
        )
        print(f"feedback json: {output_json}")
        print(f"feedback csv: {output_csv}")

    if do_best:
        best_path = FeedbackAgent.best_csv()
        print(f"best csv: {best_path}")


if __name__ == "__main__":
    setup_env()
    main()
