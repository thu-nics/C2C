#!/usr/bin/env python3
"""
Analyze tool usage statistics from evaluation CSV files.

Usage:
    python script/analysis/tool_usage_stats.py --csv path/to/file.csv
"""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_tools_used(tools_str: str) -> list[list[str]]:
    """Parse the tools_used column which is stored as a string representation of nested lists."""
    if pd.isna(tools_str) or not tools_str:
        return []
    try:
        # Try JSON parsing first
        return json.loads(tools_str)
    except json.JSONDecodeError:
        try:
            # Fall back to ast.literal_eval for Python-style strings
            return ast.literal_eval(tools_str)
        except (ValueError, SyntaxError):
            return []


def analyze_tool_usage(csv_path: str) -> dict:
    """Analyze tool usage from a CSV file."""
    df = pd.read_csv(csv_path)

    if "tools_used" not in df.columns:
        raise ValueError(f"Column 'tools_used' not found in {csv_path}")

    # Parse tools_used column
    df["parsed_tools"] = df["tools_used"].apply(parse_tools_used)

    # Flatten all tool calls and count
    tool_counter = Counter()
    total_calls = 0
    total_steps = 0
    empty_steps = 0

    for tools_list in df["parsed_tools"]:
        for step in tools_list:
            total_steps += 1
            if not step:
                empty_steps += 1
            for tool in step:
                tool_counter[tool] += 1
                total_calls += 1

    # Calculate statistics
    num_questions = len(df)
    avg_calls_per_question = total_calls / num_questions if num_questions > 0 else 0
    avg_steps_per_question = total_steps / num_questions if num_questions > 0 else 0

    return {
        "csv_path": csv_path,
        "num_questions": num_questions,
        "total_tool_calls": total_calls,
        "total_steps": total_steps,
        "empty_steps": empty_steps,
        "avg_calls_per_question": avg_calls_per_question,
        "avg_steps_per_question": avg_steps_per_question,
        "tool_counts": dict(tool_counter.most_common()),
    }


def print_stats(stats: dict) -> None:
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"Tool Usage Statistics: {Path(stats['csv_path']).name}")
    print(f"{'='*60}")

    print(f"\n📊 Overview:")
    print(f"  • Total questions: {stats['num_questions']}")
    print(f"  • Total tool calls: {stats['total_tool_calls']}")
    print(f"  • Total steps: {stats['total_steps']}")
    print(f"  • Empty steps (no tool): {stats['empty_steps']}")
    print(f"  • Avg calls/question: {stats['avg_calls_per_question']:.2f}")
    print(f"  • Avg steps/question: {stats['avg_steps_per_question']:.2f}")

    print(f"\n🔧 Tool Breakdown:")
    print(f"  {'Tool':<25} {'Count':>8} {'Percentage':>12}")
    print(f"  {'-'*45}")

    total = stats["total_tool_calls"]
    for tool, count in stats["tool_counts"].items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {tool:<25} {count:>8} {pct:>11.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze tool usage from evaluation CSV files")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    args = parser.parse_args()

    stats = analyze_tool_usage(args.csv)
    print_stats(stats)


if __name__ == "__main__":
    main()
