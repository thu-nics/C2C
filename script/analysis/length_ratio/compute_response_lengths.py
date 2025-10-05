"""
Compute statistics for model response lengths in a dataset directory.

Supports:
- CSV file at <dataset_dir>/OpenHermes_generated_results.csv
- Hugging Face dataset saved at <dataset_dir>/dataset (via save_to_disk)

Optionally uses a tokenizer to compute token lengths; otherwise uses character lengths.
Saves a JSON summary to <dataset_dir>/response_length_stats.json and prints to stdout.
"""

import os
import json
import argparse
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_csv_if_exists(dataset_dir: str) -> Optional[pd.DataFrame]:
    csv_path = os.path.join(dataset_dir, "OpenHermes_generated_results.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Failed to read CSV at {csv_path}: {e}")
    return None


def read_hf_dataset_if_exists(dataset_dir: str) -> Optional[Any]:
    dataset_path = os.path.join(dataset_dir, "dataset")
    if os.path.exists(dataset_path):
        try:
            from datasets import load_from_disk
            ds = load_from_disk(dataset_path)
            return ds
        except Exception as e:
            print(f"Failed to load HF dataset at {dataset_path}: {e}")
    return None


def compute_char_lengths(texts: List[str]) -> List[int]:
    lengths: List[int] = []
    for t in texts:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            lengths.append(0)
        else:
            lengths.append(len(str(t)))
    return lengths


def compute_token_lengths(texts: List[str], tokenizer_name: str, batch_size: int = 1024) -> List[int]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)

    lengths: List[int] = []
    batch: List[str] = []
    for t in tqdm(texts, desc="Tokenizing", unit="ex"):
        if t is None or (isinstance(t, float) and np.isnan(t)):
            t = ""
        batch.append(str(t))
        if len(batch) >= batch_size:
            enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
            lens = [len(ids) for ids in enc["input_ids"]]
            lengths.extend(lens)
            batch = []
    if batch:
        enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
        lens = [len(ids) for ids in enc["input_ids"]]
        lengths.extend(lens)
    return lengths


def summarize_lengths(lengths: List[int]) -> Dict[str, Any]:
    if len(lengths) == 0:
        return {"count": 0}
    arr = np.array(lengths, dtype=np.int64)
    percentiles = [50, 90, 95, 99]
    perc_vals = np.percentile(arr, percentiles).tolist()
    summary = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": int(arr.min()),
        "p50": float(perc_vals[0]),
        "p90": float(perc_vals[1]),
        "p95": float(perc_vals[2]),
        "p99": float(perc_vals[3]),
        "max": int(arr.max()),
    }
    return summary


def run(dataset_dir: str, column: str, tokenizer_name: Optional[str], batch_size: int, sum_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    # Load data source (CSV preferred for speed; fallback to HF dataset)
    df = read_csv_if_exists(dataset_dir)
    if df is not None:
        if sum_columns:
            # Concatenate multiple columns into one text
            for col in sum_columns:
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not found in CSV columns: {list(df.columns)}")
            texts = [
                "\n".join([
                    "" if (pd.isna(row.get(col)) if isinstance(row, dict) else pd.isna(row[col])) else str(row[col])
                    for col in sum_columns
                ])
                for _, row in df.iterrows()
            ]
        else:
            texts = df.get(column, pd.Series(dtype=object)).tolist()
    else:
        ds = read_hf_dataset_if_exists(dataset_dir)
        if ds is None:
            raise FileNotFoundError(
                f"No CSV or HF dataset found under {dataset_dir}. Expected CSV 'OpenHermes_generated_results.csv' or dataset folder 'dataset'."
            )
        # Avoid loading the full dataset into memory; stream the column
        texts = []
        if sum_columns:
            for col in sum_columns:
                if col not in ds.column_names:
                    raise KeyError(f"Column '{col}' not found in dataset columns: {ds.column_names}")
            for ex in tqdm(ds, desc="Reading dataset", unit="ex"):
                parts = []
                for col in sum_columns:
                    val = ex.get(col, "")
                    if val is None:
                        val = ""
                    parts.append(str(val))
                texts.append("\n".join(parts))
        else:
            if column not in ds.column_names:
                raise KeyError(f"Column '{column}' not found in dataset columns: {ds.column_names}")
            for ex in tqdm(ds, desc="Reading dataset", unit="ex"):
                texts.append(ex.get(column, ""))

    if tokenizer_name:
        lengths = compute_token_lengths(texts, tokenizer_name=tokenizer_name, batch_size=batch_size)
        unit = "tokens"
    else:
        lengths = compute_char_lengths(texts)
        unit = "chars"

    summary = summarize_lengths(lengths)
    summary["unit"] = unit
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute model response length statistics.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="local/teacher_datasets/openhermes_qwen_output",
        help="Directory containing CSV and/or HF dataset subfolder",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="model_response",
        help="Column name to analyze",
    )
    parser.add_argument(
        "--sum_columns",
        nargs="+",
        default=None,
        help="Concatenate these columns (e.g., input_text model_response) and analyze their combined length",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path to compute token lengths. If omitted, character lengths are used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for tokenization",
    )
    args = parser.parse_args()

    summary = run(
        dataset_dir=args.dataset_dir,
        column=args.column,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        sum_columns=args.sum_columns,
    )

    # Print
    print("\n=== Response Length Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save JSON next to dataset
    out_path = os.path.join(args.dataset_dir, "response_length_stats.json")
    try:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {out_path}")
    except Exception as e:
        print(f"Failed to save summary JSON: {e}")


if __name__ == "__main__":
    main()


