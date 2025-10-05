import os
import pandas as pd
from datetime import datetime
from datasets import Dataset

output_dir = "local/teacher_datasets/openhermes_qwen_output"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "OpenHermes_generated_results.csv")

if os.path.exists(csv_path):
    try:
        combined_df = pd.read_csv(csv_path, low_memory=False)
        print(f"Found existing CSV with {len(combined_df)} records. Skipping re-append and building dataset directly.")
    except Exception as e:
        print(f"Error reading existing CSV: {e}")
        combined_df = None

# Filter out rows with empty or None model_response to prevent ArrowTypeError
before_count = len(combined_df)
if "model_response" in combined_df.columns:
    combined_df = combined_df[
        combined_df["model_response"].notna()
        & (combined_df["model_response"].astype(str).str.strip() != "")
    ]
after_count = len(combined_df)
removed = before_count - after_count
if removed > 0:
    print(f"Filtered out {removed} rows with empty model_response")

# Normalize dtypes to avoid pyarrow ArrowTypeError (e.g., floats/None in string fields)
if "id" in combined_df.columns:
    combined_df["id"] = combined_df["id"].astype(str)
if "input_text" in combined_df.columns:
    combined_df["input_text"] = combined_df["input_text"].fillna("").astype(str)
if "model_reasoning" in combined_df.columns:
    combined_df["model_reasoning"] = combined_df["model_reasoning"].fillna("").astype(str)
else:
    combined_df["model_reasoning"] = ""
if "model_response" in combined_df.columns:
    combined_df["model_response"] = combined_df["model_response"].fillna("").astype(str)
if "is_finished" in combined_df.columns:
    combined_df["is_finished"] = combined_df["is_finished"].fillna(True).astype(bool)
else:
    combined_df["is_finished"] = True

# Convert combined data to HuggingFace dataset
dataset_dict = {
    "id": combined_df["id"].tolist(),
    "input_text": combined_df["input_text"].tolist(),
    "model_reasoning": combined_df.get("model_reasoning", pd.Series([None]*len(combined_df))).tolist(),
    "model_response": combined_df["model_response"].tolist(),
    "is_finished": combined_df.get("is_finished", pd.Series([True]*len(combined_df))).tolist(),
}

hf_dataset = Dataset.from_dict(dataset_dict)

dataset_path = os.path.join(output_dir, "dataset")
hf_dataset.save_to_disk(dataset_path)
print(f"Saved HuggingFace dataset with {len(hf_dataset)} problems to {dataset_path}")

hf_dataset_finished = hf_dataset.filter(lambda x: x["is_finished"] == True)
print(f"Filtered dataset with {len(hf_dataset_finished)} finished problems")

dataset_finished_path = os.path.join(output_dir, "dataset_finished")
hf_dataset_finished.save_to_disk(dataset_finished_path)
print(f"Saved finished HuggingFace dataset with {len(hf_dataset_finished)} problems to {dataset_finished_path}")
