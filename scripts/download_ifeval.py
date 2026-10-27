"""Download the IFEval dataset (google/IFEval on Hugging Face) and save it as a CSV.

Output: data_raw/ifeval.csv

Columns:
  key                  - prompt id
  prompt               - the instruction-following prompt
  instruction_id_list  - JSON list of instruction ids, e.g. ["punctuation:no_comma"]
  kwargs               - JSON list of argument dicts, one per instruction

The two list columns are stored as JSON strings; read them back with json.loads.

Usage (from the project root):
    python scripts/download_ifeval.py
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "google/IFEval"
FILENAME = "ifeval_input_data.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "data_raw"
OUT_FILE = OUT_DIR / "ifeval.csv"
LIST_COLUMNS = ["instruction_id_list", "kwargs"]


def main():
    jsonl_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    df = pd.read_json(jsonl_path, lines=True)

    # CSV can't hold lists, so store them as JSON text.
    for col in LIST_COLUMNS:
        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {OUT_FILE}")


if __name__ == "__main__":
    main()
