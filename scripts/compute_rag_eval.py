"""Minimal evaluation runner placeholder for Stage 4 Phase A.

This script is intentionally tiny: it loads the sample dataset and prints a brief
summary. The real evaluator will be added in later phases.
"""
import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "evaluation_dataset_sample.json"
    ds = json.loads(data_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(ds)} evaluation examples from {data_path}")


if __name__ == "__main__":
    main()
