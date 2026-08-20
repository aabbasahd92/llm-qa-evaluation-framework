"""Unit tests for Phase A evaluation scaffold: schema validator and basic metrics."""
import json
from pathlib import Path
import pytest

from rag.eval import schema, metrics


def test_validate_sample_dataset():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "evaluation_dataset_sample.json"
    ds = json.loads(data_path.read_text(encoding="utf-8"))
    # should not raise
    schema.validate_dataset(ds)


def test_schema_missing_key():
    bad = {"id": "x", "question": "q", "expected_answers": [], "gold_evidence": []}  # missing allow_abstain
    with pytest.raises(ValueError):
        schema.validate_example(bad)


def test_exact_match_and_f1():
    pred = "Reset your password from Settings > Security by clicking Reset password and using the emailed link."
    gold_variants = [
        "To reset your password, go to Settings > Security, click 'Reset password', and follow the emailed link.",
    ]
    # exact_match should be False for paraphrase, but token_f1 should be high
    assert metrics.exact_match(pred, gold_variants) is False
    f1 = metrics.token_f1(pred, gold_variants[0])
    assert f1 > 0.6


def test_token_f1_edge_cases():
    assert metrics.token_f1("", "") == 1.0
    assert metrics.token_f1("", "nonempty") == 0.0
    assert metrics.token_f1(None, None) == 1.0
