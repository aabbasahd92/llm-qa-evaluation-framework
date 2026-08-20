"""Tests for the Phase B evaluator and rule-based judge."""
from pathlib import Path
import json

from rag.eval import evaluator


def _load_sample():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "evaluation_dataset_sample.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def test_evaluate_correct_answer():
    ds = _load_sample()
    ex = ds[0]
    # an answer that paraphrases the canonical answer
    ans = "Go to Settings → Security, choose Reset password and follow the link sent by email."
    report = evaluator.evaluate_example(ex, ans)
    assert report["em"] is False
    assert report["token_f1"] > 0.6
    assert report["abstained"] is False
    assert len(report["evidence_matches"]) >= 1
    assert report["hallucination_ratio"] < 0.5


def test_evaluate_partial_and_hallucination():
    ds = _load_sample()
    ex = ds[0]
    # partial answer: missing required points and adding an unsupported claim
    ans = "Go to Settings > Security and click Reset password. Note: support staff will call you to confirm identity."
    report = evaluator.evaluate_example(ex, ans)
    assert report["em"] is False
    assert report["token_f1"] < 1.0
    # evidence match should find at least 1
    assert len(report["evidence_matches"]) >= 1
    # hallucination ratio should be > 0 because last sentence is unsupported
    assert report["hallucination_ratio"] > 0.0


def test_evaluate_abstain_behavior():
    ds = _load_sample()
    ex = ds[1]
    ans = "I don't have that information right now."
    report = evaluator.evaluate_example(ex, ans)
    assert report["abstained"] is True
    # when abstaining, other metrics are defined but EM should be False
    assert report["em"] is False


def test_evaluate_dataset_aggregate():
    ds = _load_sample()
    answers = {
        ds[0]["id"]: "Reset your password from Settings > Security by clicking Reset password and using the emailed link.",
        ds[1]["id"]: "I don't know",
    }
    rep = evaluator.evaluate_dataset(ds, answers)
    assert rep["n_examples"] == 2
    assert rep["em_rate"] >= 0.0
    assert "results" in rep
