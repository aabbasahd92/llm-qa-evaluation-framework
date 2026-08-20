"""Simple schema validation helpers for Stage 4 Phase A evaluation dataset.

This is intentionally lightweight and dependency-free (no jsonschema) so tests can run
without adding new dependencies. The validator raises ValueError on invalid input.
"""
from typing import Dict, Any, List

REQUIRED_EXAMPLE_KEYS = {"id", "question", "expected_answers", "gold_evidence", "allow_abstain"}


def _is_string_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def validate_example(example: Dict[str, Any]) -> None:
    """Validate a single evaluation example.

    Raises ValueError with a helpful message on failure.
    """
    if not isinstance(example, dict):
        raise ValueError("example must be a dict")

    missing = REQUIRED_EXAMPLE_KEYS - set(example.keys())
    if missing:
        raise ValueError(f"example missing required keys: {sorted(list(missing))}")

    if not isinstance(example["id"], str) or not example["id"]:
        raise ValueError("example.id must be a non-empty string")

    if not isinstance(example["question"], str) or not example["question"]:
        raise ValueError("example.question must be a non-empty string")

    # expected_answers: list of objects with at least expected_text
    if not isinstance(example["expected_answers"], list) or not example["expected_answers"]:
        raise ValueError("example.expected_answers must be a non-empty list")

    for idx, ans in enumerate(example["expected_answers"]):
        if not isinstance(ans, dict):
            raise ValueError(f"expected_answers[{idx}] must be an object")
        if "expected_text" not in ans or not isinstance(ans["expected_text"], str):
            raise ValueError(f"expected_answers[{idx}].expected_text must be a string")
        # optional lists
        if "required_points" in ans and not _is_string_list(ans["required_points"]):
            raise ValueError(f"expected_answers[{idx}].required_points must be a list of strings")
        if "canonical_answer_variants" in ans and not _is_string_list(ans["canonical_answer_variants"]):
            raise ValueError(f"expected_answers[{idx}].canonical_answer_variants must be a list of strings")

    # gold_evidence: list of objects with source_id and either span or text
    if not isinstance(example["gold_evidence"], list):
        raise ValueError("example.gold_evidence must be a list")
    for idx, ev in enumerate(example["gold_evidence"]):
        if not isinstance(ev, dict):
            raise ValueError(f"gold_evidence[{idx}] must be an object")
        if "source_id" not in ev or not isinstance(ev["source_id"], str):
            raise ValueError(f"gold_evidence[{idx}].source_id must be a string")
        if not ("text" in ev and isinstance(ev["text"], str)) and not ("span" in ev and isinstance(ev["span"], dict)):
            raise ValueError(f"gold_evidence[{idx}] must include either 'text' (string) or 'span' (object with start/end)")

    if not isinstance(example["allow_abstain"], bool):
        raise ValueError("example.allow_abstain must be a boolean")


def validate_dataset(dataset: List[Dict[str, Any]]) -> None:
    """Validate a list of examples."""
    if not isinstance(dataset, list):
        raise ValueError("dataset must be a list of examples")
    ids = set()
    for ex in dataset:
        validate_example(ex)
        if ex["id"] in ids:
            raise ValueError(f"duplicate example id: {ex['id']}")
        ids.add(ex["id"]) 
