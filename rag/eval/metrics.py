"""Basic evaluation metrics for Stage 4 Phase A.

Provides exact match and token-level F1. These are intentionally minimal and
suitable for unit testing and early CI integration.
"""
import re
from typing import List

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    if text is None:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def exact_match(pred: str, gold_variants: List[str]) -> bool:
    """Return True if pred matches any gold variant exactly after normalization.

    Normalization: collapse whitespace and lowercase.
    """
    if pred is None:
        return False
    norm = " ".join(pred.strip().lower().split())
    for g in gold_variants:
        if not isinstance(g, str):
            continue
        if norm == " ".join(g.strip().lower().split()):
            return True
    return False


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between pred and gold (both strings).

    Returns 0.0..1.0
    """
    p_tokens = _tokens(pred)
    g_tokens = _tokens(gold)
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    # Count overlap
    from collections import Counter

    p_counts = Counter(p_tokens)
    g_counts = Counter(g_tokens)
    common = sum(min(p_counts[t], g_counts[t]) for t in p_counts.keys() & g_counts.keys())
    if common == 0:
        return 0.0
    precision = common / sum(p_counts.values())
    recall = common / sum(g_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
