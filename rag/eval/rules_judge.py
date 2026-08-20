"""Deterministic, rule-based judge for Phase B.

Provides simple checks:
- abstention detection (phrase-based)
- evidence matching via token overlap with gold evidence
- unsupported-claim heuristic: sentence-level checks with no overlap

This is intentionally conservative and deterministic; it's a scaffold for later
improvements.
"""
from typing import Dict, Any, List
import re

_ABSTAIN_PATTERNS = [
    r"i don'?t know",
    r"cannot answer",
    r"cannot provide",
    r"no information",
    r"unable to",
    r"i am unable",
    r"i can'?t",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def detect_abstain(answer: str) -> bool:
    if not answer:
        return False
    a = answer.strip().lower()
    for p in _ABSTAIN_PATTERNS:
        if re.search(p, a):
            return True
    return False


def match_evidence_by_overlap(answer: str, gold_evidence: List[Dict[str, Any]], min_overlap_tokens: int = 3) -> List[str]:
    """Return list of source_ids that match the answer by token overlap.

    Simple heuristic: count shared tokens between answer and evidence text; if >= min_overlap_tokens,
    consider evidence matched.
    """
    ans_tokens = set(_tokens(answer))
    matched = []
    for ev in gold_evidence:
        text = ev.get("text", "")
        if not text:
            continue
        overlap = len(ans_tokens & set(_tokens(text)))
        if overlap >= min_overlap_tokens:
            matched.append(ev.get("source_id"))
    return matched


def unsupported_claims_ratio(answer: str, gold_evidence: List[Dict[str, Any]]) -> float:
    """Estimate fraction of sentences that are unsupported by gold evidence.

    Sentence is considered supported if it overlaps with any gold evidence by at least
    one token (conservative). Returns 0.0..1.0
    """
    if not answer:
        return 0.0
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    if not sents:
        return 0.0
    supported = 0
    ev_tokens = [set(_tokens(ev.get("text", ""))) for ev in gold_evidence]
    for s in sents:
        stoks = set(_tokens(s))
        if not stoks:
            supported += 1
            continue
        for et in ev_tokens:
            if stoks & et:
                supported += 1
                break
    unsupported = max(0, len(sents) - supported)
    return unsupported / len(sents)
