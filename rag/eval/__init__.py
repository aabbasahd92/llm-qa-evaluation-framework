"""RAG evaluation utilities package.
Expose schema validation and metrics functions for Stage 4 Phase A.
"""
from .schema import validate_example, validate_dataset
from .metrics import exact_match, token_f1

__all__ = ["validate_example", "validate_dataset", "exact_match", "token_f1"]
