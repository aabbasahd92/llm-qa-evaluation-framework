"""Simple report writer for Phase B evaluator."""
import json
from typing import Any, Dict
from pathlib import Path


def write_report(report: Dict[str, Any], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
