"""Robust IO helpers for JSON/JSONL family datasets.

No hard-coded paths; callers provide input/output paths.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


def load_families_any(path: Path) -> list[dict[str, Any]]:
    """Load either JSONL (one family per line) or JSON (array/object) into list[dict]."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Heuristic: JSONL if multiple lines and each line looks like JSON object.
    if suffix == ".jsonl" or ("\n" in text and text.lstrip().startswith("{") and not text.lstrip().startswith("[")):
        families: list[dict[str, Any]] = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e
            if isinstance(obj, list):
                # tolerate a list per line, flatten
                for x in obj:
                    if isinstance(x, dict):
                        families.append(x)
            elif isinstance(obj, dict):
                families.append(obj)
        return families

    # JSON
    obj = json.loads(text)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def write_jsonl(items: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_report_jsonl(report: Any, path: Path) -> None:
    """Write AuditReport to JSONL (one family record per line)."""
    from .lint import AuditReport

    if not isinstance(report, AuditReport):
        raise TypeError("write_report_jsonl expects AuditReport")
    write_jsonl([r.to_dict() for r in report.records], path)


def write_report_csv(report: Any, path: Path) -> None:
    """Write AuditReport to CSV."""
    from .lint import AuditReport

    if not isinstance(report, AuditReport):
        raise TypeError("write_report_csv expects AuditReport")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family_id",
        "task_family",
        "sub_family",
        "quality_grade",
        "quality_score",
        "recommended_action",
        "issues",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report.records:
            w.writerow(
                {
                    "family_id": r.family_id,
                    "task_family": r.task_family,
                    "sub_family": r.sub_family,
                    "quality_grade": r.quality_grade or "",
                    "quality_score": r.quality_score if r.quality_score is not None else "",
                    "recommended_action": r.recommended_action or "",
                    "issues": ";".join(sorted({i.code for i in r.issues})),
                }
            )

