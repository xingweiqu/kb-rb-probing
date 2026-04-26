"""Dataset curation tool: lint + repair + quality scoring.

This module is intentionally model-free (no LLM calls). It performs:
- Family-level lint/audit
- Deterministic repairs for common signal-polluting issues
- Strict substitution / strict symbolic generation
- MCQ option repair (type-consistent)
- Quality grading + usable subset manifest

CLI usage (example):
  python -m dataset_synthesis.curate \
    --dataset_jsonl path/to/dataset.jsonl \
    --symbolic_jsonl path/to/symbolic_dataset.jsonl \
    --out_dir ./curated_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .lint import AuditReport, lint_families
from .quality import build_usable_subset_manifest, score_families
from .repair import repair_families
from .utils_io import load_families_any, write_json, write_jsonl, write_report_csv, write_report_jsonl


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def run(
    *,
    dataset_jsonl: Path | None,
    symbolic_jsonl: Path | None,
    out_dir: Path,
) -> None:
    _ensure_out_dir(out_dir)

    base_families: list[dict[str, Any]] = []
    symbolic_families: list[dict[str, Any]] = []

    if dataset_jsonl is not None:
        base_families = load_families_any(dataset_jsonl)
    if symbolic_jsonl is not None:
        symbolic_families = load_families_any(symbolic_jsonl)

    all_families = base_families

    # 1) lint
    audit = lint_families(all_families)

    # 2) repair
    repaired = repair_families(all_families, audit)

    # 3) re-lint after repair for final report
    audit_after = lint_families(repaired)

    # 4) quality scoring + manifest
    scored = score_families(repaired, audit_after)
    manifest = build_usable_subset_manifest(scored, audit_after)

    # 5) export
    write_report_jsonl(audit_after, out_dir / "audit_report.jsonl")
    write_report_csv(audit_after, out_dir / "audit_report.csv")

    write_jsonl(scored, out_dir / "dataset_repaired.jsonl")

    # Symbolic dataset (optional): also repair and attach strict symbolic where possible.
    if symbolic_families:
        sym_audit = lint_families(symbolic_families)
        sym_repaired = repair_families(symbolic_families, sym_audit)
        sym_audit_after = lint_families(sym_repaired)
        sym_scored = score_families(sym_repaired, sym_audit_after)
        write_jsonl(sym_scored, out_dir / "symbolic_dataset_repaired.jsonl")

    write_json(
        {
            "before": audit.summary,
            "after": audit_after.summary,
        },
        out_dir / "repair_summary.json",
    )
    write_json(manifest, out_dir / "usable_subset_manifest.json")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Lint + repair existing family datasets")
    p.add_argument("--dataset_jsonl", type=str, default="", help="Path to dataset.jsonl (or dataset.json)")
    p.add_argument(
        "--symbolic_jsonl",
        type=str,
        default="",
        help="Path to symbolic_dataset.jsonl (optional)",
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = p.parse_args(argv)

    dataset_path = Path(args.dataset_jsonl) if args.dataset_jsonl else None
    symbolic_path = Path(args.symbolic_jsonl) if args.symbolic_jsonl else None
    out_dir = Path(args.out_dir)

    run(dataset_jsonl=dataset_path, symbolic_jsonl=symbolic_path, out_dir=out_dir)


if __name__ == "__main__":
    main()

