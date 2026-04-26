"""Dataset curation tool: 3-round cleanup (post-processing).

This module is intentionally model-free (no LLM calls). It implements an
iterative cleanup loop to push the pilot dataset towards main-experiment quality.

CLI usage:
  # Round 1
  python -m dataset_synthesis.curate --round 1 --dataset_jsonl dataset.jsonl --out_dir ./out

  # Round 2
  python -m dataset_synthesis.curate --round 2 --dataset_jsonl dataset.jsonl --out_dir ./out

  # Round 3
  python -m dataset_synthesis.curate --round 3 --symbolic_jsonl symbolic_dataset.jsonl --out_dir ./out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .lint import lint_families
from .quality import project_main_experiment_view, score_families
from .repair import repair_families_round
from .utils_io import load_families_any, write_json, write_jsonl, write_report_csv, write_report_jsonl


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _build_audit_json(families: list[dict[str, Any]], audit) -> dict[str, Any]:
    by_id = {r.family_id: r for r in audit.records}
    per_family: list[dict[str, Any]] = []
    for f in families:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("family_id", ""))
        meta = f.get("metadata") if isinstance(f.get("metadata"), dict) else {}
        rec = by_id.get(fid)
        per_family.append(
            {
                "family_id": fid,
                "task_family": str(f.get("task_family", "")),
                "sub_family": str(f.get("sub_family", "")),
                "issues": [
                    {
                        "code": i.code,
                        "severity": i.severity,
                        "variant": i.variant,
                        "message": i.message,
                        "recommendation": i.recommendation,
                    }
                    for i in (rec.issues if rec else [])
                ],
                "repaired": meta.get("curation_repairs", []),
                "disabled_variants": meta.get("disabled_variants", []),
                "quality_grade": meta.get("quality_grade"),
                "quality_score": meta.get("quality_score"),
                "usable_main_experiment": meta.get("usable_main_experiment"),
                "usable_main_experiment_reasons": meta.get("usable_main_experiment_reasons", []),
            }
        )
    return {
        "summary": audit.summary,
        "families": per_family,
    }


def run(*, round_id: int, dataset_jsonl: Path | None, symbolic_jsonl: Path | None, out_dir: Path) -> None:
    _ensure_out_dir(out_dir)

    if round_id in (1, 2):
        if dataset_jsonl is None:
            raise ValueError("Round 1/2 require --dataset_jsonl")
        families = load_families_any(dataset_jsonl)

        # Round chaining: round2 assumes round1 has already been applied.
        audit0 = lint_families(families)
        r1 = repair_families_round(families, audit0, round_id=1)
        audit1 = lint_families(r1)

        repaired = r1
        audit_after = audit1
        if round_id == 2:
            r2 = repair_families_round(r1, audit1, round_id=2)
            audit2 = lint_families(r2)
            repaired = r2
            audit_after = audit2

        scored = score_families(repaired, audit_after)

        if round_id == 1:
            out_path = out_dir / "dataset_round1_clean.jsonl"
            audit_path = out_dir / "audit_round1.json"
            write_jsonl(scored, out_path)
            write_json(_build_audit_json(scored, audit_after), audit_path)
            return

        # Round 2
        out_path = out_dir / "dataset_round2_structural_clean.jsonl"
        usable_path = out_dir / "usable_subset_round2.json"

        projected = [project_main_experiment_view(f) for f in scored]
        write_jsonl(projected, out_path)

        usable_ids = [
            str(f.get("family_id", ""))
            for f in scored
            if isinstance(f, dict)
            and isinstance(f.get("metadata"), dict)
            and f["metadata"].get("usable_main_experiment") is True
        ]
        write_json({"usable_family_ids": usable_ids, "total": len(projected)}, usable_path)
        return

    if round_id == 3:
        if symbolic_jsonl is None:
            raise ValueError("Round 3 require --symbolic_jsonl")
        sym_families = load_families_any(symbolic_jsonl)

        # Round chaining: round3 assumes round1+round2 already applied.
        audit0 = lint_families(sym_families)
        r1 = repair_families_round(sym_families, audit0, round_id=1)
        audit1 = lint_families(r1)
        r2 = repair_families_round(r1, audit1, round_id=2)
        audit2 = lint_families(r2)
        r3 = repair_families_round(r2, audit2, round_id=3)
        audit_after = lint_families(r3)
        repaired = r3
        scored = score_families(repaired, audit_after)

        out_path = out_dir / "symbolic_dataset_round3_clean.jsonl"
        mcq_audit_path = out_dir / "mcq_audit_round3.json"
        write_jsonl(scored, out_path)

        # MCQ audit summary (deterministic): count remaining mcq issues + repairs
        by_id = {r.family_id: r for r in audit_after.records}
        total = 0
        with_mcq = 0
        remaining_mismatch = 0
        remaining_weak = 0
        repaired_count = 0
        for f in scored:
            if not isinstance(f, dict):
                continue
            total += 1
            if isinstance(f.get("mcq_variants"), dict) and f["mcq_variants"]:
                with_mcq += 1
            meta = f.get("metadata") if isinstance(f.get("metadata"), dict) else {}
            repairs = meta.get("curation_repairs", [])
            if isinstance(repairs, list) and any(isinstance(x, dict) and x.get("code") == "mcq_repair" for x in repairs):
                repaired_count += 1
            rec = by_id.get(str(f.get("family_id", "")))
            if rec:
                for i in rec.issues:
                    if i.code == "mcq_type_mismatch":
                        remaining_mismatch += 1
                    if i.code == "weak_distractor_set":
                        remaining_weak += 1

        write_json(
            {
                "total_families": total,
                "families_with_mcq": with_mcq,
                "mcq_repaired": repaired_count,
                "remaining_mcq_type_mismatch": remaining_mismatch,
                "remaining_weak_distractor_set": remaining_weak,
            },
            mcq_audit_path,
        )
        return

    raise ValueError(f"Unknown round_id: {round_id}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Lint + repair existing family datasets")
    p.add_argument("--round", type=int, required=True, choices=[1, 2, 3], help="Cleanup round: 1/2/3")
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

    run(round_id=int(args.round), dataset_jsonl=dataset_path, symbolic_jsonl=symbolic_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
