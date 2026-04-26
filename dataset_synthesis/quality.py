"""Quality scoring for curated families.

This module turns lint results into a scalar score + grade, and produces a
manifest of a recommended usable subset.

It is deterministic and model-free.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .configs.defaults import VARIANT_TYPES
from .lint import AuditReport


# Main experiment: only keep these variants (plus conditional scaffold_2).
MAIN_EXPERIMENT_VARIANTS = [
    "original",
    "hint",
    "premise",
    "premise_removal",
    "highlight",
    "wrongclaim_bare",
    "competing_claims",
    "paraphrase",
    "strict_substitution",
    "scaffold_1",
    "scaffold_2",
]

OUT_OF_MAIN_EXPERIMENT = {
    "cot_full",
    "cot_partial",
    "cot_shuffled",
    "scaffold_3",
    "wrongclaim_confident",
    "wrongclaim_attributed",
}


def _structural_penalties(family: dict[str, Any]) -> list[tuple[str, str]]:
    """Return list of (severity, code) for structural issues not covered by lint."""
    penalties: list[tuple[str, str]] = []
    base_q = family.get("base_question")
    gold = family.get("gold_answer")
    if not isinstance(base_q, str) or not base_q.strip():
        penalties.append(("P0", "missing_base_question"))
    if not isinstance(gold, str) or not gold.strip():
        penalties.append(("P0", "missing_gold_answer"))

    nv = family.get("normal_variants")
    if not isinstance(nv, dict) or not nv:
        penalties.append(("P0", "missing_normal_variants"))
        return penalties

    # Missing keys or empty question bodies are important but not always fatal.
    missing = [k for k in VARIANT_TYPES if k not in nv]
    if missing:
        penalties.append(("P1", "missing_variant_keys"))
    for k in VARIANT_TYPES:
        v = nv.get(k)
        q = v.get("question") if isinstance(v, dict) else None
        if not isinstance(q, str) or not q.strip():
            penalties.append(("P1", "empty_variant_question"))
            break

    # MCQ schema sanity (if present)
    mcq = family.get("mcq_variants")
    if mcq is not None and not isinstance(mcq, dict):
        penalties.append(("P2", "mcq_schema_invalid"))
    return penalties


def _score_from_issues(severities: list[str]) -> float:
    """Map severities to a [0, 1] quality score."""
    score = 1.0
    for sev in severities:
        if sev == "P0":
            score -= 0.40
        elif sev == "P1":
            score -= 0.15
        elif sev == "P2":
            score -= 0.05
        else:
            score -= 0.10
    return max(0.0, min(1.0, score))


def _grade(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.55:
        return "C"
    return "D"


def _recommended_action(has_p0: bool, grade: str, task_family: str) -> str:
    if has_p0:
        return "exclude" if task_family != "Hybrid" else "review_or_exclude"
    if grade in {"A", "B"}:
        return "use"
    if grade == "C":
        return "review"
    return "exclude"


def score_families(families: list[dict[str, Any]], audit: AuditReport) -> list[dict[str, Any]]:
    """Attach `quality_score`/`quality_grade` onto audit records and family metadata."""
    by_id = {r.family_id: r for r in audit.records}
    out: list[dict[str, Any]] = []

    for f in families:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("family_id", ""))
        rec = by_id.get(fid)
        task = str(f.get("task_family", ""))

        severities: list[str] = []
        if rec is not None:
            severities.extend([i.severity for i in rec.issues])
        for sev, _code in _structural_penalties(f):
            severities.append(sev)

        score = _score_from_issues(severities)
        grade = _grade(score)
        has_p0 = "P0" in severities
        action = _recommended_action(has_p0, grade, task)

        # Update audit record
        if rec is not None:
            rec.quality_score = score
            rec.quality_grade = grade
            rec.recommended_action = action

        # Attach to family metadata (do not overwrite unrelated metadata)
        meta = f.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            f["metadata"] = meta
        meta["quality_score"] = score
        meta["quality_grade"] = grade
        meta["recommended_action"] = action

        # Also compute a dedicated main-experiment usability flag.
        usable, reasons = is_usable_for_main_experiment(f, rec)
        meta["usable_main_experiment"] = usable
        meta["usable_main_experiment_reasons"] = reasons
        out.append(f)

    return out


def is_usable_for_main_experiment(
    family: dict[str, Any],
    record: Any | None,
) -> tuple[bool, list[str]]:
    """Decide if a family should enter the main experiment after round2.

    Policy (deterministic):
    - Exclude if `metadata.exclude_from_main_experiment` is set.
    - Exclude Hybrid families flagged overcomplex.
    - Exclude if any P0 issues remain.
    - Exclude if any kept scaffold variant is disabled for leak/decorative.
    """
    reasons: list[str] = []

    meta = family.get("metadata")
    if not isinstance(meta, dict):
        meta = {}

    if meta.get("exclude_from_main_experiment") is True:
        reasons.append("exclude_from_main_experiment")

    if str(family.get("task_family", "")) == "Hybrid" and meta.get("curation_flag") == "hybrid_overcomplex":
        reasons.append("hybrid_overcomplex")

    # Only a subset of P0 issues disqualify main experiment.
    disqualify_p0 = {"answer_leak", "invalid_substitution", "hybrid_overcomplex"}
    if record is not None and getattr(record, "issues", None):
        for i in record.issues:
            if getattr(i, "severity", None) != "P0":
                continue
            code = getattr(i, "code", "unknown")
            if code in disqualify_p0:
                reasons.append(f"P0:{code}")

    usable = len(reasons) == 0
    return usable, reasons


def project_main_experiment_view(family: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of family where `normal_variants` contains only main-experiment variants.

    - Removes variants that are explicitly out-of-main-experiment.
    - Drops `scaffold_2` if it was disabled.
    """
    f = dict(family)
    nv = f.get("normal_variants")
    if not isinstance(nv, dict):
        return f

    meta = f.get("metadata")
    disabled_set: set[str] = set()
    if isinstance(meta, dict) and isinstance(meta.get("disabled_variants"), list):
        for x in meta.get("disabled_variants", []):
            if isinstance(x, dict) and isinstance(x.get("variant"), str):
                disabled_set.add(x["variant"])

    out_nv: dict[str, Any] = {}
    for k in MAIN_EXPERIMENT_VARIANTS:
        if k == "scaffold_2" and k in disabled_set:
            continue
        v = nv.get(k)
        if isinstance(v, dict):
            out_nv[k] = v

    f["normal_variants"] = out_nv
    return f


def build_usable_subset_manifest(families: list[dict[str, Any]], audit: AuditReport) -> dict[str, Any]:
    """Build a manifest describing which families are recommended to use."""
    by_id = {r.family_id: r for r in audit.records}
    by_grade: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}
    usable: list[str] = []
    excluded: list[str] = []
    tf_total: Counter[str] = Counter()
    tf_usable: Counter[str] = Counter()

    for f in families:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("family_id", ""))
        tf = str(f.get("task_family", ""))
        tf_total[tf] += 1

        rec = by_id.get(fid)
        grade = (rec.quality_grade if rec and rec.quality_grade else None) or str(
            (f.get("metadata") or {}).get("quality_grade") or "D"
        )
        grade = grade if grade in by_grade else "D"
        by_grade[grade].append(fid)

        action = (rec.recommended_action if rec and rec.recommended_action else None) or str(
            (f.get("metadata") or {}).get("recommended_action") or "exclude"
        )
        if action == "use":
            usable.append(fid)
            tf_usable[tf] += 1
        else:
            excluded.append(fid)

    return {
        "usable_family_ids": usable,
        "excluded_family_ids": excluded,
        "by_grade": by_grade,
        "by_task_family": {
            tf: {
                "total": int(tf_total[tf]),
                "usable": int(tf_usable[tf]),
            }
            for tf in sorted(tf_total.keys())
        },
        "summary": {
            "total_families": len([f for f in families if isinstance(f, dict)]),
            "usable": len(usable),
            "excluded": len(excluded),
        },
    }
