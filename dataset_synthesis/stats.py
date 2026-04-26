"""Statistics report generation."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .configs.defaults import ATOMIC_CAPABILITY_MAP, VARIANT_TYPES

logger = logging.getLogger(__name__)


def _norm_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def compute_quality_audit(families: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic quality checks aligned with the manual audit checklist.

    This does NOT run a model; it flags likely structural issues (missing variants,
    answer leakage, MCQ formatting, symbolic preamble, etc.).
    """

    def _add_example(bucket: dict[str, list[str]], key: str, family_id: str, limit: int = 5) -> None:
        xs = bucket.setdefault(key, [])
        if len(xs) < limit:
            xs.append(family_id)

    totals = {
        "families": len(families),
    }
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for f in families:
        fid = str(f.get("family_id", ""))
        base_q = str(f.get("base_question", ""))
        gold = str(f.get("gold_answer", ""))
        support_facts = [x for x in (f.get("support_facts", []) or []) if isinstance(x, str)]

        nv = f.get("normal_variants", {})
        if not isinstance(nv, dict) or not nv:
            counts["missing_normal_variants"] += 1
            _add_example(examples, "missing_normal_variants", fid)
            continue

        # Variant completeness
        missing_v = [vt for vt in VARIANT_TYPES if vt not in nv]
        if missing_v:
            counts["missing_variant_keys"] += 1
            _add_example(examples, "missing_variant_keys", fid)

        empty_q = []
        for vt in VARIANT_TYPES:
            v = nv.get(vt, {})
            q = v.get("question", "") if isinstance(v, dict) else ""
            if not isinstance(q, str) or not q.strip():
                empty_q.append(vt)
        if empty_q:
            counts["empty_variant_questions"] += 1
            _add_example(examples, "empty_variant_questions", fid)

        # Original should match base_question
        orig_q = nv.get("original", {}).get("question", "") if isinstance(nv.get("original"), dict) else ""
        if base_q and orig_q and _norm_text(base_q) != _norm_text(orig_q):
            counts["original_not_equal_base"] += 1
            _add_example(examples, "original_not_equal_base", fid)

        # Hint should not leak the gold answer (heuristic; skip tiny answers)
        hint_q = nv.get("hint", {}).get("question", "") if isinstance(nv.get("hint"), dict) else ""
        if gold and hint_q:
            gnorm = _norm_text(gold)
            if len(gnorm) >= 3 and gnorm in _norm_text(hint_q):
                counts["hint_leaks_gold"] += 1
                _add_example(examples, "hint_leaks_gold", fid)

        # Premise should inject at least one support fact (if available)
        premise_q = nv.get("premise", {}).get("question", "") if isinstance(nv.get("premise"), dict) else ""
        if support_facts and premise_q:
            sf0 = _norm_text(support_facts[0])
            if sf0 and sf0 not in _norm_text(premise_q):
                counts["premise_missing_support_fact"] += 1
                _add_example(examples, "premise_missing_support_fact", fid)

        # Premise-removal should not include the key support fact (heuristic)
        pr_q = nv.get("premise_removal", {}).get("question", "") if isinstance(nv.get("premise_removal"), dict) else ""
        if support_facts and pr_q:
            sf0 = _norm_text(support_facts[0])
            if sf0 and sf0 in _norm_text(pr_q):
                counts["premise_removal_still_has_support_fact"] += 1
                _add_example(examples, "premise_removal_still_has_support_fact", fid)

        # Wrong-claim metadata sanity
        wc = nv.get("wrongclaim_bare") if isinstance(nv.get("wrongclaim_bare"), dict) else None
        if isinstance(wc, dict):
            meta = wc.get("metadata", {}) if isinstance(wc.get("metadata"), dict) else {}
            wrong_claim = meta.get("wrong_claim", "")
            q = wc.get("question", "")
            if not wrong_claim or not isinstance(wrong_claim, str):
                counts["wrongclaim_missing_metadata"] += 1
                _add_example(examples, "wrongclaim_missing_metadata", fid)
            elif isinstance(q, str) and wrong_claim and _norm_text(wrong_claim) not in _norm_text(q):
                counts["wrongclaim_not_in_question"] += 1
                _add_example(examples, "wrongclaim_not_in_question", fid)

        # Symbolic preamble + entity map uniqueness
        sym = f.get("symbolic_variants")
        if isinstance(sym, dict) and sym.get("entity_map"):
            entity_map = sym.get("entity_map", {})
            if isinstance(entity_map, dict):
                symbols = [v for v in entity_map.values() if isinstance(v, str)]
                if len(symbols) != len(set(symbols)):
                    counts["symbolic_duplicate_symbols"] += 1
                    _add_example(examples, "symbolic_duplicate_symbols", fid)

            for vt in VARIANT_TYPES:
                sv = sym.get(vt)
                q = sv.get("question", "") if isinstance(sv, dict) else ""
                if isinstance(q, str) and q.strip() and not q.lstrip().startswith("In this system,"):
                    counts["symbolic_missing_preamble"] += 1
                    _add_example(examples, "symbolic_missing_preamble", fid)
                    break

        # MCQ sanity
        mcq = f.get("mcq_variants")
        if isinstance(mcq, dict) and mcq:
            so = mcq.get("symbolic_original", {})
            opts = so.get("options", []) if isinstance(so, dict) else []
            ci = so.get("correct_index", 0) if isinstance(so, dict) else 0
            if not (isinstance(opts, list) and len(opts) == 4 and all(isinstance(x, str) and x.strip() for x in opts)):
                counts["mcq_bad_options"] += 1
                _add_example(examples, "mcq_bad_options", fid)
            elif len(set(_norm_text(x) for x in opts)) != 4:
                counts["mcq_duplicate_options"] += 1
                _add_example(examples, "mcq_duplicate_options", fid)
            if not (isinstance(ci, int) and 0 <= ci < 4):
                counts["mcq_bad_correct_index"] += 1
                _add_example(examples, "mcq_bad_correct_index", fid)

    # Convert to percents for quick scanning.
    pct = {
        k: round(v / totals["families"] * 100, 1) if totals["families"] else 0
        for k, v in counts.items()
    }
    return {
        "totals": totals,
        "issue_counts": dict(counts),
        "issue_pct": pct,
        "examples": examples,
    }


def compute_stats(families: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset statistics."""
    total = len(families)
    block_counts = Counter(f.get("task_family", "unknown") for f in families)
    sub_family_counts = Counter(f.get("sub_family", "unknown") for f in families)

    # Variant coverage
    variant_coverage: dict[str, int] = {vt: 0 for vt in VARIANT_TYPES}
    for f in families:
        nv = f.get("normal_variants", {})
        for vt in VARIANT_TYPES:
            if vt in nv:
                variant_coverage[vt] += 1

    variant_coverage_pct = {
        vt: round(count / total * 100, 1) if total > 0 else 0
        for vt, count in variant_coverage.items()
    }

    # Symbolic coverage
    symbolic_count = sum(
        1 for f in families
        if f.get("symbolic_variants") and isinstance(f["symbolic_variants"], dict)
        and f["symbolic_variants"].get("entity_map")
    )

    # MCQ coverage
    mcq_count = sum(
        1 for f in families
        if f.get("mcq_variants") and isinstance(f["mcq_variants"], dict)
        and len(f["mcq_variants"]) > 0
    )

    # Atomic capability coverage
    capability_coverage: dict[str, dict[str, Any]] = {}
    for cap, vtypes in ATOMIC_CAPABILITY_MAP.items():
        covered = sum(1 for f in families if all(
            vt in f.get("normal_variants", {}) for vt in vtypes
        ))
        capability_coverage[cap] = {
            "variants": vtypes,
            "families_with_full_coverage": covered,
            "coverage_pct": round(covered / total * 100, 1) if total > 0 else 0,
        }

    return {
        "total_families": total,
        "block_counts": dict(block_counts),
        "sub_family_counts": dict(sub_family_counts),
        "variant_coverage_counts": variant_coverage,
        "variant_coverage_pct": variant_coverage_pct,
        "symbolic_families": symbolic_count,
        "symbolic_coverage_pct": round(symbolic_count / total * 100, 1) if total > 0 else 0,
        "mcq_families": mcq_count,
        "mcq_coverage_pct": round(mcq_count / total * 100, 1) if total > 0 else 0,
        "atomic_capability_coverage": capability_coverage,
        "quality_audit": compute_quality_audit(families),
    }


def export_stats(families: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    """Compute and export statistics to a JSON file."""
    stats = compute_stats(families)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Stats exported to %s", output_path)
    return stats
