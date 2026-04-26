"""Deterministic repair orchestrator for existing pilot datasets.

This module is intentionally model-free (no LLM calls). It applies targeted,
structure-preserving repairs driven by lint issues.

This file implements a 3-round cleanup loop (post-processing only):

Round 1: anti-leak cleanup
- Remove gold answer leakage from cot/scaffold variants
- Disable high-risk variants (cot_*, scaffold_3, wrongclaim_confident/attributed)

Round 2: structural cleanup
- Force strict, structure-preserving substitution (KB: pseudowords; RB: renaming; Hybrid: pseudowords)
- Detect/disable decorative scaffold in KB
- Auto-review Hybrid families; over-complex ones are excluded from main experiment

Round 3: symbolic + MCQ cleanup
- Generate strict_symbolic variants (role-token based)
- Keep symbolic_overlay vs strict_symbolic distinct
- Repair MCQ options to be type-consistent + role-tagged
"""

from __future__ import annotations

import copy
import re
from typing import Any

from .lint import AuditReport
from .builders.mcq_repair import repair_mcq
from .builders.strict_substitution import build_strict_substitution
from .builders.strict_symbolic import build_strict_symbolic_variants


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _token_contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    if re.fullmatch(r"[A-Za-z0-9_\-]+", needle):
        pat = rf"\b{re.escape(needle)}\b"
        return re.search(pat, haystack, flags=re.IGNORECASE) is not None
    return _norm(needle) in _norm(haystack)


def _remove_answer_leak(text: str, gold: str) -> str:
    """Remove direct mentions of the gold answer while keeping structure.

    Deterministic heuristics:
    - Drop lines that contain the gold answer (token-aware)
    - Also remove common explicit-final-answer phrases on those lines
    - If everything is removed, return the original text (caller can fallback)
    """
    if not isinstance(text, str) or not text.strip() or not gold or len(_norm(gold)) < 2:
        return text

    lines = text.splitlines()
    kept: list[str] = []
    removed_any = False
    for ln in lines:
        if _token_contains(ln, gold):
            removed_any = True
            # drop the whole line to avoid leaving partial leakage.
            continue
        kept.append(ln)

    out = "\n".join(kept).strip()
    if removed_any and out:
        return out
    return text


def _drop_final_answer_lines(text: str) -> str:
    """Remove explicit final-answer disclosure lines in CoT/scaffold.

    This is intentionally heuristic and language-agnostic.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    bad = re.compile(r"(?i)\b(the\s+answer\s+is|therefore\s*,?\s+the\s+answer|final\s+answer)\b|答案是|最终答案")
    kept: list[str] = []
    for ln in text.splitlines():
        if bad.search(ln):
            continue
        kept.append(ln)
    out = "\n".join(kept).strip()
    return out or " ".join(text.split())


def _ensure_variant_dict(family: dict[str, Any], key: str) -> dict[str, Any]:
    nv = family.get(key)
    if not isinstance(nv, dict):
        nv = {}
        family[key] = nv
    return nv


def _record_repair(family: dict[str, Any], code: str, details: dict[str, Any] | None = None) -> None:
    meta = family.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        family["metadata"] = meta
    xs = meta.get("curation_repairs")
    if not isinstance(xs, list):
        xs = []
        meta["curation_repairs"] = xs
    entry: dict[str, Any] = {"code": code}
    if details:
        entry.update(details)
    xs.append(entry)


def _record_disabled(family: dict[str, Any], variant: str, reason: str) -> None:
    meta = family.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        family["metadata"] = meta
    xs = meta.get("disabled_variants")
    if not isinstance(xs, list):
        xs = []
        meta["disabled_variants"] = xs
    xs.append({"variant": variant, "reason": reason})


def _disable_variant(
    family: dict[str, Any],
    variant: str,
    *,
    reason: str,
    blank_question: bool = False,
) -> None:
    nv = _ensure_variant_dict(family, "normal_variants")
    v = nv.get(variant)
    # Do not create new variants during cleanup.
    if v is None:
        _record_disabled(family, variant, reason)
        return
    if not isinstance(v, dict):
        _record_disabled(family, variant, reason)
        return
    m = v.get("metadata")
    if not isinstance(m, dict):
        m = {}
        v["metadata"] = m
    m["disabled"] = True
    m["disabled_reason"] = reason
    if blank_question:
        # Ensure disabled variants cannot carry leaked answers.
        base_q = str(family.get("base_question", "") or "")
        v["question"] = base_q
    _record_disabled(family, variant, reason)


def _mask_gold(text: str, gold: str) -> str:
    """Replace occurrences of gold answer with a stable placeholder (Round1)."""
    if not isinstance(text, str) or not text or not gold:
        return text
    if re.fullmatch(r"[A-Za-z0-9_\-]+", gold):
        pat = rf"\b{re.escape(gold)}\b"
        return re.sub(pat, "[MASK]", text, flags=re.IGNORECASE)
    return text.replace(gold, "[MASK]")


FINAL_VARIANTS_MAIN_EXPERIMENT = [
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


DISALLOW_FOR_MAIN_EXPERIMENT = {
    "cot_full",
    "cot_partial",
    "cot_shuffled",
    "scaffold_3",
    "wrongclaim_confident",
    "wrongclaim_attributed",
}


def repair_families(families: list[dict[str, Any]], audit: AuditReport) -> list[dict[str, Any]]:
    """Backward-compatible repair: applies all rounds (1→2→3).

    Prefer `repair_families_round(..., round_id=1|2|3)` for iterative cleanup.
    """
    out = repair_families_round(families, audit, round_id=1)
    out2 = repair_families_round(out, audit, round_id=2)
    out3 = repair_families_round(out2, audit, round_id=3)
    return out3


def repair_families_round(
    families: list[dict[str, Any]],
    audit: AuditReport,
    *,
    round_id: int,
) -> list[dict[str, Any]]:
    """Apply deterministic repairs for a specific cleanup round."""
    by_id = {r.family_id: r for r in audit.records}
    repaired: list[dict[str, Any]] = []

    for f in families:
        if not isinstance(f, dict):
            continue
        g: dict[str, Any] = copy.deepcopy(f)
        fid = str(g.get("family_id", ""))
        rec = by_id.get(fid)
        issues = rec.issues if rec else []

        gold = str(g.get("gold_answer", "") or "")
        base_q = str(g.get("base_question", "") or "")

        # -----------------------------
        # Round 1: anti-leak cleanup
        # -----------------------------
        if round_id == 1:
            # 1) remove leakage in any flagged CoT/scaffold variant
            for iss in issues:
                if iss.code != "answer_leak" or not iss.variant:
                    continue
                nv = _ensure_variant_dict(g, "normal_variants")
                v = nv.get(iss.variant)
                if isinstance(v, dict) and isinstance(v.get("question"), str):
                    before = v["question"]
                    # Mask, then also drop whole lines that still contain gold.
                    masked = _mask_gold(before, gold)
                    masked = _drop_final_answer_lines(masked)
                    after = _remove_answer_leak(masked, gold)
                    if isinstance(after, str) and after.strip() and not _token_contains(after, gold):
                        v["question"] = after
                    else:
                        # final fallback: safe base question with gold masked out
                        v["question"] = _mask_gold(base_q, gold)
                    m = v.get("metadata")
                    if not isinstance(m, dict):
                        m = {}
                        v["metadata"] = m
                    m["curation_removed_gold"] = True
                    m["curation_removed_gold_variant"] = iss.variant
                    _record_repair(g, "answer_leak", {"variant": iss.variant, "round": 1})

            # 2) disable high-risk CoT variants
            for vt in ("cot_full", "cot_partial", "cot_shuffled"):
                _disable_variant(g, vt, reason="round1_disable_high_risk_cot", blank_question=True)

            # 3) temporarily out of main experiment
            for vt in ("scaffold_3", "wrongclaim_confident", "wrongclaim_attributed"):
                _disable_variant(g, vt, reason="out_of_main_experiment", blank_question=True)

        # -----------------------------
        # Round 2: structural cleanup
        # -----------------------------
        if round_id == 2:
            nv = _ensure_variant_dict(g, "normal_variants")

            # 1) always build strict substitution, store under a dedicated key
            prev = nv.get("substitution") if isinstance(nv.get("substitution"), dict) else None
            rebuilt = build_strict_substitution(g, variant_key="substitution")
            if prev and isinstance(rebuilt, dict):
                meta_prev = prev.get("metadata") if isinstance(prev.get("metadata"), dict) else {}
                meta_new = rebuilt.get("metadata") if isinstance(rebuilt.get("metadata"), dict) else {}
                meta_new = {"original_metadata": meta_prev, **meta_new}
                rebuilt["metadata"] = meta_new
            nv["strict_substitution"] = rebuilt
            _record_repair(g, "strict_substitution", {"variant": "strict_substitution", "round": 2})

            # disable old substitution by default (kept only for debugging)
            if "substitution" in nv:
                _disable_variant(g, "substitution", reason="replaced_by_strict_substitution", blank_question=True)

            # 2) KB decorative scaffold -> disable
            for iss in issues:
                if iss.code != "decorative_scaffold" or not iss.variant:
                    continue
                _disable_variant(g, iss.variant, reason="decorative_scaffold", blank_question=True)
                _record_repair(g, "decorative_scaffold", {"variant": iss.variant, "round": 2})

            # scaffold_3 is always out of main experiment
            _disable_variant(g, "scaffold_3", reason="out_of_main_experiment", blank_question=True)

        # Hybrid overcomplex -> deterministic deprecation marker (Round2)
        if round_id == 2 and any(i.code == "hybrid_overcomplex" for i in issues):
            meta = g.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                g["metadata"] = meta
            meta["curation_flag"] = "hybrid_overcomplex"
            meta["exclude_from_main_experiment"] = True
            _record_repair(g, "hybrid_overcomplex", {"action": "mark_review", "round": 2})

        # -----------------------------
        # Round 3: symbolic + MCQ cleanup
        # -----------------------------
        if round_id == 3:
            # 1) strict symbolic generation (kept separate from symbolic_overlay)
            if not (isinstance(g.get("strict_symbolic_variants"), dict) and isinstance(g.get("strict_symbolic_variants", {}).get("variants"), dict)):
                strict = build_strict_symbolic_variants(g)
                g["strict_symbolic_variants"] = strict
                _record_repair(g, "strict_symbolic", {"attached": "strict_symbolic_variants", "round": 3})

            # 2) MCQ repair (if present)
            mcq_before = g.get("mcq_variants")
            updated = repair_mcq(g)
            if updated is not None:
                g["mcq_variants"] = updated
                _record_repair(g, "mcq_repair", {"variant": "mcq_variants.symbolic_original", "round": 3})
            else:
                if mcq_before:
                    _record_repair(g, "mcq_repair_skipped", {"reason": "unsupported_schema", "round": 3})

        # Always disable variants that are explicitly out of main experiment
        for vt in sorted(DISALLOW_FOR_MAIN_EXPERIMENT):
            _disable_variant(g, vt, reason="out_of_main_experiment", blank_question=True)

        repaired.append(g)

    return repaired
