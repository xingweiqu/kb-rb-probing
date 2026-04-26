"""Deterministic repair orchestrator for existing pilot datasets.

This module is intentionally model-free (no LLM calls). It applies targeted,
structure-preserving repairs driven by lint issues.

Repairs implemented:
- answer leakage: remove gold answer mentions from CoT/scaffold variants
- invalid substitution: rebuild substitution variant via strict substitution
- symbolic not clean: attach strict symbolic variants (role-token based)
- MCQ type mismatch / weak distractors: repair MCQ options deterministically
- decorative scaffold: disable decorative scaffolds by falling back to base/original
- hybrid over-complexity: mark family for review/deprecation (deterministic)
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


def repair_families(families: list[dict[str, Any]], audit: AuditReport) -> list[dict[str, Any]]:
    """Apply deterministic repairs to families based on lint report."""
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

        # Answer leakage repairs
        for iss in issues:
            if iss.code != "answer_leak" or not iss.variant:
                continue
            nv = _ensure_variant_dict(g, "normal_variants")
            v = nv.get(iss.variant)
            if isinstance(v, dict) and isinstance(v.get("question"), str):
                before = v["question"]
                after = _remove_answer_leak(before, gold)
                # If removal made it empty, fallback to base question.
                if isinstance(after, str) and after.strip():
                    v["question"] = after
                else:
                    v["question"] = base_q
                m = v.get("metadata")
                if not isinstance(m, dict):
                    m = {}
                    v["metadata"] = m
                m["curation_removed_gold"] = True
                m["curation_removed_gold_variant"] = iss.variant
                _record_repair(g, "answer_leak", {"variant": iss.variant})

        # Invalid substitution -> strict substitution rebuild
        if any(i.code == "invalid_substitution" for i in issues):
            nv = _ensure_variant_dict(g, "normal_variants")
            prev = nv.get("substitution") if isinstance(nv.get("substitution"), dict) else None
            rebuilt = build_strict_substitution(g, variant_key="substitution")
            if prev and isinstance(rebuilt, dict):
                # Preserve previous metadata for debugging.
                meta_prev = prev.get("metadata") if isinstance(prev.get("metadata"), dict) else {}
                meta_new = rebuilt.get("metadata") if isinstance(rebuilt.get("metadata"), dict) else {}
                meta_new = {"original_metadata": meta_prev, **meta_new}
                rebuilt["metadata"] = meta_new
            nv["substitution"] = rebuilt
            _record_repair(g, "invalid_substitution", {"variant": "substitution", "strategy": "strict_substitution"})

        # Symbolic overlay not clean -> attach strict symbolic variants
        if any(i.code == "symbolic_not_clean" for i in issues):
            strict = build_strict_symbolic_variants(g)
            g["strict_symbolic_variants"] = strict
            _record_repair(g, "symbolic_not_clean", {"attached": "strict_symbolic_variants"})

        # MCQ repairs
        if any(i.code in {"mcq_type_mismatch", "weak_distractor_set"} for i in issues):
            mcq_before = g.get("mcq_variants")
            updated = repair_mcq(g)
            if updated is not None:
                g["mcq_variants"] = updated
                _record_repair(g, "mcq_repair", {"variant": "mcq_variants.symbolic_original"})
            else:
                # keep a trace if no MCQ exists
                if mcq_before:
                    _record_repair(g, "mcq_repair_skipped", {"reason": "unsupported_schema"})

        # Decorative scaffold -> disable by falling back to base/original
        for iss in issues:
            if iss.code != "decorative_scaffold" or not iss.variant:
                continue
            nv = _ensure_variant_dict(g, "normal_variants")
            v = nv.get(iss.variant)
            if not isinstance(v, dict):
                continue
            orig = nv.get("original") if isinstance(nv.get("original"), dict) else None
            fallback = str((orig or {}).get("question") or base_q)
            v["question"] = fallback
            m = v.get("metadata")
            if not isinstance(m, dict):
                m = {}
                v["metadata"] = m
            m["disabled"] = True
            m["disabled_reason"] = "decorative_scaffold"
            _record_repair(g, "decorative_scaffold", {"variant": iss.variant})

        # Hybrid overcomplex -> deterministic deprecation marker
        if any(i.code == "hybrid_overcomplex" for i in issues):
            meta = g.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                g["metadata"] = meta
            meta["curation_flag"] = "hybrid_overcomplex"
            meta["recommended_action"] = "review_or_exclude"
            # Light-touch shortening to reduce extreme length without semantic rewrite.
            if isinstance(g.get("support_facts"), list):
                g["support_facts"] = [x for x in g["support_facts"] if isinstance(x, str)][:4]
            if isinstance(g.get("gold_reasoning_chain"), list):
                g["gold_reasoning_chain"] = [x for x in g["gold_reasoning_chain"] if isinstance(x, str)][:4]
            if isinstance(g.get("required_steps"), int):
                g["required_steps"] = min(int(g.get("required_steps") or 1), 3)
            _record_repair(g, "hybrid_overcomplex", {"action": "mark_review"})

        repaired.append(g)

    return repaired

