"""Family-level lint/audit for existing datasets.

The lint is heuristic and model-free. It focuses on experimental-signal pollution:
- answer leakage
- invalid substitution
- symbolic not clean
- MCQ type mismatch / weak distractors
- decorative scaffolds
- hybrid over-complexity

The output is an AuditReport with per-family issues and summary stats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .configs.defaults import SYMBOL_POOL, VARIANT_TYPES


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str  # "P0" | "P1" | "P2"
    message: str
    variant: str | None = None
    recommendation: str | None = None


@dataclass
class AuditRecord:
    family_id: str
    task_family: str
    sub_family: str
    issues: list[Issue] = field(default_factory=list)

    # Filled later by quality scoring
    quality_score: float | None = None
    quality_grade: str | None = None
    recommended_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "task_family": self.task_family,
            "sub_family": self.sub_family,
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "message": i.message,
                    "variant": i.variant,
                    "recommendation": i.recommendation,
                }
                for i in self.issues
            ],
            "quality_score": self.quality_score,
            "quality_grade": self.quality_grade,
            "recommended_action": self.recommended_action,
        }


@dataclass
class AuditReport:
    records: list[AuditRecord]
    summary: dict[str, Any]


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _token_contains(haystack: str, needle: str) -> bool:
    """Token-aware substring match.

    - For alnum needles (e.g., x, 10, Paris), uses word boundaries.
    - Otherwise uses case-insensitive literal substring.
    """
    if not needle:
        return False
    h = haystack
    n = needle
    if re.fullmatch(r"[A-Za-z0-9_\-]+", n):
        pat = rf"\b{re.escape(n)}\b"
        return re.search(pat, h, flags=re.IGNORECASE) is not None
    return _norm(n) in _norm(h)


def _iter_variants(family: dict[str, Any]) -> Iterable[tuple[str, str]]:
    nv = family.get("normal_variants")
    if isinstance(nv, dict):
        for k, v in nv.items():
            if isinstance(v, dict) and isinstance(v.get("question"), str):
                yield f"normal.{k}", v["question"]

    sym = family.get("symbolic_variants")
    if isinstance(sym, dict):
        for k, v in sym.items():
            if k in ("entity_map", "source_family_id"):
                continue
            if isinstance(v, dict) and isinstance(v.get("question"), str):
                yield f"symbolic_overlay.{k}", v["question"]

    strict = family.get("strict_symbolic_variants")
    if isinstance(strict, dict):
        variants = strict.get("variants")
        if isinstance(variants, dict):
            for k, v in variants.items():
                if isinstance(v, dict) and isinstance(v.get("question"), str):
                    yield f"strict_symbolic.{k}", v["question"]


def _guess_option_type(x: str) -> str:
    s = x.strip()
    if not s:
        return "empty"
    lo = s.lower()
    if lo in {"yes", "no", "true", "false"}:
        return "bool"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return "number"
    if s in SYMBOL_POOL or (len(s) == 1 and ("\u2200" <= s <= "\u22ff" or "\u2190" <= s <= "\u21ff")):
        return "symbol"
    return "entity"


def _mcq_types(family: dict[str, Any]) -> tuple[str | None, list[str]]:
    mcq = family.get("mcq_variants")
    if not isinstance(mcq, dict):
        return None, []
    item = mcq.get("symbolic_original")
    if not isinstance(item, dict):
        return None, []
    opts = item.get("options")
    if not isinstance(opts, list):
        return None, []
    types = [_guess_option_type(str(o)) for o in opts]
    # majority
    maj = None
    if types:
        maj = max(set(types), key=types.count)
    return maj, types


def lint_family(family: dict[str, Any]) -> AuditRecord:
    fid = str(family.get("family_id", ""))
    task = str(family.get("task_family", ""))
    sub = str(family.get("sub_family", ""))
    rec = AuditRecord(family_id=fid, task_family=task, sub_family=sub)

    gold = str(family.get("gold_answer", "") or "")

    # 1) answer leakage (P0)
    if gold and len(_norm(gold)) >= 2:
        nv = family.get("normal_variants")
        if isinstance(nv, dict):
            for vt in ("cot_full", "cot_partial", "cot_shuffled", "scaffold_3", "scaffold_2"):
                v = nv.get(vt)
                q = v.get("question", "") if isinstance(v, dict) else ""
                if isinstance(q, str) and _token_contains(q, gold):
                    rec.issues.append(
                        Issue(
                            code="answer_leak",
                            severity="P0",
                            variant=vt,
                            message=f"{vt} 直接包含 gold_answer: {gold!r}",
                            recommendation="rewrite_cot/scaffold_remove_final_answer",
                        )
                    )

    # 2) invalid substitution (P0)
    nv = family.get("normal_variants")
    if isinstance(nv, dict) and "substitution" in nv:
        sub_v = nv.get("substitution")
        q = sub_v.get("question", "") if isinstance(sub_v, dict) else ""
        meta = sub_v.get("metadata", {}) if isinstance(sub_v, dict) else {}
        subst_map = meta.get("substitution_map") if isinstance(meta, dict) else None
        # Heuristic: if substitution still contains original structure entity labels, it's broken.
        structure = family.get("underlying_structure", {})
        orig_labels: list[str] = []
        if isinstance(structure, dict):
            for n in structure.get("nodes", []) or []:
                if isinstance(n, dict) and isinstance(n.get("label"), str):
                    orig_labels.append(n["label"])
            for e in structure.get("edges", []) or []:
                if isinstance(e, dict) and isinstance(e.get("relation"), str):
                    orig_labels.append(e["relation"])
            for v in structure.get("variables", []) or []:
                if isinstance(v, dict) and isinstance(v.get("name"), str):
                    orig_labels.append(v["name"])
        leaked = [lbl for lbl in orig_labels if lbl and isinstance(q, str) and _token_contains(q, lbl)]

        bad_map = False
        if isinstance(subst_map, dict):
            for _k, v in subst_map.items():
                if not isinstance(v, str) or not v:
                    bad_map = True
                    break
                # strict substitution expects lowercase pseudowords or variable letters
                if re.search(r"[A-Z]", v) and task in {"KB", "Hybrid"}:
                    bad_map = True
                    break
                if " " in v.strip() and task in {"KB", "Hybrid"}:
                    bad_map = True
                    break
        else:
            bad_map = True

        if leaked or bad_map:
            rec.issues.append(
                Issue(
                    code="invalid_substitution",
                    severity="P0",
                    variant="substitution",
                    message="substitution 可能破坏同构/未替换干净",
                    recommendation="rewrite_substitution_as_strict_isomorphic_substitution",
                )
            )

    # 3) symbolic not clean (P0)
    sym = family.get("symbolic_variants")
    strict = family.get("strict_symbolic_variants")
    if isinstance(sym, dict) and sym.get("entity_map"):
        has_strict = (
            isinstance(strict, dict)
            and isinstance(strict.get("variants"), dict)
            and bool(strict.get("variants"))
        )
        # overlay preamble leaks real entities by construction
        if not has_strict:
            rec.issues.append(
                Issue(
                    code="symbolic_not_clean",
                    severity="P0",
                    message="symbolic_variants 属于 symbolic_overlay（preamble 含真实实体），需要生成 strict_symbolic",
                    recommendation="rewrite_symbolic_to_strict_symbolic",
                )
            )

    # 4) MCQ type mismatch / weak distractors (P0)
    maj, types = _mcq_types(family)
    if types:
        if len(set(types)) > 1:
            rec.issues.append(
                Issue(
                    code="mcq_type_mismatch",
                    severity="P0",
                    message=f"MCQ option 类型不一致: {types}",
                    recommendation="rewrite_mcq_options_with_type_consistency",
                )
            )
        # weak distractors: duplicates / empty / too short
        mcq = family.get("mcq_variants", {}).get("symbolic_original", {})
        opts = mcq.get("options", []) if isinstance(mcq, dict) else []
        normed = [_norm(str(x)) for x in opts]
        if len(set(normed)) != len(normed):
            rec.issues.append(
                Issue(
                    code="weak_distractor_set",
                    severity="P0",
                    message="MCQ options 存在重复",
                    recommendation="rewrite_mcq_options_with_type_consistency",
                )
            )
        if any(t == "empty" for t in types):
            rec.issues.append(
                Issue(
                    code="weak_distractor_set",
                    severity="P0",
                    message="MCQ options 存在空选项",
                    recommendation="rewrite_mcq_options_with_type_consistency",
                )
            )

    # 5) decorative scaffold (P1)
    if task == "KB" and isinstance(nv, dict):
        base_q = str(family.get("base_question", "") or "")
        for vt in ("scaffold_1", "scaffold_2", "scaffold_3", "scaffold_shuffled"):
            v = nv.get(vt)
            q = v.get("question", "") if isinstance(v, dict) else ""
            if not isinstance(q, str) or not q.strip():
                continue
            # If scaffold is basically base question with generic tokens only -> decorative.
            added = _norm(q).replace(_norm(base_q), "").strip()
            if added and not re.search(r"\b(france|germany|japan|capital|symbol|author)\b", added, flags=re.I):
                # allow some content words; otherwise treat as decorative
                generic = re.sub(r"\b(step|follow|identify|then|now|first|second|third|answer|question|solve)\b", "", added)
                generic = re.sub(r"[^a-z]+", " ", generic).strip()
                if not generic:
                    rec.issues.append(
                        Issue(
                            code="decorative_scaffold",
                            severity="P1",
                            variant=vt,
                            message="KB scaffold 可能仅形式重述（decorative scaffold）",
                            recommendation="drop_or_disable_decorative_scaffold",
                        )
                    )

    # 6) hybrid overcomplex (P0)
    if task == "Hybrid":
        base_q = str(family.get("base_question", "") or "")
        support_facts = family.get("support_facts", []) or []
        required_steps = int(family.get("required_steps", 0) or 0)
        if required_steps >= 4 or len(support_facts) >= 6 or len(base_q) >= 220 or re.search(r"\b(nobel|prize|born in|according to)\b", base_q, flags=re.I):
            rec.issues.append(
                Issue(
                    code="hybrid_overcomplex",
                    severity="P0",
                    message="Hybrid family 可能过难/不自然/链路过长",
                    recommendation="rewrite_hybrid_to_shorter_support_chain_or_deprecate",
                )
            )

    # Additional checks (P1/P2) placeholders: keep schema-compatible
    # - noncritical_premise
    # - ineffective_removal
    # - inconsistent_variant_type
    # - wrongclaim_not_competitive
    return rec


def lint_families(families: list[dict[str, Any]]) -> AuditReport:
    records = [lint_family(f) for f in families]

    # summary
    by_code: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for r in records:
        for i in r.issues:
            by_code[i.code] = by_code.get(i.code, 0) + 1
            by_sev[i.severity] = by_sev.get(i.severity, 0) + 1
    summary = {
        "total_families": len(records),
        "families_with_issues": sum(1 for r in records if r.issues),
        "issue_counts": dict(sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))),
        "severity_counts": dict(sorted(by_sev.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    return AuditReport(records=records, summary=summary)
