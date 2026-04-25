"""Post-generation validation for item families.

Five rules:
1. missing_variants       — family must have all normal_variants + symbolic_variants; else discard
2. symbolic_inside_word   — symbol adjacent to [A-Za-z] in symbolic questions → contamination
3. scaffold_answer_leak   — scaffold question explicitly contains gold_answer → leaks answer
4. premise_no_new_info    — premise variant adds no new support fact vs original → not critical
5. removal_not_effective  — premise_removal is still trivially solvable → removal ineffective
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .configs.defaults import SYMBOL_POOL, VARIANT_TYPES

# Symbols that should never be adjacent to letters
_SYMBOL_RE = re.compile(
    r"[A-Za-z][" + re.escape("".join(SYMBOL_POOL)) + r"]"
    r"|[" + re.escape("".join(SYMBOL_POOL)) + r"][A-Za-z]"
)

_SCAFFOLD_VARIANTS = ["scaffold_1", "scaffold_2", "scaffold_3", "scaffold_shuffled"]


@dataclass
class ValidationResult:
    family_id: str
    discard: bool = False
    issues: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)

    def add_issue(self, code: str, detail: str = "") -> None:
        msg = code if not detail else f"{code}: {detail}"
        self.issues.append(msg)
        self.flags[code] = detail or True


# ---------------------------------------------------------------------------
# Rule 1: missing_variants
# ---------------------------------------------------------------------------

def check_missing_variants(family: dict[str, Any], result: ValidationResult) -> None:
    """Discard if any required normal_variant or symbolic_variants is absent."""
    normal = family.get("normal_variants") or {}
    missing_normal = [vt for vt in VARIANT_TYPES if vt not in normal]
    if missing_normal:
        result.discard = True
        result.add_issue("missing_variants", f"missing normal: {missing_normal}")
        return

    sym = family.get("symbolic_variants")
    if not sym or not isinstance(sym, dict) or not sym.get("entity_map"):
        result.discard = True
        result.add_issue("missing_variants", "symbolic_variants absent or has no entity_map")
        return

    missing_sym = [vt for vt in VARIANT_TYPES if vt not in sym]
    if missing_sym:
        result.discard = True
        result.add_issue("missing_variants", f"missing symbolic: {missing_sym}")


# ---------------------------------------------------------------------------
# Rule 2: symbolic_inside_word
# ---------------------------------------------------------------------------

def check_symbolic_inside_word(family: dict[str, Any], result: ValidationResult) -> None:
    """Flag any symbolic variant question where a symbol is adjacent to a letter."""
    sym = family.get("symbolic_variants") or {}
    contaminated: list[str] = []

    for vtype in VARIANT_TYPES:
        variant = sym.get(vtype)
        if not variant:
            continue
        question = variant.get("question", "") if isinstance(variant, dict) else ""
        if _SYMBOL_RE.search(question):
            matches = _SYMBOL_RE.findall(question)
            contaminated.append(f"{vtype}({matches})")

    if contaminated:
        result.discard = True
        result.add_issue("symbolic_inside_word", "; ".join(contaminated))


# ---------------------------------------------------------------------------
# Rule 3: scaffold_answer_leak
# ---------------------------------------------------------------------------

def check_scaffold_answer_leak(family: dict[str, Any], result: ValidationResult) -> None:
    """Flag scaffold variants that explicitly contain the gold answer."""
    gold = (family.get("gold_answer") or "").strip().lower()
    if not gold:
        return

    normal = family.get("normal_variants") or {}
    leaking: list[str] = []

    for vtype in _SCAFFOLD_VARIANTS:
        variant = normal.get(vtype)
        if not variant:
            continue
        question = (variant.get("question", "") if isinstance(variant, dict) else "").lower()
        # Word-boundary match to avoid false positives on substrings
        pattern = r"\b" + re.escape(gold) + r"\b"
        if re.search(pattern, question):
            leaking.append(vtype)

    if leaking:
        if (family.get("task_family") or "").upper() == "RB":
            result.discard = True
        result.add_issue("scaffold_leaks_answer", f"gold='{gold}' found in: {leaking}")


# ---------------------------------------------------------------------------
# Rule 4: premise_no_new_information
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, strip punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def check_premise_no_new_info(family: dict[str, Any], result: ValidationResult) -> None:
    """Flag if premise variant adds no new content tokens vs original."""
    normal = family.get("normal_variants") or {}
    original = normal.get("original")
    premise = normal.get("premise")
    if not original or not premise:
        return

    orig_q = original.get("question", "") if isinstance(original, dict) else ""
    prem_q = premise.get("question", "") if isinstance(premise, dict) else ""

    orig_tokens = _tokenize(orig_q)
    prem_tokens = _tokenize(prem_q)
    new_tokens = prem_tokens - orig_tokens

    # Also check added_premise metadata field
    added_premise = ""
    if isinstance(premise, dict):
        added_premise = (premise.get("metadata") or {}).get("added_premise", "")

    if not new_tokens and not added_premise:
        result.add_issue("premise_not_critical", "premise adds no new tokens vs original")
    elif len(new_tokens) < 3 and not added_premise:
        result.add_issue(
            "premise_not_critical",
            f"premise adds only {len(new_tokens)} new token(s): {new_tokens}",
        )


# ---------------------------------------------------------------------------
# Rule 5: removal_not_effective
# ---------------------------------------------------------------------------

def check_removal_not_effective(family: dict[str, Any], result: ValidationResult) -> None:
    """Flag if premise_removal is too similar to original (removal had no effect)."""
    normal = family.get("normal_variants") or {}
    original = normal.get("original")
    removal = normal.get("premise_removal")
    if not original or not removal:
        return

    orig_q = original.get("question", "") if isinstance(original, dict) else ""
    rem_q = removal.get("question", "") if isinstance(removal, dict) else ""

    orig_tokens = _tokenize(orig_q)
    rem_tokens = _tokenize(rem_q)

    if not orig_tokens:
        return

    # Jaccard similarity
    intersection = orig_tokens & rem_tokens
    union = orig_tokens | rem_tokens
    similarity = len(intersection) / len(union) if union else 1.0

    # Also check: if gold answer still appears verbatim in removal question, it's likely still solvable
    gold = (family.get("gold_answer") or "").strip().lower()
    gold_in_removal = bool(gold and re.search(r"\b" + re.escape(gold) + r"\b", rem_q.lower()))

    if similarity > 0.85:
        result.add_issue(
            "removal_not_effective",
            f"Jaccard similarity={similarity:.2f} (removal too similar to original)",
        )
    elif gold_in_removal:
        result.add_issue(
            "removal_not_effective",
            f"gold answer '{gold}' still present in premise_removal question",
        )


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

_RULES = [
    check_missing_variants,
    check_symbolic_inside_word,
    check_scaffold_answer_leak,
    check_premise_no_new_info,
    check_removal_not_effective,
]


def validate_family(family: dict[str, Any]) -> ValidationResult:
    """Run all validation rules on a single family."""
    result = ValidationResult(family_id=family.get("family_id", "unknown"))
    for rule in _RULES:
        rule(family, result)
        if result.discard:
            # No point running further checks on a discarded family
            break
    return result


def validate_dataset(
    families: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[ValidationResult]]:
    """Validate all families.

    Returns:
        kept:    families that passed (discard=False)
        results: ValidationResult for every family (including kept ones with non-fatal flags)
    """
    kept: list[dict[str, Any]] = []
    results: list[ValidationResult] = []

    for family in families:
        vr = validate_family(family)
        results.append(vr)
        if not vr.discard:
            kept.append(family)

    discarded = sum(1 for r in results if r.discard)
    flagged = sum(1 for r in results if not r.discard and r.issues)
    print(
        f"Validation: {len(families)} total, "
        f"{discarded} discarded, "
        f"{flagged} flagged (kept with warnings), "
        f"{len(kept)} clean."
    )
    return kept, results
