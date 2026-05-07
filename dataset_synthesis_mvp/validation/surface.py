"""Surface quality validation (Level 1)."""

import re
from typing import Any
from difflib import SequenceMatcher


def _guess_type(x: str) -> str:
    """Guess answer type."""
    x = str(x).strip().lower()
    if x in ["yes", "no", "true", "false"]:
        return "bool"
    if re.match(r"^-?\d+(\.\d+)?$", x):
        return "number"
    return "entity"


class SurfaceValidator:
    """Surface quality checks (all BLOCK severity)."""

    def validate(self, item: dict[str, Any], family: dict[str, Any]) -> list[dict[str, Any]]:
        issues = []

        # 1. Type mismatch (wrongclaim)
        if item["variant"] == "wrongclaim":
            wrong_claim = item.get("metadata", ).get("wrong_claim")
            if wrong_claim and not self._is_same_type(wrong_claim, family["gold_answer"]):
                issues.append({
                    "code": "type_mismatch",
                    "severity": "BLOCK",
                    "message": f"Wrong claim type != gold type",
                    "action": "regenerate"
                })

        # 2. Non-structural removal
        if item["variant"] in ["bridge_removal", "rule_removal", "both_blocked"]:
            if self._is_decorative_removal(item, family):
                issues.append({
                    "code": "decorative_removal",
                    "severity": "BLOCK",
                    "message": "Removal is decorative, not structural",
                    "action": "regenerate"
                })

        return issues

    def _is_same_type(self, wrong_claim: str, gold_answer: str) -> bool:
        """Check if wrong claim and gold are same type."""
        return _guess_type(wrong_claim) == _guess_type(gold_answer)

    def _is_decorative_removal(self, item: dict[str, Any], family: dict[str, Any]) -> bool:
        """Flag a removal as decorative when it only wraps the original question.

        Structural removals delete a clause / fact / entity, so the question
        is typically *shorter* than (or similar in length to) the base. Pure
        decoration prepends or appends a phrase, so the question is longer
        AND uses one of a few telltale wrappers.
        """
        q = item["question"].lower().strip()
        base = family["base_question"].lower().strip()

        decorative_phrases = (
            "without relying",
            "without using",
            "without the",
            "ignoring the",
            "answer without",
            "do not use",
            "do not rely",
            "don't rely",
            "don't use",
        )
        has_phrase = any(p in q for p in decorative_phrases)
        if has_phrase:
            return True

        # Near-identical to base AND no length reduction => no real removal
        ratio = SequenceMatcher(None, q, base).ratio()
        if ratio > 0.95 and len(q) >= len(base):
            return True

        return False
