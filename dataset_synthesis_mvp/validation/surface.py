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
        """Check if removal is fake (just added prefix)."""
        q = item["question"].lower()
        base = family["base_question"].lower()

        # If removal question is 80%+ similar to base, it's decorative
        ratio = SequenceMatcher(None, q, base).ratio()
        if ratio > 0.8:
            return True

        # If contains "without relying/using", it's decorative
        decorative_phrases = [
            "without relying",
            "without using",
            "without the",
            "ignoring the",
            "answer without"
        ]
        if any(phrase in q for phrase in decorative_phrases):
            return True

        return False
