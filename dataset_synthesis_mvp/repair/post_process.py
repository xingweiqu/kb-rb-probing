"""Post-process generated items to fix issues."""

import re
from typing import Any


class PostProcessor:
    """Auto-fix generated items."""

    def process_item(self, item: dict[str, Any], family: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply fixes based on issues."""
        for issue in issues:
            if issue["code"] == "direct_answer_leak":
                item = self._remove_answer_leak(item, family)
            elif issue["code"] == "type_mismatch":
                item = self._fix_type_mismatch(item, family)
            elif issue["code"] == "metadata_leak":
                item = self._clean_metadata(item, family)

        return item

    def _remove_answer_leak(self, item: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
        """Mask answer in question."""
        gold = family["gold_answer"]
        question = item["question"]

        pattern = rf"\b{re.escape(gold)}\b"
        question = re.sub(pattern, "[MASK]", question, flags=re.IGNORECASE)

        item["question"] = question
        item.setdefault("metadata", {})["auto_repaired"] = True
        item["metadata"]["repair_type"] = "answer_leak_removed"
        return item

    def _fix_type_mismatch(self, item: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
        """Generate same-type wrong claim."""
        gold = family["gold_answer"]

        def guess_type(x):
            x = str(x).strip().lower()
            if x in ["yes", "no", "true", "false"]:
                return "bool"
            if re.match(r"^-?\d+(\.\d+)?$", x):
                return "number"
            return "entity"

        gold_type = guess_type(gold)

        if gold_type == "bool":
            wrong = "No" if gold.lower() in ["yes", "true"] else "Yes"
        elif gold_type == "number":
            try:
                wrong = str(float(gold) + 1)
            except:
                wrong = "0"
        else:
            wrong = gold + "_alt"

        item.setdefault("metadata", {})["wrong_claim"] = wrong
        item["metadata"]["auto_repaired"] = True
        return item

    def _clean_metadata(self, item: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
        """Remove answer from metadata."""
        gold = family["gold_answer"].lower()
        metadata = item.get("metadata", {})

        for key, value in list(metadata.items()):
            if isinstance(value, str) and gold in value.lower():
                metadata[key] = "[REDACTED]"

        item["metadata"]["auto_repaired"] = True
        return item
