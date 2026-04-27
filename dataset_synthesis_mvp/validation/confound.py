"""Confound detection (Level 2)."""

from typing import Any
from collections import defaultdict


class ConfoundValidator:
    """Detect probing confounds (all WARN severity)."""

    def validate(self, item: dict[str, Any], family: dict[str, Any], all_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        issues = []

        # 1. Mode confound
        if self._has_mode_confound(item, all_items):
            issues.append({
                "code": "mode_confound",
                "severity": "WARN",
                "message": "This variant's label is 100% predicted by mode",
                "action": "balance_or_drop"
            })

        # 2. Metadata confound
        if self._has_metadata_confound(item, family, all_items):
            issues.append({
                "code": "metadata_confound",
                "severity": "WARN",
                "message": "Label predictable from task_family/variant/mode",
                "action": "balance_or_drop"
            })

        # 3. Length confound
        if self._has_length_confound(item, family):
            issues.append({
                "code": "length_confound",
                "severity": "WARN",
                "message": "Question length differs significantly from baseline",
                "action": "normalize_length"
            })

        return issues

    def _has_mode_confound(self, item: dict[str, Any], all_items: list[dict[str, Any]]) -> bool:
        """Check if mode completely determines label."""
        variant = item["variant"]
        mode = item.get("mode", "natural")

        same_variant = [x for x in all_items if x["variant"] == variant]
        symbolic_items = [x for x in same_variant if x.get("mode") == "symbolic"]

        if len(symbolic_items) >= 5:
            labels = [x.get("expected_label") for x in symbolic_items if "expected_label" in x]
            if labels and (labels.count(True) == len(labels) or labels.count(False) == len(labels)):
                return True
        return False

    def _has_metadata_confound(self, item: dict[str, Any], family: dict[str, Any], all_items: list[dict[str, Any]]) -> bool:
        """Check if metadata predicts label."""
        task = family["task_family"]
        variant = item["variant"]

        same_group = [x for x in all_items
                      if x.get("task_family") == task and x["variant"] == variant]

        if len(same_group) >= 5:
            labels = [x.get("expected_label") for x in same_group if "expected_label" in x]
            if labels:
                pos_ratio = labels.count(True) / len(labels)
                if pos_ratio > 0.9 or pos_ratio < 0.1:
                    return True
        return False

    def _has_length_confound(self, item: dict[str, Any], family: dict[str, Any]) -> bool:
        """Check if length differs significantly."""
        base_len = len(family["base_question"].split())
        item_len = len(item["question"].split())

        if item_len > base_len * 2 or item_len < base_len * 0.5:
            return True
        return False
