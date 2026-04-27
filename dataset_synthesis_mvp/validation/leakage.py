"""5-type leakage detection."""

import re
from typing import Any


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _token_contains(haystack: str, needle: str) -> bool:
    """Token-aware substring match."""
    if not needle:
        return False
    if re.fullmatch(r"[A-Za-z0-9_\-]+", needle):
        pat = rf"\b{re.escape(needle)}\b"
        return re.search(pat, haystack, flags=re.IGNORECASE) is not None
    return _norm(needle) in _norm(haystack)


class LeakageValidator:
    """Detect 5 types of answer leakage."""

    def validate(self, item: dict[str, Any], family: dict[str, Any]) -> list[dict[str, Any]]:
        issues = []

        # 1. Direct leak
        if self._has_direct_leak(item, family):
            issues.append({
                "code": "direct_answer_leak",
                "severity": "BLOCK",
                "message": f"Question contains gold answer: {family['gold_answer']}",
                "action": "regenerate"
            })

        # 2. Indirect leak
        if self._has_indirect_leak(item, family):
            issues.append({
                "code": "indirect_answer_leak",
                "severity": "BLOCK",
                "message": "Question contains reasoning that reveals answer",
                "action": "regenerate"
            })

        # 3. Symbolic leak
        if self._has_symbolic_leak(item, family):
            issues.append({
                "code": "symbolic_entity_leak",
                "severity": "BLOCK",
                "message": "Symbolic question contains unreplaced real entities",
                "action": "fix_symbolic_replacement"
            })

        # 4. Metadata leak
        if self._has_metadata_leak(item, family):
            issues.append({
                "code": "metadata_leak",
                "severity": "WARN",
                "message": "Metadata contains answer",
                "action": "clean_metadata"
            })

        # 5. Paraphrase leak
        if self._has_paraphrase_leak(item, family):
            issues.append({
                "code": "paraphrase_oversimplification",
                "severity": "WARN",
                "message": "Paraphrase may have simplified too much",
                "action": "review_paraphrase"
            })

        return issues

    def _has_direct_leak(self, item: dict[str, Any], family: dict[str, Any]) -> bool:
        """Direct answer in question."""
        gold = str(family["gold_answer"]).lower().strip()
        question = item["question"].lower()
        return _token_contains(question, gold)

    def _has_indirect_leak(self, item: dict[str, Any], family: dict[str, Any]) -> bool:
        """Scaffold/cot reveals answer indirectly."""
        if item["variant"] not in ["scaffold", "wrong_intermediate"]:
            return False

        question = item["question"].lower()
        gold = family["gold_answer"].lower()

        leak_patterns = [
            rf"therefore[,\s]+.*{re.escape(gold)}",
            rf"so the answer is[,\s]+.*{re.escape(gold)}",
            rf"thus[,\s]+.*{re.escape(gold)}",
            rf"final answer[:\s]+.*{re.escape(gold)}",
        ]

        for pattern in leak_patterns:
            if re.search(pattern, question):
                return True
        return False

    def _has_symbolic_leak(self, item: dict[str, Any], family: dict[str, Any]) -> bool:
        """Real entities not replaced in symbolic mode."""
        if item.get("mode") != "symbolic":
            return False

        structure = family.get("underlying_structure", {})
        question = item["question"].lower()

        entities = []
        for node in structure.get("nodes", []):
            if node.get("label"):
                entities.append(node["label"].lower())
        for edge in structure.get("edges", []):
            if edge.get("relation"):
                entities.append(edge["relation"].lower())
        if family.get("gold_answer"):
            entities.append(family["gold_answer"].lower())

        for entity in entities:
            if entity in question:
                return True
        return False

    def _has_metadata_leak(self, item: dict[str, Any], family: dict[str, Any]) -> bool:
        """Metadata contains answer."""
        metadata = item.get("metadata", {})
        gold = family["gold_answer"].lower()

        for value in metadata.values():
            if isinstance(value, str) and gold in value.lower():
                return True
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and gold in v.lower():
                        return True
        return False

    def _has_paraphrase_leak(self, item: dict[str, Any], family: dict[str, Any]) -> bool:
        """Paraphrase oversimplified (Hybrid only)."""
        if item["variant"] != "paraphrase":
            return False
        if family.get("task_family") != "Hybrid":
            return False

        base = family["base_question"]
        para = item["question"]

        base_words = set(re.findall(r"\b[A-Z][a-z]+\b", base))
        para_words = set(re.findall(r"\b[A-Z][a-z]+\b", para))

        if len(base_words & para_words) < len(base_words) * 0.7:
            return True
        return False
