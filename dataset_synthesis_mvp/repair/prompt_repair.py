"""Auto-repair prompts based on validation issues."""

from typing import Any


class PromptRepairer:
    """Repair prompts based on validation issues."""

    def repair_prompt(self, original_prompt: str, issues: list[dict[str, Any]], variant_type: str) -> str:
        """Add constraints to prompt based on issues."""
        issue_codes = [i["code"] for i in issues]
        repairs = []

        if "direct_answer_leak" in issue_codes or "indirect_answer_leak" in issue_codes:
            repairs.append(f"- CRITICAL: The {variant_type} question MUST NOT contain the gold answer verbatim or reveal it through reasoning steps.")

        if "type_mismatch" in issue_codes:
            repairs.append(f"- CRITICAL: For {variant_type}, wrong_claim must be the SAME TYPE as gold_answer (both numbers, both entities, both yes/no).")

        if "decorative_removal" in issue_codes:
            repairs.append(f"- CRITICAL: For {variant_type}, you MUST structurally remove the key fact/rule. Do NOT just add 'Answer without...' prefix.")

        if "symbolic_entity_leak" in issue_codes:
            repairs.append(f"- CRITICAL: All real entities must be replaced with symbols. Check that no entity names remain.")

        if not repairs:
            return original_prompt

        return original_prompt + "\n\nIMPORTANT CONSTRAINTS:\n" + "\n".join(repairs)
