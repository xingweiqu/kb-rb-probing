"""Variant generation with self-improving loop."""

import asyncio
import logging
from typing import Any

from .api_client import APIClient
from .config import VARIANT_DEFINITIONS, ATOMIC_VARIANTS
from .validation.surface import SurfaceValidator
from .validation.leakage import LeakageValidator
from .validation.confound import ConfoundValidator
from .repair.prompt_repair import PromptRepairer
from .repair.post_process import PostProcessor

logger = logging.getLogger(__name__)


class GenerationLoop:
    """Self-improving variant generation with retry."""

    def __init__(self, client: APIClient):
        self.client = client
        self.surface_validator = SurfaceValidator()
        self.leakage_validator = LeakageValidator()
        self.confound_validator = ConfoundValidator()
        self.repairer = PromptRepairer()
        self.post_processor = PostProcessor()

    async def generate_with_retry(
        self,
        family: dict[str, Any],
        variant_type: str,
        all_items: list[dict[str, Any]],
        max_retries: int = 3
    ) -> dict[str, Any]:
        """Generate variant with auto-retry on failure."""
        prompt = self._build_initial_prompt(family, variant_type)

        for attempt in range(max_retries):
            # 1. Generate
            item = await self._generate_variant(prompt, family, variant_type)

            # 2. Validate
            issues = self._validate_all(item, family, all_items)

            # 3. If pass, return
            if not issues:
                logger.info(f"✓ {variant_type} passed validation (attempt {attempt+1})")
                return item

            # 4. Log issues
            logger.warning(f"✗ {variant_type} failed validation (attempt {attempt+1}): {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue['code']}: {issue['message']}")

            # 5. Try post-process repair
            item = self.post_processor.process_item(item, family, issues)
            issues_after = self._validate_all(item, family, all_items)
            if not issues_after:
                logger.info(f"✓ {variant_type} fixed by post-processor")
                return item

            # 6. Repair prompt and retry
            if attempt < max_retries - 1:
                prompt = self.repairer.repair_prompt(prompt, issues, variant_type)
                logger.info(f"↻ Retrying {variant_type} with repaired prompt...")

        # Max retries reached
        logger.error(f"✗ {variant_type} failed after {max_retries} attempts")
        item["metadata"]["failed"] = True
        return item

    async def _generate_variant(
        self,
        prompt: str,
        family: dict[str, Any],
        variant_type: str
    ) -> dict[str, Any]:
        """Call API to generate variant."""
        system = self._build_system_prompt(variant_type)
        user = prompt

        result = await self.client.call_api_json_async(system, user, response_format={"type": "json_object"})

        return {
            "family_id": family["family_id"],
            "task_family": family["task_family"],
            "variant": variant_type,
            "mode": "natural",
            "question": result.get("question", ""),
            "gold_answer": family["gold_answer"],
            "metadata": result.get("metadata", {})
        }

    def _build_system_prompt(self, variant_type: str) -> str:
        """Build system prompt for variant generation."""
        return (
            "You are a dataset designer that outputs ONLY a JSON object. "
            "No markdown, no code fences, no commentary, no explanation. "
            "Your entire response MUST be a single JSON object."
        )

    def _build_initial_prompt(self, family: dict[str, Any], variant_type: str) -> str:
        """Build initial user prompt."""
        definition = VARIANT_DEFINITIONS.get(variant_type, "")
        return f"""Generate a `{variant_type}` variant for the base item below.

Variant definition: {definition}

The output JSON object MUST have exactly these keys:
- question: string, the variant question text (must NOT literally contain the gold_answer)
- metadata: object, variant-specific metadata (e.g., wrong_claim, injected_premise, hint_text)

Base item:
- family_id: {family['family_id']}
- task_family: {family['task_family']}
- base_question: {family['base_question']}
- gold_answer: {family['gold_answer']}
- support_facts: {family.get('support_facts', [])}

Output the JSON object now."""

    def _validate_all(
        self,
        item: dict[str, Any],
        family: dict[str, Any],
        all_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run all validators."""
        issues = []
        issues.extend(self.surface_validator.validate(item, family))
        issues.extend(self.leakage_validator.validate(item, family))
        issues.extend(self.confound_validator.validate(item, family, all_items))
        return [i for i in issues if i["severity"] == "BLOCK"]  # Only block on BLOCK issues


async def generate_variants_for_family(
    client: APIClient,
    family: dict[str, Any],
    all_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Generate all variants for a family."""
    task = family["task_family"]
    variants_to_gen = ATOMIC_VARIANTS.get(task, [])

    loop = GenerationLoop(client)
    items = []

    # Always generate original first
    original = {
        "family_id": family["family_id"],
        "task_family": task,
        "variant": "original",
        "mode": "natural",
        "question": family["base_question"],
        "gold_answer": family["gold_answer"],
        "metadata": {}
    }
    items.append(original)

    # Generate atomic variants
    for variant_type in variants_to_gen:
        item = await loop.generate_with_retry(family, variant_type, all_items + items)
        items.append(item)

    return items
