"""Base item generation (Stage 2) - convert structure to natural language."""

import asyncio
import logging
from typing import Any

from .api_client import APIClient

logger = logging.getLogger(__name__)


BASE_ITEM_SYSTEM = (
    "You are a dataset designer that outputs ONLY a JSON object with no markdown, "
    "no code fences, no commentary, and no explanation. "
    "Your entire response MUST be a single JSON object."
)


async def generate_base_item(client: APIClient, structure: dict[str, Any]) -> dict[str, Any]:
    """Generate base question from structure."""
    import json

    user_prompt = f"""Convert the following structure into a natural-language question.

CRITICAL CONSTRAINTS:
- The question MUST NOT contain the gold_answer literally.
- Output ONLY a JSON object (no markdown, no fences, no commentary).

The JSON object MUST have exactly these keys:
- base_question: string, the clear natural-language question
- gold_answer: string, the correct answer (must match the structure's gold_answer)
- support_facts: array of strings, supporting facts from the structure

Structure:
{json.dumps(structure, indent=2, ensure_ascii=False)}

Output the JSON object now."""

    result = await client.call_api_json_async(
        BASE_ITEM_SYSTEM,
        user_prompt,
        response_format={"type": "json_object"}
    )

    return {
        **structure,
        "base_question": result.get("base_question", ""),
        "gold_answer": result.get("gold_answer", structure.get("gold_answer", "")),
        "support_facts": result.get("support_facts", structure.get("support_facts", [])),
        "underlying_structure": structure
    }


async def generate_all_base_items(
    client: APIClient,
    structures: list[dict[str, Any]],
    concurrency: int = 14
) -> list[dict[str, Any]]:
    """Generate base items for all structures with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def one(s):
        async with sem:
            logger.info(f"Generating base item for {s['family_id']}...")
            return await generate_base_item(client, s)

    families = await asyncio.gather(*[one(s) for s in structures])
    logger.info(f"Generated {len(families)} base items")
    return families
