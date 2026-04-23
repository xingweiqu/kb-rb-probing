"""Structure generation — Stage 1 of the pipeline.

Dispatches to block-specific builders to generate underlying structures.
"""

from __future__ import annotations

import logging
from typing import Any

from .api_client import APIClient
from .builders.hybrid import generate_hybrid_structures
from .builders.kb import generate_kb_structures
from .builders.rb import generate_rb_structures
from .configs.defaults import FAMILY_COUNTS

logger = logging.getLogger(__name__)

BLOCK_GENERATORS = {
    "KB": generate_kb_structures,
    "RB": generate_rb_structures,
    "Hybrid": generate_hybrid_structures,
}


def generate_all_structures(
    client: APIClient,
    counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Generate underlying structures for all blocks.

    Returns a flat list of structure dicts, each tagged with task_family.
    """
    counts = counts or FAMILY_COUNTS
    all_structures: list[dict[str, Any]] = []

    for block, count in counts.items():
        generator = BLOCK_GENERATORS.get(block)
        if not generator:
            logger.warning("No generator for block %s, skipping", block)
            continue

        logger.info("Generating %d structures for %s...", count, block)
        structures = generator(client, count)
        logger.info("Got %d structures for %s", len(structures), block)

        for i, s in enumerate(structures):
            s["task_family"] = block
            if "family_id" not in s:
                s["family_id"] = f"{block.lower()}_{i + 1:03d}"
        all_structures.extend(structures)

    return all_structures


def generate_base_items(
    client: APIClient,
    structures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate base questions and gold answers for each structure.

    Returns the structures enriched with base_question, gold_answer, etc.
    """
    from .builders.kb import generate_kb_base_item
    from .builders.rb import generate_rb_base_item
    from .builders.hybrid import generate_hybrid_base_item

    block_item_generators = {
        "KB": generate_kb_base_item,
        "RB": generate_rb_base_item,
        "Hybrid": generate_hybrid_base_item,
    }

    enriched: list[dict[str, Any]] = []
    for s in structures:
        block = s.get("task_family", "")
        gen = block_item_generators.get(block)
        if not gen:
            logger.warning("No base item generator for block %s, skipping %s", block, s.get("family_id"))
            continue

        logger.info("Generating base item for %s...", s.get("family_id"))
        base = gen(client, s)

        merged = {**s, **base}
        merged.setdefault("base_item_id", f"{s.get('family_id', '')}_base")
        enriched.append(merged)

    return enriched
