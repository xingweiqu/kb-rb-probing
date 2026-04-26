"""Structure generation — Stage 1 of the pipeline.

Dispatches to block-specific builders to generate underlying structures.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .api_client import APIClient
from .builders.hybrid import generate_hybrid_base_item_async, generate_hybrid_structures_async
from .builders.kb import generate_kb_base_item_async, generate_kb_structures_async
from .builders.rb import generate_rb_base_item_async, generate_rb_structures_async
from .configs.defaults import BATCH_SIZE, CONCURRENCY, FAMILY_COUNTS

logger = logging.getLogger(__name__)

BLOCK_GENERATORS_ASYNC = {
    "KB": generate_kb_structures_async,
    "RB": generate_rb_structures_async,
    "Hybrid": generate_hybrid_structures_async,
}


def generate_all_structures(
    client: APIClient,
    counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Generate underlying structures for all blocks.

    Returns a flat list of structure dicts, each tagged with task_family.
    """
    return asyncio.run(generate_all_structures_async(client, counts))


async def generate_all_structures_async(
    client: APIClient,
    counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Generate underlying structures for all blocks (concurrently by block)."""
    counts = counts or FAMILY_COUNTS

    tasks: list[tuple[str, int, asyncio.Task[list[dict[str, Any]]]]] = []
    for block, count in counts.items():
        generator = BLOCK_GENERATORS_ASYNC.get(block)
        if not generator:
            logger.warning("No generator for block %s, skipping", block)
            continue
        logger.info("Generating %d structures for %s...", count, block)
        tasks.append((block, count, asyncio.create_task(generator(client, count))))

    all_structures: list[dict[str, Any]] = []
    for block, _count, task in tasks:
        structures = await task
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
    return asyncio.run(generate_base_items_async(client, structures))


async def generate_base_items_async(
    client: APIClient,
    structures: list[dict[str, Any]],
    *,
    concurrency: int = CONCURRENCY,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Generate base items for each structure with bounded concurrency."""

    block_item_generators_async = {
        "KB": generate_kb_base_item_async,
        "RB": generate_rb_base_item_async,
        "Hybrid": generate_hybrid_base_item_async,
    }

    sem = asyncio.Semaphore(concurrency)

    async def one(s: dict[str, Any]) -> dict[str, Any] | None:
        block = s.get("task_family", "")
        gen = block_item_generators_async.get(block)
        if not gen:
            logger.warning("No base item generator for block %s, skipping %s", block, s.get("family_id"))
            return None
        async with sem:
            logger.info("Generating base item for %s...", s.get("family_id"))
            base = await gen(client, s)
        merged = {**s, **base}
        merged.setdefault("base_item_id", f"{s.get('family_id', '')}_base")
        return merged

    enriched: list[dict[str, Any]] = []
    for i in range(0, len(structures), batch_size):
        batch = structures[i : i + batch_size]
        out = await asyncio.gather(*(one(s) for s in batch))
        for item in out:
            if item is not None:
                enriched.append(item)

    return enriched
