"""5-stage pipeline orchestrator with checkpoint management.

Usage:
    from dataset_synthesis.pipeline import run_pipeline
    run_pipeline(output_dir="./synthesis_output")

Or from CLI:
    python -m dataset_synthesis.pipeline --output_dir ./synthesis_output
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .api_client import APIClient
from .configs.defaults import (
    CHECKPOINT_DIR,
    FAMILY_COUNTS,
    MODEL,
    OUTPUT_DIR,
    SYMBOLIC_CONTROL_COUNT,
)
from .export import export_all, export_symbolic_dataset
from .configs.defaults import BATCH_SIZE, CONCURRENCY
from .mcq import generate_mcq_async
from .stats import export_stats
from .structures import generate_all_structures_async
from .symbolic import generate_all_symbolic
from .variants import generate_variants_async

from .builders.hybrid import generate_hybrid_base_item_async
from .builders.kb import generate_kb_base_item_async
from .builders.rb import generate_rb_base_item_async

logger = logging.getLogger(__name__)


def _load_checkpoint(path: Path) -> list[dict[str, Any]] | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded checkpoint %s (%d items)", path, len(data))
        return data
    return None


def _save_checkpoint(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved checkpoint %s (%d items)", path, len(data))


def run_pipeline(
    output_dir: str = OUTPUT_DIR,
    model: str = MODEL,
    family_counts: dict[str, int] | None = None,
    symbolic_count: int = SYMBOLIC_CONTROL_COUNT,
    mock: bool = False,
    export_symbolic: bool = False,
) -> list[dict[str, Any]]:
    """Run the full 5-stage synthesis pipeline.

    Args:
        output_dir: Root directory for checkpoints and output.
        model: Model name for API calls.
        family_counts: Override family counts per block.
        symbolic_count: Number of SymbolicControl families to derive.

    Returns:
        The final list of family dicts.
    """
    return asyncio.run(
        run_pipeline_async(
            output_dir=output_dir,
            model=model,
            family_counts=family_counts,
            symbolic_count=symbolic_count,
            mock=mock,
            export_symbolic=export_symbolic,
        )
    )


async def run_pipeline_async(
    *,
    output_dir: str = OUTPUT_DIR,
    model: str = MODEL,
    family_counts: dict[str, int] | None = None,
    symbolic_count: int = SYMBOLIC_CONTROL_COUNT,
    mock: bool = False,
    export_symbolic: bool = False,
    concurrency: int = CONCURRENCY,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    base = Path(output_dir)
    ckpt = base / CHECKPOINT_DIR
    out = base / "output"

    client = APIClient(model=model, mock=mock)
    counts = family_counts or FAMILY_COUNTS

    block_item_generators_async = {
        "KB": generate_kb_base_item_async,
        "RB": generate_rb_base_item_async,
        "Hybrid": generate_hybrid_base_item_async,
    }

    async def _bounded_map(items: list[Any], worker):
        sem = asyncio.Semaphore(concurrency)

        async def one(x):
            async with sem:
                return await worker(x)

        results: list[Any] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            out_batch = await asyncio.gather(*(one(x) for x in batch))
            results.extend(out_batch)
        return results

    try:
        # Stage 1: Structure generation
        ckpt_1 = ckpt / "01_structures.json"
        structures = _load_checkpoint(ckpt_1) or []
        if not structures:
            logger.info("=== Stage 1: Generating structures ===")
            structures = await generate_all_structures_async(client, counts)
            _save_checkpoint(structures, ckpt_1)

        # Stage 2: Base item generation
        ckpt_2 = ckpt / "02_base_items.json"
        base_items = _load_checkpoint(ckpt_2) or []
        done_ids = {x.get("family_id", "") for x in base_items}
        remaining_structures = [s for s in structures if s.get("family_id", "") not in done_ids]
        if remaining_structures:
            logger.info("=== Stage 2: Generating base items (%d remaining) ===", len(remaining_structures))

            async def _base_worker(s: dict[str, Any]) -> dict[str, Any] | None:
                block = s.get("task_family", "")
                gen = block_item_generators_async.get(block)
                if not gen:
                    logger.warning("No base item generator for block %s, skipping %s", block, s.get("family_id"))
                    return None
                logger.info("Generating base item for %s...", s.get("family_id"))
                base_item = await gen(client, s)
                merged = {**s, **base_item}
                merged.setdefault("base_item_id", f"{s.get('family_id', '')}_base")
                merged.setdefault("underlying_structure", s)
                return merged

            new_items = await _bounded_map(remaining_structures, _base_worker)
            base_items.extend([x for x in new_items if x is not None])
            _save_checkpoint(base_items, ckpt_2)

        # Stage 3: Variant generation
        ckpt_3 = ckpt / "03_variants.json"
        with_variants = _load_checkpoint(ckpt_3) or []
        done_ids = {x.get("family_id", "") for x in with_variants}
        remaining_families = [f for f in base_items if f.get("family_id", "") not in done_ids]
        if remaining_families:
            logger.info("=== Stage 3: Generating variants (%d remaining) ===", len(remaining_families))

            async def _variant_worker(family: dict[str, Any]) -> dict[str, Any]:
                variants = await generate_variants_async(client, family)
                return {**family, "normal_variants": variants}

            new_fams = await _bounded_map(remaining_families, _variant_worker)
            with_variants.extend(new_fams)
            _save_checkpoint(with_variants, ckpt_3)

        # Stage 4: Symbolic counterpart (programmatic, no API)
        ckpt_4 = ckpt / "04_symbolic.json"
        with_symbolic = _load_checkpoint(ckpt_4) or []
        if not with_symbolic:
            logger.info("=== Stage 4: Generating symbolic counterparts ===")
            with_symbolic = generate_all_symbolic(with_variants)
            _save_checkpoint(with_symbolic, ckpt_4)

        # Stage 5: MCQ generation
        ckpt_5 = ckpt / "05_mcq.json"
        with_mcq = _load_checkpoint(ckpt_5) or []
        done_ids = {x.get("family_id", "") for x in with_mcq}
        remaining_families = [f for f in with_symbolic if f.get("family_id", "") not in done_ids]
        if remaining_families:
            logger.info("=== Stage 5: Generating MCQ (%d remaining) ===", len(remaining_families))

            async def _mcq_worker(family: dict[str, Any]) -> dict[str, Any]:
                sym = family.get("symbolic_variants", {})
                if sym and sym.get("original"):
                    mcq = await generate_mcq_async(client, family)
                    return {**family, "mcq_variants": mcq}
                return {**family, "mcq_variants": {}}

            new_fams = await _bounded_map(remaining_families, _mcq_worker)
            with_mcq.extend(new_fams)
            _save_checkpoint(with_mcq, ckpt_5)

        # Export
        logger.info("=== Exporting final dataset ===")
        families = export_all(ckpt, out)
        if export_symbolic:
            # Note: currently exports one derived family per source family.
            export_symbolic_dataset(families, out)
        stats = export_stats(families, out / "stats.json")
        logger.info(
            "Pipeline complete. %d families, stats: %s",
            len(families),
            json.dumps(stats, indent=2),
        )
        return families
    finally:
        await client.aclose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the dataset synthesis pipeline")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model", default=MODEL, help="Model name for API calls")
    parser.add_argument("--kb", type=int, default=FAMILY_COUNTS["KB"], help="Number of KB families")
    parser.add_argument("--rb", type=int, default=FAMILY_COUNTS["RB"], help="Number of RB families")
    parser.add_argument("--hybrid", type=int, default=FAMILY_COUNTS["Hybrid"], help="Number of Hybrid families")
    parser.add_argument("--symbolic", type=int, default=SYMBOLIC_CONTROL_COUNT, help="Number of SymbolicControl families")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode (no external API calls / no ANTHROPIC_API_KEY needed)",
    )
    parser.add_argument(
        "--export_symbolic_dataset",
        action="store_true",
        help="Export a standalone SymbolicControl dataset (one per family) to output/symbolic_dataset.jsonl",
    )
    args = parser.parse_args()

    run_pipeline(
        output_dir=args.output_dir,
        model=args.model,
        family_counts={"KB": args.kb, "RB": args.rb, "Hybrid": args.hybrid},
        symbolic_count=args.symbolic,
        mock=args.mock,
        export_symbolic=args.export_symbolic_dataset,
    )
