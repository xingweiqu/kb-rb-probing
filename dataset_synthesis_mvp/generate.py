"""Main entry point for data generation pipeline."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .api_client import APIClient
from .config import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_AUTH_TOKEN,
    MODEL,
    FAMILY_COUNTS,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    CONCURRENCY
)
from .structures import generate_structures
from .base_items import generate_all_base_items
from .variants import generate_variants_for_family
from .symbolic import generate_all_symbolic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _load_checkpoint(path: Path) -> Any:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint {path} ({len(data) if isinstance(data, list) else 'N/A'} items)")
        return data
    return None


def _save_checkpoint(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved checkpoint {path}")


async def run_pipeline(
    output_dir: str = OUTPUT_DIR,
    model: str = MODEL,
    family_counts: dict[str, int] = None,
    mock: bool = False
) -> list[dict[str, Any]]:
    """Run the complete data generation pipeline."""
    base = Path(output_dir)
    ckpt = base / CHECKPOINT_DIR
    out = base / "output"

    # Initialize API client
    client = APIClient(
        model=model,
        mock=mock,
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_AUTH_TOKEN
    )

    counts = family_counts or FAMILY_COUNTS

    try:
        # Stage 1: Generate structures
        ckpt_1 = ckpt / "01_structures.json"
        structures = _load_checkpoint(ckpt_1)
        if not structures:
            logger.info("=== Stage 1: Generating structures ===")
            structures = await generate_structures(client, counts)
            _save_checkpoint(structures, ckpt_1)

        # Stage 2: Generate base items
        ckpt_2 = ckpt / "02_base_items.json"
        families = _load_checkpoint(ckpt_2)
        if not families:
            logger.info("=== Stage 2: Generating base items ===")
            families = await generate_all_base_items(client, structures, CONCURRENCY)
            _save_checkpoint(families, ckpt_2)

        # Stage 3: Generate variants
        ckpt_3 = ckpt / "03_variants.json"
        all_items = _load_checkpoint(ckpt_3)
        if not all_items:
            logger.info("=== Stage 3: Generating variants ===")
            all_items = []
            families_dict = {f["family_id"]: f for f in families}

            for family in families:
                logger.info(f"Generating variants for {family['family_id']}...")
                items = await generate_variants_for_family(client, family, all_items)
                all_items.extend(items)

            _save_checkpoint(all_items, ckpt_3)

        # Stage 4: Generate symbolic
        ckpt_4 = ckpt / "04_symbolic.json"
        all_items_with_symbolic = _load_checkpoint(ckpt_4)
        if not all_items_with_symbolic:
            logger.info("=== Stage 4: Generating symbolic variants ===")
            families_dict = {f["family_id"]: f for f in families}
            symbolic_items = generate_all_symbolic(all_items, families_dict)
            all_items_with_symbolic = all_items + symbolic_items
            _save_checkpoint(all_items_with_symbolic, ckpt_4)

        # Stage 5: Export
        logger.info("=== Stage 5: Exporting dataset ===")
        out.mkdir(parents=True, exist_ok=True)
        output_file = out / "dataset.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for item in all_items_with_symbolic:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(all_items_with_symbolic)} items to {output_file}")

        # Generate quality report
        failed = [i for i in all_items_with_symbolic if i.get("metadata", {}).get("failed")]
        repaired = [i for i in all_items_with_symbolic if i.get("metadata", {}).get("auto_repaired")]

        logger.info(f"\n{'='*60}")
        logger.info("QUALITY REPORT")
        logger.info(f"{'='*60}")
        total = len(all_items_with_symbolic)
        logger.info(f"Total items: {total}")
        if total > 0:
            logger.info(f"Failed: {len(failed)} ({len(failed)/total*100:.1f}%)")
            logger.info(f"Auto-repaired: {len(repaired)} ({len(repaired)/total*100:.1f}%)")
        else:
            logger.info("Failed: 0\nAuto-repaired: 0")
        logger.info(f"{'='*60}\n")

        return all_items_with_symbolic

    finally:
        await client.aclose()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate atomic capability benchmark dataset")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model", default=MODEL, help="Model name")
    parser.add_argument("--kb", type=int, default=FAMILY_COUNTS["KB"], help="KB family count")
    parser.add_argument("--rb", type=int, default=FAMILY_COUNTS["RB"], help="RB family count")
    parser.add_argument("--hybrid", type=int, default=FAMILY_COUNTS["Hybrid"], help="Hybrid family count")
    parser.add_argument("--mock", action="store_true", help="Mock mode (no API calls)")

    args = parser.parse_args()

    asyncio.run(run_pipeline(
        output_dir=args.output_dir,
        model=args.model,
        family_counts={"KB": args.kb, "RB": args.rb, "Hybrid": args.hybrid},
        mock=args.mock
    ))


if __name__ == "__main__":
    main()
