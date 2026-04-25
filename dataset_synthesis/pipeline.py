"""5-stage pipeline orchestrator with checkpoint management.

Usage:
    from dataset_synthesis.pipeline import run_pipeline
    run_pipeline(output_dir="./synthesis_output")

Or from CLI:
    python -m dataset_synthesis.pipeline --output_dir ./synthesis_output
"""

from __future__ import annotations

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
from .export import export_all
from .mcq import generate_all_mcq
from .stats import export_stats
from .structures import generate_all_structures, generate_base_items
from .symbolic import generate_all_symbolic
from .validate import validate_dataset
from .variants import generate_all_variants

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    base = Path(output_dir)
    ckpt = base / CHECKPOINT_DIR
    out = base / "output"

    client = APIClient(model=model)
    counts = family_counts or FAMILY_COUNTS

    # Stage 1: Structure generation
    ckpt_1 = ckpt / "01_structures.json"
    structures = _load_checkpoint(ckpt_1)
    if structures is None:
        logger.info("=== Stage 1: Generating structures ===")
        structures = generate_all_structures(client, counts)
        _save_checkpoint(structures, ckpt_1)

    # Stage 2: Base item generation
    ckpt_2 = ckpt / "02_base_items.json"
    base_items = _load_checkpoint(ckpt_2)
    if base_items is None:
        logger.info("=== Stage 2: Generating base items ===")
        done_ids = set()
        base_items = []
        for s in structures:
            fid = s.get("family_id", "")
            if fid in done_ids:
                continue
            base_items_partial = generate_base_items(client, [s])
            base_items.extend(base_items_partial)
            done_ids.add(fid)
            _save_checkpoint(base_items, ckpt_2)

    # Stage 3: Variant generation
    ckpt_3 = ckpt / "03_variants.json"
    with_variants = _load_checkpoint(ckpt_3)
    if with_variants is None:
        logger.info("=== Stage 3: Generating variants ===")
        with_variants = []
        for family in base_items:
            enriched = generate_all_variants(client, [family])
            with_variants.extend(enriched)
            _save_checkpoint(with_variants, ckpt_3)

    # Stage 4: Symbolic counterpart (programmatic, no API)
    ckpt_4 = ckpt / "04_symbolic.json"
    with_symbolic = _load_checkpoint(ckpt_4)
    if with_symbolic is None:
        logger.info("=== Stage 4: Generating symbolic counterparts ===")
        with_symbolic = generate_all_symbolic(with_variants)
        _save_checkpoint(with_symbolic, ckpt_4)

    # Stage 5: MCQ generation
    ckpt_5 = ckpt / "05_mcq.json"
    with_mcq = _load_checkpoint(ckpt_5)
    if with_mcq is None:
        logger.info("=== Stage 5: Generating MCQ ===")
        with_mcq = []
        for family in with_symbolic:
            enriched = generate_all_mcq(client, [family])
            with_mcq.extend(enriched)
            _save_checkpoint(with_mcq, ckpt_5)

    # Merge checkpoints → raw families, then validate
    logger.info("=== Validating families ===")
    from .export import merge_checkpoints
    raw_families = merge_checkpoints(ckpt)
    kept, val_results = validate_dataset(raw_families)
    n_discarded = len(raw_families) - len(kept)
    logger.info("Validation: %d total, %d discarded, %d kept", len(raw_families), n_discarded, len(kept))

    # Export validated dataset
    logger.info("=== Exporting final dataset ===")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    with open(out / "dataset.jsonl", "w", encoding="utf-8") as f:
        for fam in kept:
            f.write(json.dumps(fam, ensure_ascii=False) + "\n")

    # Validation report — only families that had issues
    with open(out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"family_id": r.family_id, "discard": r.discard, "issues": r.issues}
                for r in val_results
                if r.issues
            ],
            f, ensure_ascii=False, indent=2,
        )

    stats = export_stats(kept, out / "stats.json")
    logger.info("Pipeline complete. %d families kept.", len(kept))
    return kept


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the dataset synthesis pipeline")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model", default=MODEL, help="Model name for API calls")
    parser.add_argument("--kb", type=int, default=FAMILY_COUNTS["KB"], help="Number of KB families")
    parser.add_argument("--rb", type=int, default=FAMILY_COUNTS["RB"], help="Number of RB families")
    parser.add_argument("--hybrid", type=int, default=FAMILY_COUNTS["Hybrid"], help="Number of Hybrid families")
    parser.add_argument("--symbolic", type=int, default=SYMBOLIC_CONTROL_COUNT, help="Number of SymbolicControl families")
    args = parser.parse_args()

    run_pipeline(
        output_dir=args.output_dir,
        model=args.model,
        family_counts={"KB": args.kb, "RB": args.rb, "Hybrid": args.hybrid},
        symbolic_count=args.symbolic,
    )
