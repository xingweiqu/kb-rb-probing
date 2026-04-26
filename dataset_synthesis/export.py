"""Export — merge checkpoints into final dataset files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .schema import Family
from .symbolic import derive_symbolic_control_family

logger = logging.getLogger(__name__)


def merge_checkpoints(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Merge all stage checkpoints into complete family dicts.

    Reads checkpoints in order and merges by family_id.
    """
    stages = [
        "01_structures.json",
        "02_base_items.json",
        "03_variants.json",
        "04_symbolic.json",
        "05_mcq.json",
    ]

    families_by_id: dict[str, dict[str, Any]] = {}

    for stage_file in stages:
        path = checkpoint_dir / stage_file
        if not path.exists():
            logger.warning("Checkpoint %s not found, skipping", path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        for item in data:
            fid = item.get("family_id", "")
            if not fid:
                continue
            if fid in families_by_id:
                families_by_id[fid].update(item)
            else:
                families_by_id[fid] = item

    return list(families_by_id.values())


def export_json(families: list[dict[str, Any]], output_path: Path) -> None:
    """Export the full dataset as a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(families, f, ensure_ascii=False, indent=2)
    logger.info("Exported %d families to %s", len(families), output_path)


def export_jsonl(families: list[dict[str, Any]], output_path: Path) -> None:
    """Export the dataset as JSONL (one family per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for family in families:
            f.write(json.dumps(family, ensure_ascii=False) + "\n")
    logger.info("Exported %d families to %s", len(families), output_path)


def export_all(
    checkpoint_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Merge checkpoints and export to all formats.

    Returns the merged family list.
    """
    families = merge_checkpoints(checkpoint_dir)
    logger.info("Merged %d families from checkpoints", len(families))

    export_json(families, output_dir / "dataset.json")
    export_jsonl(families, output_dir / "dataset.jsonl")

    return families


def export_symbolic_dataset(
    families: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Export a standalone SymbolicControl dataset derived from families.

    One derived symbolic family per source family.
    """
    derived: list[dict[str, Any]] = []
    for f in families:
        sym = derive_symbolic_control_family(f)
        if sym is not None:
            derived.append(sym)

    export_json(derived, output_dir / "symbolic_dataset.json")
    export_jsonl(derived, output_dir / "symbolic_dataset.jsonl")
    logger.info("Exported %d SymbolicControl families to %s", len(derived), output_dir)
    return derived
