"""Statistics report generation."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .configs.defaults import ATOMIC_CAPABILITY_MAP, VARIANT_TYPES

logger = logging.getLogger(__name__)


def compute_stats(families: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset statistics."""
    total = len(families)
    block_counts = Counter(f.get("task_family", "unknown") for f in families)
    sub_family_counts = Counter(f.get("sub_family", "unknown") for f in families)

    # Variant coverage
    variant_coverage: dict[str, int] = {vt: 0 for vt in VARIANT_TYPES}
    for f in families:
        nv = f.get("normal_variants", {})
        for vt in VARIANT_TYPES:
            if vt in nv:
                variant_coverage[vt] += 1

    variant_coverage_pct = {
        vt: round(count / total * 100, 1) if total > 0 else 0
        for vt, count in variant_coverage.items()
    }

    # Symbolic coverage
    symbolic_count = sum(
        1 for f in families
        if f.get("symbolic_variants") and isinstance(f["symbolic_variants"], dict)
        and f["symbolic_variants"].get("entity_map")
    )

    # MCQ coverage
    mcq_count = sum(
        1 for f in families
        if f.get("mcq_variants") and isinstance(f["mcq_variants"], dict)
        and len(f["mcq_variants"]) > 0
    )

    # Atomic capability coverage
    capability_coverage: dict[str, dict[str, Any]] = {}
    for cap, vtypes in ATOMIC_CAPABILITY_MAP.items():
        covered = sum(1 for f in families if all(
            vt in f.get("normal_variants", {}) for vt in vtypes
        ))
        capability_coverage[cap] = {
            "variants": vtypes,
            "families_with_full_coverage": covered,
            "coverage_pct": round(covered / total * 100, 1) if total > 0 else 0,
        }

    return {
        "total_families": total,
        "block_counts": dict(block_counts),
        "sub_family_counts": dict(sub_family_counts),
        "variant_coverage_counts": variant_coverage,
        "variant_coverage_pct": variant_coverage_pct,
        "symbolic_families": symbolic_count,
        "symbolic_coverage_pct": round(symbolic_count / total * 100, 1) if total > 0 else 0,
        "mcq_families": mcq_count,
        "mcq_coverage_pct": round(mcq_count / total * 100, 1) if total > 0 else 0,
        "atomic_capability_coverage": capability_coverage,
    }


def export_stats(families: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    """Compute and export statistics to a JSON file."""
    stats = compute_stats(families)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Stats exported to %s", output_path)
    return stats
