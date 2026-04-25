"""Repair pilot_raw.json in-place.

Two passes:
  1. Scaffold-leak repair  — for families that have normal_variants but some
     scaffold variants contain the gold_answer, call repair_scaffold_leaks().
  2. Full re-generation    — for families whose normal_variants is None/missing
     (e.g. rb_006 timed out), call generate_variants() from scratch.

Usage:
    python -m dataset_synthesis.pilot_qc.repair_pilot

Requires APIClient.call_api to be implemented (see api_client.py).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_synthesis.api_client import APIClient
from dataset_synthesis.variants import (
    _find_leaking_scaffolds,
    generate_variants,
    repair_scaffold_leaks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PILOT_PATH = Path(__file__).parent / "pilot_raw.json"


def _load() -> list[dict]:
    with open(PILOT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: list[dict]) -> None:
    with open(PILOT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d families to %s", len(data), PILOT_PATH)


def repair(client: APIClient | None = None) -> None:
    if client is None:
        client = APIClient()

    families = _load()
    changed = 0

    for family in families:
        fid = family.get("family_id", "?")
        gold = str(family.get("gold_answer", ""))
        variants = family.get("normal_variants")

        # --- Pass 1: missing variants (full re-generation) ---
        if not variants:
            logger.info("[%s] No variants found — re-generating all 19...", fid)
            try:
                new_variants = generate_variants(client, family)
                family["normal_variants"] = new_variants
                family.pop("variant_error", None)
                changed += 1
                logger.info("[%s] Re-generation complete.", fid)
            except Exception as exc:
                logger.error("[%s] Re-generation failed: %s", fid, exc)
                family["variant_error"] = str(exc)
            continue

        # --- Pass 2: scaffold leak repair ---
        leaking = _find_leaking_scaffolds(variants, gold)
        if leaking:
            logger.info("[%s] Scaffold leaks in %s — repairing...", fid, leaking)
            try:
                repaired = repair_scaffold_leaks(client, family, variants, leaking)
                # Verify repair worked
                still_leaking = _find_leaking_scaffolds(repaired, gold)
                if still_leaking:
                    logger.warning(
                        "[%s] Still leaking after repair: %s", fid, still_leaking
                    )
                else:
                    logger.info("[%s] Scaffold repair successful.", fid)
                family["normal_variants"] = repaired
                changed += 1
            except Exception as exc:
                logger.error("[%s] Scaffold repair failed: %s", fid, exc)

    if changed:
        _save(families)
        logger.info("Repaired %d families.", changed)
    else:
        logger.info("Nothing to repair.")


if __name__ == "__main__":
    repair()
