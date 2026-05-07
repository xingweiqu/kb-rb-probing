"""Build a manual-audit sheet for the intervention dataset.

We sample paired (original, variant) items per (task family, intervention)
cell and emit a CSV the annotator can fill in. The intervention cell
naming follows the human-readable scheme used in figures and prose:

    KB family       : Hint Gain (KP)        / Paraphrase Drop (KB)         / Wrong-Claim Drop (KD)
    RB family       : Scaffold Gain (RP)    / Rule-Removal Drop (RB)       / Wrong-Step Drop (RD)
    Hybrid family   : Bridge-Fact Gain (CP) / Retrieval-Block Drop (CB)    / Wrong-Bridge Drop (CD)

Until the Wrong-Bridge variant is generated server-side (PART 1 of the ARR
revision plan), the CD column will be empty. We still emit a placeholder
row block so the audit sheet structure is final.

Usage:
    python -m scripts.create_manual_audit_sheet \
        --dataset runs/full_25/output/dataset.jsonl \
        --output reports/arr_revision/manual_audit_sample.csv \
        --per_cell 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


VARIANT_TO_CELL = {
    "hint": ("KB", "Hint Gain (KP)"),
    "paraphrase": ("KB", "Paraphrase Drop (KB)"),
    "wrongclaim": ("KB", "Wrong-Claim Drop (KD)"),
    "scaffold": ("RB", "Scaffold Gain (RP)"),
    "rule_removal": ("RB", "Rule-Removal Drop (RB)"),
    "wrong_intermediate": ("RB", "Wrong-Step Drop (RD)"),
    "explicit_fact": ("Hybrid", "Bridge-Fact Gain (CP)"),
    "retrieval_blocked": ("Hybrid", "Retrieval-Block Drop (CB)"),
    "wrong_bridge": ("Hybrid", "Wrong-Bridge Drop (CD)"),  # not yet in dataset
    # both_blocked is moved to auxiliary lower-bound control
}

CELL_ORDER = [
    "Hint Gain (KP)", "Paraphrase Drop (KB)", "Wrong-Claim Drop (KD)",
    "Scaffold Gain (RP)", "Rule-Removal Drop (RB)", "Wrong-Step Drop (RD)",
    "Bridge-Fact Gain (CP)", "Retrieval-Block Drop (CB)", "Wrong-Bridge Drop (CD)",
]

AUDIT_FIELDS = [
    "item_id", "backbone_id", "family", "cell_name", "surface_mode",
    "original_prompt", "variant_prompt", "gold_answer", "expected_wrong_answer",
    "validator_flags",
    "annotator_gold_same_answer",       # yes / no
    "annotator_perturbation_valid",     # yes / no
    "annotator_wrong_claim_plausible",  # yes / no / na
    "annotator_no_gold_leakage",        # yes / no
    "annotator_surface_artifact",       # yes / no
    "annotator_overall_pass",           # yes / no
    "comments",
]


def _expected_wrong_answer(item: dict) -> str:
    """Best-effort extraction of the wrong-answer expectation from metadata."""
    meta = item.get("metadata") or {}
    if item["variant"] == "wrongclaim":
        return str(meta.get("wrong_claim", "") or "").strip()
    if item["variant"] == "wrong_intermediate":
        return str(meta.get("wrong_claim", meta.get("wrong_intermediate", "")) or "").strip()
    if item["variant"] == "wrong_bridge":
        return str(meta.get("wrong_bridge_implied_answer", "") or "").strip()
    return ""


def _validator_flags(item: dict) -> str:
    meta = item.get("metadata") or {}
    flags = []
    if meta.get("auto_repaired"):
        flags.append("auto_repaired")
    if meta.get("failed"):
        flags.append("validator_failed")
    if meta.get("repair_type"):
        flags.append(f"repair:{meta['repair_type']}")
    return ";".join(flags)


def build_audit_sheet(dataset_path: Path, output_path: Path,
                      per_cell: int = 10, seed: int = 42) -> None:
    items = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
    rng = random.Random(seed)

    # group originals by family_id+mode for pairing
    by_backbone_mode = {}
    by_cell = defaultdict(list)
    for it in items:
        key = (it["family_id"], it["mode"])
        if it["variant"] == "original":
            by_backbone_mode[key] = it
        else:
            cell = VARIANT_TO_CELL.get(it["variant"])
            if cell is None:
                continue
            by_cell[cell].append(it)

    rows = []
    for cell_label in CELL_ORDER:
        family = next((f for v, (f, l) in VARIANT_TO_CELL.items() if l == cell_label), "Hybrid")
        candidates = []
        for (fam_label, current_label), bucket in by_cell.items():
            if current_label == cell_label:
                candidates = bucket
                break
        if not candidates:
            logger.warning("no items found for cell %s — emitting %d empty placeholders",
                           cell_label, per_cell)
            for k in range(per_cell):
                rows.append({
                    "item_id": f"PLACEHOLDER_{cell_label}_{k}",
                    "backbone_id": "",
                    "family": family,
                    "cell_name": cell_label,
                    "surface_mode": "",
                    "original_prompt": "",
                    "variant_prompt": "",
                    "gold_answer": "",
                    "expected_wrong_answer": "",
                    "validator_flags": "AWAITING_GENERATION",
                    **{f: "" for f in AUDIT_FIELDS[10:]},
                })
            continue

        # try to balance natural / symbolic 50/50
        nat = [it for it in candidates if it["mode"] == "natural"]
        sym = [it for it in candidates if it["mode"] == "symbolic"]
        rng.shuffle(nat); rng.shuffle(sym)
        half = per_cell // 2
        chosen = nat[:half] + sym[:per_cell - half]
        if len(chosen) < per_cell:
            extra = [it for it in candidates if it not in chosen]
            rng.shuffle(extra)
            chosen.extend(extra[: per_cell - len(chosen)])

        for it in chosen:
            orig_key = (it["family_id"], it["mode"])
            orig = by_backbone_mode.get(orig_key)
            rows.append({
                "item_id": f"{it['family_id']}_{it['variant']}_{it['mode']}",
                "backbone_id": it["family_id"],
                "family": it["task_family"],
                "cell_name": cell_label,
                "surface_mode": it["mode"],
                "original_prompt": (orig or {}).get("question", ""),
                "variant_prompt": it.get("question", ""),
                "gold_answer": it.get("gold_answer", ""),
                "expected_wrong_answer": _expected_wrong_answer(it),
                "validator_flags": _validator_flags(it),
                **{f: "" for f in AUDIT_FIELDS[10:]},
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %d rows -> %s", len(rows), output_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="runs/full_25/output/dataset.jsonl")
    p.add_argument("--output", default="reports/arr_revision/manual_audit_sample.csv")
    p.add_argument("--per_cell", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_audit_sheet(Path(args.dataset), Path(args.output), args.per_cell, args.seed)


if __name__ == "__main__":
    main()
