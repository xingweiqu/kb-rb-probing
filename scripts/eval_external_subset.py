"""Evaluate the behavior intervention matrix on an externally authored subset.

Reads `data/external_subset.jsonl` (or any path you pass in), runs each
(original, variant) pair through a model to get gold-answer log-probability,
computes Δlogp/token, and writes a CSV that mirrors the cell metric used in
the main matrix. Hidden-state probing is intentionally not run here — the
external subset is meant to be a behavior-only robustness check.

Local on the dev mac if model_outputs.jsonl is already produced server-side
for the listed model. Otherwise call from the server with a model path.

Usage:
    python -m scripts.eval_external_subset \
        --input data/external_subset.jsonl \
        --model_outputs runs/Qwen3-8B/model_outputs.jsonl \
        --output reports/arr_revision/external_subset_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_external(path: Path) -> list[dict]:
    items = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_comment" in obj:
            continue
        items.append(obj)
    return items


def _logprob_index(model_outputs_path: Path) -> dict[tuple[str, str, str], float]:
    """Build a (item_id, role, cot_state) -> mean gold logprob index.

    `role` is 'original' or 'variant'; we expect the external subset to
    feed paired prompts through extract_hidden_states.py with row IDs of
    the form `<item_id>__<role>` so they appear in model_outputs.jsonl.
    """
    out: dict[tuple[str, str, str], float] = {}
    for line in open(model_outputs_path, encoding="utf-8"):
        rec = json.loads(line)
        fid = rec.get("family_id") or rec.get("item_id") or ""
        variant = rec.get("variant", "")
        cot = rec.get("cot_state", "no_cot")
        lp = rec.get("gold_logprob_mean")
        if lp is None:
            continue
        if "__" in fid:
            stem, role = fid.split("__", 1)
        else:
            stem, role = fid, variant
        out[(stem, role, cot)] = float(lp)
    return out


def evaluate(input_path: Path, model_outputs_path: Path, output_path: Path,
             cot_state: str = "no_cot") -> None:
    items = _load_external(input_path)
    logger.info("loaded %d external items", len(items))

    if model_outputs_path.exists():
        index = _logprob_index(model_outputs_path)
    else:
        logger.warning("%s missing — emitting placeholder rows for the orchestrator",
                       model_outputs_path)
        index = {}

    rows = []
    for it in items:
        iid = it.get("item_id", "")
        orig_lp = index.get((iid, "original", cot_state))
        var_lp = index.get((iid, "variant", cot_state))
        delta = (var_lp - orig_lp) if (orig_lp is not None and var_lp is not None) else None
        rows.append({
            "item_id": iid,
            "family": it.get("family", ""),
            "cell_name": it.get("cell_name", ""),
            "source": it.get("source", ""),
            "gold_answer": it.get("gold_answer", ""),
            "expected_wrong_answer": it.get("expected_wrong_answer", ""),
            "orig_logprob_mean": orig_lp if orig_lp is not None else "",
            "variant_logprob_mean": var_lp if var_lp is not None else "",
            "delta_logprob_mean": f"{delta:.4f}" if delta is not None else "MISSING",
            "cot_state": cot_state,
            "notes": it.get("notes", ""),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader(); writer.writerows(rows)
    logger.info("wrote %d rows -> %s", len(rows), output_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/external_subset.jsonl")
    p.add_argument("--model_outputs", default="runs/Qwen3-8B/model_outputs.jsonl")
    p.add_argument("--output", default="reports/arr_revision/external_subset_results.csv")
    p.add_argument("--cot_state", default="no_cot")
    args = p.parse_args()
    evaluate(Path(args.input), Path(args.model_outputs), Path(args.output), args.cot_state)


if __name__ == "__main__":
    main()
