"""Roll up linear_probe.json + lora_probe.json + probe_geometry.json + label
distribution into a single ``summary.json`` per run.

Designed so the GPU side only needs to push this one file back to the repo —
hidden_*.npy and per-stage JSONs stay local for follow-up runs.

Run after the four probing stages finish. ``run_probing.sh`` invokes it
automatically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any


def _clean(obj: Any) -> Any:
    """Replace NaN/inf with None so the result is valid JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _trim_task_family(rec: dict) -> dict:
    """Drop the per-layer array from task_family records and surface the best layer."""
    by_layer = rec.get("by_layer") or []
    best_layer = rec.get("best_layer")
    best = None
    for r in by_layer:
        if r.get("layer") == best_layer:
            best = r
            break
    return {
        "probe_type": rec.get("probe_type", "linear"),
        "judge": rec.get("judge"),
        "rank": rec.get("rank"),
        "cot_state": rec.get("cot_state"),
        "pool": rec.get("pool"),
        "n_samples": rec.get("n_samples"),
        "class_counts": rec.get("class_counts"),
        "label_names": rec.get("label_names"),
        "best_layer": best_layer,
        "best_balanced_accuracy": (best or {}).get("balanced_accuracy"),
        "best_accuracy": (best or {}).get("accuracy"),
        "confusion_matrix": (best or {}).get("confusion_matrix"),
    }


def _trim_capability_top(records: list[dict], top_k: int = 10) -> list[dict]:
    keep = []
    for r in records:
        if r.get("skipped"):
            continue
        by_layer = r.get("by_layer") or []
        best_layer = r.get("best_layer")
        best = next((b for b in by_layer if b.get("layer") == best_layer), None)
        if best is None:
            continue
        keep.append({
            "probe_type": r.get("probe_type"),
            "judge": r.get("judge"),
            "rank": r.get("rank"),
            "capability": r.get("capability"),
            "cot_state": r.get("cot_state"),
            "pool": r.get("pool"),
            "n_samples": r.get("n_samples"),
            "class_counts": r.get("class_counts"),
            "best_layer": best_layer,
            "best_balanced_accuracy": best.get("balanced_accuracy"),
            "best_accuracy": best.get("accuracy"),
            "confusion_matrix": best.get("confusion_matrix"),
        })
    keep.sort(key=lambda d: d["best_balanced_accuracy"] or -1, reverse=True)
    return keep[:top_k]


def _capability_skip_stats(records: list[dict]) -> dict:
    skipped = sum(1 for r in records if r.get("skipped"))
    return {
        "total": len(records),
        "skipped": skipped,
        "kept": len(records) - skipped,
    }


def _capability_label_distribution(labels: dict) -> dict:
    """Per (capability, judge, cot_state) -> (pos, neg, drop, total)."""
    out: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for fid, by_cot in labels.items():
        for cot, cell_root in by_cot.items():
            for judge in ("binary", "delta", "zscore"):
                if isinstance(cell_root, dict) and judge in cell_root:
                    cell = cell_root[judge]
                elif judge == "binary" and not isinstance(cell_root, dict):
                    cell = cell_root
                else:
                    continue
                for cap, val in cell.items():
                    bucket = out.setdefault(cap, {}).setdefault(judge, {}).setdefault(
                        cot, {"pos": 0, "neg": 0, "drop": 0, "total": 0}
                    )
                    bucket["total"] += 1
                    if val is True:
                        bucket["pos"] += 1
                    elif val is False:
                        bucket["neg"] += 1
                    else:
                        bucket["drop"] += 1
    return out


def _logits_diagnostics(model_outputs_path: Path) -> dict:
    """Aggregate top-1 mismatch rate, gold rank distribution, and per-variant
    competitor stats from model_outputs.jsonl. Designed to surface why raw
    Δlogprob looks weird (e.g. RD with_cot positive shift).
    """
    if not model_outputs_path.exists():
        return {}
    rows_by = {}
    for line in open(model_outputs_path, encoding="utf-8"):
        r = json.loads(line)
        rank = r.get("gold_first_token_rank", -1)
        if rank is None:
            rank = -1
        key = (r.get("variant", "?"), r.get("cot_state", "?"))
        bucket = rows_by.setdefault(key, [])
        bucket.append({
            "family_id": r.get("family_id"),
            "rank": int(rank),
            "lp_mean": r.get("gold_logprob_mean"),
            "top1_token": (r.get("top_k_tokens") or [[None]])[0][0] if r.get("top_k_tokens") else None,
            "top1_lp": (r.get("top_k_logprobs") or [[None]])[0][0] if r.get("top_k_logprobs") else None,
            "gold_first_token": (r.get("gold_token_strs") or [None])[0] if r.get("gold_token_strs") else None,
        })
    summary = {}
    for (variant, cot), rows in rows_by.items():
        ranks = [x["rank"] for x in rows if x["rank"] >= 0]
        if not ranks:
            continue
        top1_match = sum(1 for x in rows if x["rank"] == 0)
        top5_match = sum(1 for x in rows if 0 <= x["rank"] < 5)
        summary.setdefault(variant, {})[cot] = {
            "n": len(rows),
            "top1_match_rate": top1_match / len(rows) if rows else 0.0,
            "top5_match_rate": top5_match / len(rows) if rows else 0.0,
            "median_rank": int(sorted(ranks)[len(ranks) // 2]),
            "max_rank": int(max(ranks)),
            "examples_top1_mismatch_with_high_lp": sorted(
                [x for x in rows if x["rank"] > 0 and x.get("lp_mean") is not None],
                key=lambda x: x.get("lp_mean") or float("-inf"),
                reverse=True,
            )[:3],
        }
    return summary


def _delta_stats(labels: dict) -> dict:
    """Aggregate Δlogprob distribution per (capability, cot_state)."""
    import statistics
    rows: dict[tuple[str, str], list[float]] = {}
    for fid, by_cot in labels.items():
        for cot, cell_root in by_cot.items():
            if not isinstance(cell_root, dict):
                continue
            values = cell_root.get("delta_value") or {}
            for cap, v in values.items():
                if v is None:
                    continue
                rows.setdefault((cap, cot), []).append(float(v))
    out: dict = {}
    for (cap, cot), vs in rows.items():
        if not vs:
            continue
        out.setdefault(cap, {})[cot] = {
            "n": len(vs),
            "mean": float(statistics.fmean(vs)),
            "median": float(statistics.median(vs)),
            "stdev": float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
            "min": float(min(vs)),
            "max": float(max(vs)),
        }
    return out


def _geometry_summary(geom: dict) -> dict:
    per_setting = []
    for s in geom.get("per_setting", []):
        sils = s.get("silhouette_per_layer") or []
        best_layer = s.get("best_silhouette_layer")
        best_sil = sils[best_layer] if best_layer is not None and best_layer < len(sils) else None
        per_setting.append({
            "cot_state": s.get("cot_state"),
            "pool": s.get("pool"),
            "n_samples": s.get("n_samples"),
            "class_counts": s.get("class_counts"),
            "best_silhouette_layer": best_layer,
            "best_silhouette": best_sil,
        })
    cot_shift_last_layer = []
    for s in geom.get("cot_shift", []):
        last_layer_norms = {
            cls: vals[-1] if vals else None
            for cls, vals in (s.get("shift_norm_per_layer") or {}).items()
        }
        cot_shift_last_layer.append({
            "pool": s.get("pool"),
            **last_layer_norms,
        })
    return {"per_setting": per_setting, "cot_shift_last_layer": cot_shift_last_layer}


def make_summary(run_dir: str, model_path: str, output: str | None = None) -> dict:
    p = Path(run_dir)
    output = output or str(p / "summary.json")

    linear = json.load(open(p / "linear_probe.json", encoding="utf-8")) if (p / "linear_probe.json").exists() else {"task_family": [], "capability": []}
    lora = json.load(open(p / "lora_probe.json", encoding="utf-8")) if (p / "lora_probe.json").exists() else {"task_family": [], "capability": []}
    geom = json.load(open(p / "probe_geometry.json", encoding="utf-8")) if (p / "probe_geometry.json").exists() else {}
    labels = json.load(open(p / "capability_labels.json", encoding="utf-8")) if (p / "capability_labels.json").exists() else {}
    logits_diag = _logits_diagnostics(p / "model_outputs.jsonl")

    summary = {
        "model_path": model_path,
        "run_dir": run_dir,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_family": {
            "linear": [_trim_task_family(r) for r in linear.get("task_family", [])],
            "lora": [_trim_task_family(r) for r in lora.get("task_family", [])],
        },
        "capability": {
            "label_distribution": _capability_label_distribution(labels),
            "delta_value_stats": _delta_stats(labels),
            "linear": {
                **_capability_skip_stats(linear.get("capability", [])),
                "probe_type": "linear",
                "top": _trim_capability_top(linear.get("capability", []), top_k=10),
            },
            "lora": {
                **_capability_skip_stats(lora.get("capability", [])),
                "probe_type": "lora",
                "top": _trim_capability_top(lora.get("capability", []), top_k=10),
            },
        },
        "geometry": _geometry_summary(geom),
        "logits_diagnostics": logits_diag,
    }

    summary = _clean(summary)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", output)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="probing run dir, e.g. runs/Qwen3-8B")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output", help="defaults to <run_dir>/summary.json")
    args = p.parse_args()
    make_summary(args.run_dir, args.model_path, args.output)


if __name__ == "__main__":
    main()
