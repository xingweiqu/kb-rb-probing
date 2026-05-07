"""Family-vs-capacity, wrong-claim margin, and bootstrap CIs.

Reads, for each model:
    runs/<model>/model_outputs_with_wrong.jsonl
        Per-item rows with gold_logprob_mean and (for wrongclaim /
        wrong_intermediate / wrong_bridge) wrong_logprob_mean.

Computes per (model, capacity, cot_state):
    family_score:  mean log p_original(gold) over the 25 originals in the
                   capacity's task family
    delta_logp:    mean log p_v(gold) - log p_o(gold)  (per-item paired)
    abs_delta:     mean |delta_logp|
    For wrong-claim cells (KD, RD, CD):
        gold_drop:        mean log p_v(gold) - log p_o(gold)        (= delta_logp)
        wrong_gain:       mean log p_v(wrong) - log p_o(wrong)
        margin_original:  mean log p_o(gold) - log p_o(wrong)
        margin_variant:   mean log p_v(gold) - log p_v(wrong)
        margin_drop:      margin_variant - margin_original
        forced_choice:    fraction of items with margin_variant > 0

Bootstrap CIs over family_id (1000 resamples) for every reported number.

Outputs:
    reports/margin/family_vs_capacity.csv
    reports/margin/wrong_claim_margin.csv
    figures/_cross_model/fig_family_vs_capacity.png
    figures/_cross_model/fig_wrong_claim_margin.png

Usage (local, after pulling model_outputs_with_wrong.jsonl from server):
    python -m scripts.margin_analysis \\
        --runs runs/Qwen3-1.7B-Base runs/Qwen3-4B-Base \\
               runs/Qwen3-8B-Base runs/Qwen3-8B \\
        --output_dir reports/margin \\
        --figure_dir figures/_cross_model
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

logger = logging.getLogger(__name__)


VARIANT_TO_CAPACITY = {
    "hint": "KP",                "paraphrase": "KB",          "wrongclaim": "KD",
    "scaffold": "RP",            "rule_removal": "RB",        "wrong_intermediate": "RD",
    "explicit_fact": "CP",       "retrieval_blocked": "CB",   "wrong_bridge": "CD",
}
WRONG_CLAIM_CELLS = {"KD", "RD", "CD"}
TASK_FAMILY_OF = {
    "KP": "KB", "KB": "KB", "KD": "KB",
    "RP": "RB", "RB": "RB", "RD": "RB",
    "CP": "Hybrid", "CB": "Hybrid", "CD": "Hybrid",
}
N_BOOT = 1000


def load_outputs(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    # Filter symbolic-mode out so analysis matches paper.
    rows = [r for r in rows if r.get("mode", "natural") != "symbolic"]
    return rows


def index_by_pair(rows: list[dict]) -> dict[tuple, dict]:
    """Index by (family_id, variant, cot_state). Each row = one observation."""
    idx = {}
    for r in rows:
        key = (r["family_id"], r["variant"], r["cot_state"])
        idx[key] = r
    return idx


def boot_ci(values: list[float], n_boot: int = N_BOOT,
            alpha: float = 0.05) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(0xC1)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return mean(values), lo, hi


def family_score(idx: dict, family_prefix: str, cot: str) -> tuple[float, float, float]:
    """Mean log p_original(gold) over originals in a family, with bootstrap CI."""
    vals = [r["gold_logprob_mean"] for (fid, v, c), r in idx.items()
            if v == "original" and c == cot and fid.startswith(family_prefix)]
    return boot_ci(vals)


def capacity_delta(idx: dict, capacity: str, cot: str) -> dict:
    out = {"capacity": capacity, "cot_state": cot}
    family_prefix = {"KB": "kb_", "RB": "rb_", "Hybrid": "hybrid_"}[TASK_FAMILY_OF[capacity]]
    variant = {v: k for k, v in VARIANT_TO_CAPACITY.items()}[capacity]
    deltas = []
    gold_drops, wrong_gains, margin_origs, margin_vars = [], [], [], []
    fc_correct = []
    for (fid, v, c), row_v in idx.items():
        if v != variant or c != cot or not fid.startswith(family_prefix):
            continue
        row_o = idx.get((fid, "original", c))
        if row_o is None:
            continue
        d = row_v["gold_logprob_mean"] - row_o["gold_logprob_mean"]
        deltas.append(d)
        if capacity in WRONG_CLAIM_CELLS:
            wo = row_o.get("wrong_logprob_mean")
            wv = row_v.get("wrong_logprob_mean")
            if wo is None or wv is None:
                continue
            gold_drops.append(d)
            wrong_gains.append(wv - wo)
            margin_origs.append(row_o["gold_logprob_mean"] - wo)
            mv = row_v["gold_logprob_mean"] - wv
            margin_vars.append(mv)
            fc_correct.append(1.0 if mv > 0 else 0.0)
    out["n"] = len(deltas)
    out["delta_mean"], out["delta_lo"], out["delta_hi"] = boot_ci(deltas)
    out["abs_delta_mean"], _, _ = boot_ci([abs(x) for x in deltas])
    if capacity in WRONG_CLAIM_CELLS:
        out["gold_drop_mean"], out["gold_drop_lo"], out["gold_drop_hi"] = boot_ci(gold_drops)
        out["wrong_gain_mean"], out["wrong_gain_lo"], out["wrong_gain_hi"] = boot_ci(wrong_gains)
        out["margin_orig_mean"], out["margin_orig_lo"], out["margin_orig_hi"] = boot_ci(margin_origs)
        out["margin_var_mean"], out["margin_var_lo"], out["margin_var_hi"] = boot_ci(margin_vars)
        out["margin_drop_mean"] = out["margin_var_mean"] - out["margin_orig_mean"]
        out["forced_choice_correct"], out["fc_lo"], out["fc_hi"] = boot_ci(fc_correct)
    return out


def per_model_table(model: str, run_dir: Path) -> tuple[list[dict], list[dict]]:
    path = run_dir / "model_outputs_with_wrong.jsonl"
    if not path.exists():
        path = run_dir / "model_outputs.jsonl"  # fall back if wrong-scoring not run yet
    rows = load_outputs(path)
    idx = index_by_pair(rows)

    family_rows, capacity_rows = [], []
    for fam, prefix in [("KB", "kb_"), ("RB", "rb_"), ("Hybrid", "hybrid_")]:
        for cot in ("no_cot", "with_cot"):
            m, lo, hi = family_score(idx, prefix, cot)
            family_rows.append({
                "model": model, "family": fam, "cot_state": cot,
                "mean_orig_gold_lp": m, "lo": lo, "hi": hi,
            })
    for cap in VARIANT_TO_CAPACITY.values():
        for cot in ("no_cot", "with_cot"):
            row = capacity_delta(idx, cap, cot)
            row["model"] = model
            capacity_rows.append(row)
    return family_rows, capacity_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", path, len(rows))


def plot_family_vs_capacity(family_rows: list[dict], capacity_rows: list[dict],
                            figpath: Path) -> None:
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in family_rows},
                    key=lambda m: ("Base" not in m, m))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("KB", ["KP", "KB", "KD"], "Knowledge"),
        ("RB", ["RP", "RB", "RD"], "Reasoning"),
        ("Hybrid", ["CP", "CB", "CD"], "Bridge"),
    ]
    for ax, (fam, caps, title) in zip(axes, panels):
        # Left axis: family score
        fam_vals = [next(r for r in family_rows
                         if r["model"] == m and r["family"] == fam
                         and r["cot_state"] == "no_cot")["mean_orig_gold_lp"]
                    for m in models]
        ax.plot(models, fam_vals, "k-o", label=f"{title} family score (orig logp)")
        ax.set_ylabel("mean orig gold log-prob")
        ax.set_title(title)
        ax2 = ax.twinx()
        for cap in caps:
            cap_vals = [next(r for r in capacity_rows
                             if r["model"] == m and r["capacity"] == cap
                             and r["cot_state"] == "no_cot")["abs_delta_mean"]
                        for m in models]
            ax2.plot(models, cap_vals, "--o", label=cap)
        ax2.set_ylabel("|mean Δlogp|")
        ax2.legend(loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Family-level score vs atomic-capacity profile (no-CoT)")
    fig.tight_layout()
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=180)
    plt.close(fig)
    logger.info("wrote %s", figpath)


def plot_wrong_claim_margin(capacity_rows: list[dict], figpath: Path) -> None:
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in capacity_rows},
                    key=lambda m: ("Base" not in m, m))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [("KD", "Wrong-Claim (Knowledge)"),
              ("RD", "Wrong-Step (Reasoning)"),
              ("CD", "Wrong-Bridge (Hybrid)")]
    for ax, (cap, title) in zip(axes, panels):
        rows = [r for r in capacity_rows
                if r["capacity"] == cap and r["cot_state"] == "no_cot"
                and "gold_drop_mean" in r]
        if not rows:
            ax.set_visible(False)
            continue
        rows.sort(key=lambda r: ("Base" not in r["model"], r["model"]))
        m_axis = [r["model"] for r in rows]
        gd = [r["gold_drop_mean"] for r in rows]
        gd_lo = [r["gold_drop_lo"] for r in rows]
        gd_hi = [r["gold_drop_hi"] for r in rows]
        mv = [r["margin_var_mean"] for r in rows]
        mv_lo = [r["margin_var_lo"] for r in rows]
        mv_hi = [r["margin_var_hi"] for r in rows]
        ax.errorbar(m_axis, gd,
                    yerr=[[g - l for g, l in zip(gd, gd_lo)],
                          [h - g for g, h in zip(gd, gd_hi)]],
                    fmt="o-", color="C0", label="gold_drop (confidence cost)")
        ax2 = ax.twinx()
        ax2.errorbar(m_axis, mv,
                     yerr=[[m - l for m, l in zip(mv, mv_lo)],
                           [h - m for m, h in zip(mv, mv_hi)]],
                     fmt="s--", color="C3", label="margin_variant (decision)")
        ax.set_title(title)
        ax.set_ylabel("gold log-prob drop", color="C0")
        ax2.set_ylabel("variant gold-vs-wrong margin", color="C3")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Wrong-Claim Robustness: confidence cost vs decision margin")
    fig.tight_layout()
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=180)
    plt.close(fig)
    logger.info("wrote %s", figpath)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more runs/<model>/ directories")
    p.add_argument("--output_dir", default="reports/margin")
    p.add_argument("--figure_dir", default="figures/_cross_model")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    family_rows, capacity_rows = [], []
    for run in args.runs:
        run_dir = Path(run)
        model = run_dir.name
        f, c = per_model_table(model, run_dir)
        family_rows.extend(f)
        capacity_rows.extend(c)

    out = Path(args.output_dir)
    write_csv(out / "family_vs_capacity.csv", family_rows)
    write_csv(out / "wrong_claim_margin.csv",
              [r for r in capacity_rows if r["capacity"] in WRONG_CLAIM_CELLS])
    write_csv(out / "all_capacity_rows.csv", capacity_rows)

    fig_dir = Path(args.figure_dir)
    plot_family_vs_capacity(family_rows, capacity_rows, fig_dir / "fig_family_vs_capacity.png")
    plot_wrong_claim_margin(capacity_rows, fig_dir / "fig_wrong_claim_margin.png")


if __name__ == "__main__":
    main()
