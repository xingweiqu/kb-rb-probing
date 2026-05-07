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
            wv = row_v.get("wrong_logprob_mean")
            if wv is None:
                continue
            gold_drops.append(d)
            mv = row_v["gold_logprob_mean"] - wv
            margin_vars.append(mv)
            fc_correct.append(1.0 if mv > 0 else 0.0)
            # margin_orig + wrong_gain require log p_original(A'), which is
            # only available when the original row was also scored against
            # the planted-wrong answer. Left optional.
            wo = row_o.get("wrong_logprob_mean")
            if wo is not None:
                wrong_gains.append(wv - wo)
                margin_origs.append(row_o["gold_logprob_mean"] - wo)
    out["n"] = len(deltas)
    out["delta_mean"], out["delta_lo"], out["delta_hi"] = boot_ci(deltas)
    out["abs_delta_mean"], _, _ = boot_ci([abs(x) for x in deltas])
    if capacity in WRONG_CLAIM_CELLS:
        out["gold_drop_mean"], out["gold_drop_lo"], out["gold_drop_hi"] = boot_ci(gold_drops)
        out["margin_var_mean"], out["margin_var_lo"], out["margin_var_hi"] = boot_ci(margin_vars)
        out["forced_choice_correct"], out["fc_lo"], out["fc_hi"] = boot_ci(fc_correct)
        if wrong_gains:
            out["wrong_gain_mean"], out["wrong_gain_lo"], out["wrong_gain_hi"] = boot_ci(wrong_gains)
            out["margin_orig_mean"], out["margin_orig_lo"], out["margin_orig_hi"] = boot_ci(margin_origs)
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
    """Scaling figure: family score (black) vs three atomic capacities
    (colored) per family, x = model size on log scale, two series per
    line (instruct solid, base dashed). One subplot per family."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    panels = [
        ("KB", ["KP", "KB", "KD"], "Knowledge"),
        ("RB", ["RP", "RB", "RD"], "Reasoning"),
        ("Hybrid", ["CP", "CB", "CD"], "Bridge"),
    ]
    cap_colors = {"KP": "#1f77b4", "KB": "#ff7f0e", "KD": "#2ca02c",
                  "RP": "#1f77b4", "RB": "#ff7f0e", "RD": "#2ca02c",
                  "CP": "#1f77b4", "CB": "#ff7f0e", "CD": "#2ca02c"}
    for ax, (fam, caps, title) in zip(axes, panels):
        ax2 = ax.twinx()
        for cot in ("no_cot",):
            for is_inst, ls, marker in [(True, "-", "o"), (False, "--", "s")]:
                fam_pts = []
                for r in family_rows:
                    if (r["family"] == fam and r["cot_state"] == cot
                            and _is_instruct(r["model"]) == is_inst):
                        sz = _model_size_b(r["model"])
                        if sz is None:
                            continue
                        v = r["mean_orig_gold_lp"]
                        if not _isnan(v):
                            fam_pts.append((sz, v))
                fam_pts.sort()
                if fam_pts:
                    xs, ys = zip(*fam_pts)
                    ax.plot(xs, ys, color="black", linestyle=ls, marker=marker,
                            markersize=6, alpha=0.85, linewidth=2.0,
                            label=f"family score ({'instruct' if is_inst else 'base'})")
                for cap in caps:
                    cap_pts = []
                    for r in capacity_rows:
                        if (r["capacity"] == cap and r["cot_state"] == cot
                                and _is_instruct(r["model"]) == is_inst):
                            sz = _model_size_b(r["model"])
                            if sz is None:
                                continue
                            v = r.get("abs_delta_mean")
                            if v is not None and not _isnan(v):
                                cap_pts.append((sz, v))
                    cap_pts.sort()
                    if cap_pts:
                        xs, ys = zip(*cap_pts)
                        ax2.plot(xs, ys, color=cap_colors[cap], linestyle=ls,
                                 marker=marker, markersize=5, alpha=0.7,
                                 label=f"{cap} ({'instruct' if is_inst else 'base'})")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("model size (B params, log)", fontsize=12)
        ax.set_ylabel("family score: mean orig gold logp", fontsize=12)
        ax2.set_ylabel("|mean Δlogp/token| per capacity", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax2.tick_params(axis="y", labelsize=11)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, framealpha=0.9, ncol=2)
    fig.suptitle("Family score vs atomic-capacity profile (no-CoT)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", figpath)


def _model_size_b(name: str) -> float | None:
    """Approximate parameter count in billions, parsed from model name."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)B", name)
    return float(m.group(1)) if m else None


def _is_instruct(name: str) -> bool:
    return not name.endswith("-Base")


def plot_wrong_claim_margin(capacity_rows: list[dict], figpath: Path) -> None:
    """Scaling figure: x = model size (log B params), y = signal magnitude.
    One subplot per cell (KD / RD / CD), two series per subplot (instruct
    solid, base dashed). Two y-axes per subplot (gold_drop blue, margin
    red) so the empirical claim "gold_drop ↑ AND margin ↑ as scale grows"
    is visible at a glance. Error bars are 95% bootstrap CIs by family_id."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    panels = [("KD", "Wrong-Claim (Knowledge)"),
              ("RD", "Wrong-Step (Reasoning)"),
              ("CD", "Wrong-Bridge (Hybrid)")]
    for ax, (cap, title) in zip(axes, panels):
        rows = [r for r in capacity_rows
                if r["capacity"] == cap and r["cot_state"] == "no_cot"
                and "gold_drop_mean" in r and not _isnan(r["gold_drop_mean"])]
        if not rows:
            ax.set_visible(False)
            continue
        # Group by instruct/base, attach size
        for r in rows:
            r["_size"] = _model_size_b(r["model"])
        rows = [r for r in rows if r["_size"] is not None]
        instruct = sorted([r for r in rows if _is_instruct(r["model"])],
                          key=lambda r: r["_size"])
        base     = sorted([r for r in rows if not _is_instruct(r["model"])],
                          key=lambda r: r["_size"])

        ax2 = ax.twinx()
        for series, marker, ls, label_suffix in [
                (instruct, "o", "-",  "instruct"),
                (base,     "s", "--", "base"),
        ]:
            if not series:
                continue
            xs = [r["_size"] for r in series]
            gd = [r["gold_drop_mean"] for r in series]
            gd_lo = [r["gold_drop_lo"] for r in series]
            gd_hi = [r["gold_drop_hi"] for r in series]
            mv = [r["margin_var_mean"] for r in series]
            mv_lo = [r["margin_var_lo"] for r in series]
            mv_hi = [r["margin_var_hi"] for r in series]
            ax.errorbar(xs, gd,
                        yerr=[[g - l for g, l in zip(gd, gd_lo)],
                              [h - g for g, h in zip(gd, gd_hi)]],
                        fmt=marker, linestyle=ls, color="#1f6feb",
                        markersize=7, capsize=3, alpha=0.85,
                        label=f"gold_drop ({label_suffix})")
            ax2.errorbar(xs, mv,
                         yerr=[[m - l for m, l in zip(mv, mv_lo)],
                               [h - m for m, h in zip(mv, mv_hi)]],
                         fmt=marker, linestyle=ls, color="#d6336c",
                         markersize=7, capsize=3, alpha=0.85,
                         label=f"margin_variant ({label_suffix})")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("model size (B params, log)", fontsize=12)
        ax.set_ylabel("gold log-prob drop  (variant − orig)", color="#1f6feb", fontsize=12)
        ax2.set_ylabel("variant margin  (gold − wrong)", color="#d6336c", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax2.tick_params(axis="y", labelsize=11)
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5, zorder=0)
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle("Wrong-distractor cells: confidence cost vs decision margin (scaling)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", figpath)


def _isnan(v) -> bool:
    try:
        return v != v   # NaN-safe
    except TypeError:
        return False


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
