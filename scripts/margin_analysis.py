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
                            figpath: Path,
                            model_filter: list[str] | None = None) -> None:
    """3 rows (Knowledge / Reasoning / Bridge) x 4 cols (family + 3 caps).
    First column: family score (mean orig gold logp); remaining columns:
    |mean Δlogp/token| for KP/KB/KD etc. Filtered to a single coherent
    series so the trend is readable; default is Qwen3 instruct (0.6B,
    1.7B, 4B, 8B) — the cleanest scaling line.

    Reading rule: if a family score and an atomic-capacity signal in the
    same row move in different directions, the family number is hiding
    the capacity-level direction. That is the §6.2 thesis."""
    import matplotlib.pyplot as plt

    if model_filter is None:
        model_filter = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    panels = [
        ("KB", ["KP", "KB", "KD"], "Knowledge"),
        ("RB", ["RP", "RB", "RD"], "Reasoning"),
        ("Hybrid", ["CP", "CB", "CD"], "Bridge"),
    ]

    def _series(rows, key_fn):
        pts = []
        for r in rows:
            if r["model"] not in model_filter:
                continue
            sz = _model_size_b(r["model"])
            if sz is None:
                continue
            v = key_fn(r)
            if v is None or _isnan(v):
                continue
            pts.append((sz, v, r["model"]))
        pts.sort()
        return pts

    def _draw(ax, pts, color):
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        names = [p[2] for p in pts]
        ax.plot(xs, ys, color=color, linestyle="-", marker="o",
                markersize=12, linewidth=2.8, alpha=0.95)
        for x, y, n in zip(xs, ys, names):
            short = n.replace("Qwen3-", "").replace("Qwen3.5-", "")
            ax.annotate(short, (x, y), xytext=(8, 8),
                        textcoords="offset points",
                        fontsize=10, fontweight="bold")

    for row, (fam, caps, title) in enumerate(panels):
        # Column 0: family score
        ax = axes[row, 0]
        fam_subset = [r for r in family_rows
                      if r["family"] == fam and r["cot_state"] == "no_cot"]
        _draw(ax, _series(fam_subset, lambda r: r["mean_orig_gold_lp"]),
              color="black")
        ax.set_xscale("log")
        ax.set_title(f"{title} family score", fontsize=14, fontweight="bold")
        ax.set_ylabel("mean orig gold logp", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.3)

        # Columns 1..3: per-capacity
        for col, cap in enumerate(caps, start=1):
            ax = axes[row, col]
            cap_subset = [r for r in capacity_rows
                          if r["capacity"] == cap and r["cot_state"] == "no_cot"]
            _draw(ax, _series(cap_subset, lambda r: r.get("abs_delta_mean")),
                  color="#1f6feb")
            ax.set_xscale("log")
            ax.set_title(f"{title}: {cap}", fontsize=14, fontweight="bold")
            ax.set_ylabel("|mean Δlogp/token|", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("model size (B params, log)", fontsize=12)
    fig.suptitle("Family score vs atomic-capacity scaling (Qwen3 instruct, no-CoT)",
                 fontsize=17, fontweight="bold", y=1.0)
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


def plot_wrong_claim_margin(capacity_rows: list[dict], figpath: Path,
                            model_filter: list[str] | None = None,
                            title_suffix: str = "") -> None:
    """3 rows (KD / RD / CD) x 2 cols (gold_drop / margin_variant).
    Filtered to a single coherent series so trends are visible.
    Default filter: Qwen3 instruct only (the cleanest series)."""
    import matplotlib.pyplot as plt

    if model_filter is None:
        model_filter = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    panels = [("KD", "Wrong-Claim (Knowledge)"),
              ("RD", "Wrong-Step (Reasoning)"),
              ("CD", "Wrong-Bridge (Hybrid)")]

    def _series(cap, key_mean, key_lo, key_hi):
        pts = []
        for r in capacity_rows:
            if r["capacity"] != cap or r["cot_state"] != "no_cot":
                continue
            if r["model"] not in model_filter:
                continue
            sz = _model_size_b(r["model"])
            if sz is None:
                continue
            v = r.get(key_mean)
            if v is None or _isnan(v):
                continue
            lo = r.get(key_lo); hi = r.get(key_hi)
            pts.append((sz, v, r["model"],
                        v - lo if lo is not None and not _isnan(lo) else 0.0,
                        hi - v if hi is not None and not _isnan(hi) else 0.0))
        pts.sort()
        return pts

    for row, (cap, title) in enumerate(panels):
        for col, (key_mean, key_lo, key_hi, ylabel, header, color) in enumerate([
                ("gold_drop_mean", "gold_drop_lo", "gold_drop_hi",
                 r"log $p_v$(gold) − log $p_o$(gold)",
                 "Confidence cost", "#1f6feb"),
                ("margin_var_mean", "margin_var_lo", "margin_var_hi",
                 r"log $p_v$(gold) − log $p_v$(wrong)",
                 "Decision margin", "#d6336c"),
        ]):
            ax = axes[row, col]
            pts = _series(cap, key_mean, key_lo, key_hi)
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                names = [p[2] for p in pts]
                lerr = [p[3] for p in pts]
                herr = [p[4] for p in pts]
                ax.errorbar(xs, ys, yerr=[lerr, herr],
                            color=color, linestyle="-", marker="o",
                            markersize=14, linewidth=3.0, capsize=6,
                            elinewidth=2.0, alpha=0.95)
                # Annotate each point with model name
                for x, y, n in zip(xs, ys, names):
                    short = n.replace("Qwen3-", "").replace("Qwen3.5-", "")
                    ax.annotate(short, (x, y), xytext=(8, 8),
                                textcoords="offset points",
                                fontsize=11, fontweight="bold")
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.4, zorder=0)
            ax.set_xscale("log")
            ax.set_title(f"{title}: {header}", fontsize=15, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=13)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("model size (B params, log)", fontsize=13)
    fig.suptitle(f"Wrong-distractor cells: confidence cost vs decision margin{title_suffix}",
                 fontsize=18, fontweight="bold", y=1.0)
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
