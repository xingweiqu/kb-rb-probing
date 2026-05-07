"""Plot figures from one or more probing summary.json files.

Designed for paper figures. Each invocation reads ``runs/<name>/summary.json``
files and writes PNGs to ``figures/<group>/`` (or any --output-dir). Doesn't
need any of the heavy artifacts (hidden_*.npy, etc.) — summary.json is
self-contained for everything we plot.

Figures produced (per model):
    fig_taskfamily_acc.png        Best balanced acc per (cot, pool) bar chart
    fig_taskfamily_cm.png         Confusion matrices, 2x2 grid over (cot, pool)
    fig_capability_heatmap.png    Capability x cot bacc heatmap (linear probe)
    fig_logits_diag.png           Per-variant top-1 / top-5 match rates
    fig_cot_shift.png             CoT shift L2 norm by class (last layer)
    fig_delta_distribution.png    Δlogprob distribution by capability x cot

Cross-model figures (when multiple summaries are provided):
    fig_scaling_taskfamily.png    task_family bacc vs model size
    fig_scaling_capability.png    capability bacc vs model size, faceted by cap

Usage:
    python -m probing_mvp.make_plots runs/Qwen3-8B/summary.json
    python -m probing_mvp.make_plots runs/*/summary.json --output-dir figures/all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


CAPABILITY_ORDER = ["KP", "KD", "KB", "RP", "RD", "RB", "CP", "CB", "CB-control"]
COT_ORDER = ["no_cot", "with_cot"]
POOL_ORDER = ["last", "mean"]
FAMILY_NAMES = ["KB", "RB", "Hybrid"]
FAMILY_COLORS = {"KB": "#2c7fb8", "RB": "#d95f0e", "Hybrid": "#7b3294"}
JUDGE_ORDER = ["binary", "delta", "zscore"]


def _model_name(summary: dict) -> str:
    return Path(summary.get("run_dir", "run")).name or "run"


def _model_size(name: str) -> float | None:
    """Extract numeric size in B from a model directory name like 'Qwen3-1.7B'."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def fig_taskfamily_acc(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    settings = []
    linear_vals, lora_vals = [], []
    for cot in COT_ORDER:
        for pool in POOL_ORDER:
            settings.append(f"{cot}\n{pool}")
            lin = next((r for r in summary["task_family"]["linear"]
                        if r["cot_state"] == cot and r["pool"] == pool), None)
            lor = next((r for r in summary["task_family"]["lora"]
                        if r["cot_state"] == cot and r["pool"] == pool), None)
            linear_vals.append(lin.get("best_balanced_accuracy") if lin else 0)
            lora_vals.append(lor.get("best_balanced_accuracy") if lor else 0)
    xs = np.arange(len(settings))
    ax.bar(xs - width/2, linear_vals, width, label="linear", color="#3182bd")
    ax.bar(xs + width/2, lora_vals, width, label="LoRA r=16", color="#e6550d")
    ax.set_xticks(xs)
    ax.set_xticklabels(settings)
    ax.set_ylim(0, 1.05)
    ax.axhline(1/3, color="gray", ls="--", lw=0.7, label="chance (1/3)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"task_family probe accuracy — {_model_name(summary)}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_taskfamily_cm(summary: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for ax, (cot, pool) in zip(axes.flat, [(c, p) for c in COT_ORDER for p in POOL_ORDER]):
        rec = next((r for r in summary["task_family"]["linear"]
                    if r["cot_state"] == cot and r["pool"] == pool), None)
        if not rec or not rec.get("confusion_matrix"):
            ax.set_visible(False); continue
        cm = np.array(rec["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=10)
        ax.set_xticks(range(len(FAMILY_NAMES))); ax.set_xticklabels(FAMILY_NAMES)
        ax.set_yticks(range(len(FAMILY_NAMES))); ax.set_yticklabels(FAMILY_NAMES)
        ax.set_xlabel("predicted")
        if pool == "last": ax.set_ylabel("true")
        bacc = rec.get("best_balanced_accuracy")
        ax.set_title(f"{cot}/{pool} (bacc={bacc:.2f}, layer {rec['best_layer']})", fontsize=10)
    fig.suptitle(f"task_family confusion (linear) — {_model_name(summary)}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _capability_bacc_grid(summary: dict, probe_kind: str = "linear") -> tuple[np.ndarray, list[str], list[str]]:
    """Return (grid, row_labels, col_labels) where rows are capabilities and
    cols are (judge, cot_state, pool) combos. Cell = best_balanced_accuracy.
    """
    records = summary["capability"][probe_kind].get("top", [])
    # also pull skipped placeholders if available — we want every (cap, cot, pool, judge) cell
    cell_lookup = {}
    for r in records:
        cell_lookup[(r["capability"], r.get("judge", "?"), r["cot_state"], r["pool"])] = r.get("best_balanced_accuracy")
    cols = []
    for judge in JUDGE_ORDER:
        for cot in COT_ORDER:
            for pool in POOL_ORDER:
                cols.append((judge, cot, pool))
    grid = np.full((len(CAPABILITY_ORDER), len(cols)), np.nan)
    for ri, cap in enumerate(CAPABILITY_ORDER):
        for ci, (judge, cot, pool) in enumerate(cols):
            v = cell_lookup.get((cap, judge, cot, pool))
            grid[ri, ci] = v if v is not None else np.nan
    col_labels = [f"{j}\n{c}/{p}" for (j, c, p) in cols]
    return grid, CAPABILITY_ORDER, col_labels


def fig_capability_heatmap(summary: dict, out: Path) -> None:
    grid, rows, cols = _capability_bacc_grid(summary, "linear")
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.6), 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.55 < v < 0.85 else "white")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="balanced accuracy")
    ax.set_title(f"capability probe (linear) — {_model_name(summary)}\nNaN = skipped or no signal")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_logits_diag(summary: dict, out: Path) -> None:
    diag = summary.get("logits_diagnostics", {})
    if not diag:
        logger.warning("no logits_diagnostics in summary; skipping fig_logits_diag")
        return
    variants = sorted(diag.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    width = 0.35
    for ax, cot in zip(axes, COT_ORDER):
        top1 = [diag[v].get(cot, {}).get("top1_match_rate", 0) for v in variants]
        top5 = [diag[v].get(cot, {}).get("top5_match_rate", 0) for v in variants]
        xs = np.arange(len(variants))
        ax.bar(xs - width/2, top1, width, label="top-1", color="#2c7fb8")
        ax.bar(xs + width/2, top5, width, label="top-5", color="#7fcdbb")
        ax.set_xticks(xs); ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        ax.set_title(cot)
        ax.set_ylim(0, 1.05); ax.set_ylabel("gold-token match rate")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"gold-token rank diagnostics — {_model_name(summary)}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_cot_shift(summary: dict, out: Path) -> None:
    shifts = summary.get("geometry", {}).get("cot_shift_last_layer", [])
    if not shifts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    pools = [s["pool"] for s in shifts]
    width = 0.25
    xs = np.arange(len(pools))
    for i, fam in enumerate(FAMILY_NAMES):
        vals = [s.get(fam, 0) or 0 for s in shifts]
        ax.bar(xs + (i - 1) * width, vals, width, label=fam, color=FAMILY_COLORS[fam])
    ax.set_xticks(xs); ax.set_xticklabels(pools)
    ax.set_ylabel("L2 displacement (with_cot − no_cot, last layer)")
    ax.set_title(f"CoT shift by task family — {_model_name(summary)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_delta_distribution(summary: dict, out: Path) -> None:
    stats = summary.get("capability", {}).get("delta_value_stats", {})
    if not stats:
        return
    caps = [c for c in CAPABILITY_ORDER if c in stats]
    fig, ax = plt.subplots(figsize=(max(7, len(caps) * 0.7), 4.5))
    width = 0.35
    xs = np.arange(len(caps))
    for i, cot in enumerate(COT_ORDER):
        means = [stats[c].get(cot, {}).get("mean", 0) for c in caps]
        stds = [stats[c].get(cot, {}).get("stdev", 0) for c in caps]
        ax.bar(xs + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
               label=cot, color="#2c7fb8" if cot == "no_cot" else "#fdae61")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(caps, rotation=45, ha="right")
    ax.set_ylabel("mean Δlogprob/token (variant − original)")
    ax.set_title(f"Δlogprob distribution by capability — {_model_name(summary)}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def per_model_figures(summary_path: Path, output_dir: Path) -> None:
    summary = json.load(open(summary_path, encoding="utf-8"))
    name = _model_name(summary)
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("plotting %s -> %s", name, model_dir)
    fig_taskfamily_acc(summary, model_dir / "fig_taskfamily_acc.png")
    fig_taskfamily_cm(summary, model_dir / "fig_taskfamily_cm.png")
    fig_capacity_matrix(summary, model_dir / "fig_capacity_matrix.png")
    fig_capability_heatmap(summary, model_dir / "fig_capability_heatmap.png")
    fig_logits_diag(summary, model_dir / "fig_logits_diag.png")
    fig_cot_shift(summary, model_dir / "fig_cot_shift.png")
    fig_delta_distribution(summary, model_dir / "fig_delta_distribution.png")


def fig_scaling_taskfamily(summaries: list[dict], out: Path) -> None:
    points = []
    for s in summaries:
        name = _model_name(s)
        size = _model_size(name)
        if size is None: continue
        is_base = "base" in name.lower()
        rec = next((r for r in s["task_family"]["linear"]
                    if r["cot_state"] == "no_cot" and r["pool"] == "last"), None)
        if rec:
            points.append((name, size, rec.get("best_balanced_accuracy") or 0, is_base))
    if not points:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for is_base, marker, label in [(False, "o", "instruct"), (True, "s", "base")]:
        sub = [(s, b, n) for (n, s, b, ib) in points if ib == is_base]
        if sub:
            sub.sort()
            sizes, baccs, names = zip(*[(s, b, n) for s, b, n in sub])
            ax.plot(sizes, baccs, marker=marker, label=label, markersize=8, linewidth=1.5)
            for s, b, n in zip(sizes, baccs, names):
                ax.annotate(n, (s, b), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(1/3, color="gray", ls="--", lw=0.7, label="chance")
    ax.set_xlabel("model size (B params, log scale)")
    ax.set_ylabel("task_family balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("task_family probe scaling")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_scaling_capability(summaries: list[dict], out: Path) -> None:
    """Faceted scaling plot — one subplot per capability, x=size, y=best bacc.
    Uses delta judge no_cot/last as the canonical setting.
    """
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for ax, cap in zip(axes.flat, CAPABILITY_ORDER):
        for is_base, marker in [(False, "o"), (True, "s")]:
            xs, ys, names = [], [], []
            for s in summaries:
                name = _model_name(s)
                size = _model_size(name)
                if size is None: continue
                if ("base" in name.lower()) != is_base: continue
                rec = next((r for r in s["capability"]["linear"].get("top", [])
                            if r.get("capability") == cap and r.get("judge") == "delta"
                            and r["cot_state"] == "no_cot" and r["pool"] == "last"), None)
                bacc = rec.get("best_balanced_accuracy") if rec else None
                if bacc is None or np.isnan(bacc): continue
                xs.append(size); ys.append(bacc); names.append(name)
            if xs:
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                label = "base" if is_base else "instruct"
                ax.plot(xs, ys, marker=marker, label=label, markersize=6, linewidth=1.2)
        ax.axhline(0.5, color="gray", ls="--", lw=0.6)
        ax.set_xscale("log")
        ax.set_title(cap, fontsize=10)
        ax.set_ylim(0.3, 1.0)
    for ax in axes[-1, :]: ax.set_xlabel("size (B)")
    for ax in axes[:, 0]: ax.set_ylabel("bacc")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=2)
    fig.suptitle("capability probe scaling (delta judge, no_cot, last)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Failure-diagnosis cross matrix: task family x atomic capability
# ----------------------------------------------------------------------------

# Capability ordered in 3x3 native blocks: KB-task uses KP/KB/KD; RB-task uses
# RP/RB/RD; Hybrid-task uses CP/CB/CD. Plus the double-block control column.
MATRIX_CAPS = ["KP", "KB", "KD", "RP", "RB", "RD", "CP", "CB", "CD"]
TASK_FAMILY_TO_CAPS = {
    "KB": ["KP", "KB", "KD"],
    "RB": ["RP", "RB", "RD"],
    "Hybrid": ["CP", "CB"],   # CD reserved (no variant generated)
}


def _bacc_lookup(summary: dict, capability: str, judge: str = "zscore",
                 cot: str = "no_cot", pool: str = "last") -> float | None:
    """Pull a single (capability, judge, cot, pool) balanced accuracy from
    summary.capability.linear.top. Returns None when the cell is missing,
    skipped, or has a NaN/null bacc (typical when a class collapses)."""
    for r in summary.get("capability", {}).get("linear", {}).get("top", []):
        if (r.get("capability") == capability and r.get("judge") == judge
                and r.get("cot_state") == cot and r.get("pool") == pool
                and not r.get("skipped")):
            v = r.get("best_balanced_accuracy")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)
    return None


def _delta_signal_lookup(summary: dict, capability: str,
                         cot: str = "no_cot") -> float | None:
    """Fallback metric: |mean Δlogprob| (per token, variant − original) under
    the named CoT state. Works when linear.top is sparse or truncated. Larger
    magnitude = stronger failure-mode signal regardless of direction."""
    stats = summary.get("capability", {}).get("delta_value_stats", {}).get(capability, {})
    cell = stats.get(cot)
    if not cell:
        return None
    m = cell.get("mean")
    if m is None:
        return None
    return float(abs(m))


def _build_capacity_matrix(summary: dict, judge: str = "zscore",
                           cot: str = "no_cot", pool: str = "last",
                           metric: str = "bacc") -> np.ndarray:
    """Return a 3x9 matrix indexed by (task_family, capability). Off-diagonal
    blocks are NaN by construction — those (task, capability) pairs have no
    variant in our dataset.

    metric:
      "bacc"   - pull probe balanced accuracy from summary.capability.linear.top
                 (only available for top-N cells emitted by make_summary).
      "delta"  - pull |mean Δlogprob/token| from delta_value_stats; available
                 for every (capability, cot_state). Use this when bacc cells
                 are missing because make_summary truncated to top-N.
    """
    M = np.full((len(FAMILY_NAMES), len(MATRIX_CAPS)), np.nan)
    for ri, fam in enumerate(FAMILY_NAMES):
        native = TASK_FAMILY_TO_CAPS.get(fam, [])
        for ci, cap in enumerate(MATRIX_CAPS):
            if cap not in native:
                continue
            if metric == "bacc":
                v = _bacc_lookup(summary, cap, judge=judge, cot=cot, pool=pool)
            elif metric == "delta":
                v = _delta_signal_lookup(summary, cap, cot=cot)
            else:
                raise ValueError(f"unknown metric {metric!r}")
            if v is not None:
                M[ri, ci] = v
    return M


def _annotate_matrix(ax, M, cell_text_fn=None) -> None:
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#888")
            else:
                txt = cell_text_fn(v) if cell_text_fn else f"{v:.2f}"
                color = "white" if v > 0.75 or v < 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)


def _matrix_color_range(metric: str) -> tuple[float, float, str]:
    if metric == "bacc":
        return 0.4, 1.0, "balanced accuracy"
    if metric == "delta":
        return 0.0, 3.0, "|Δlogprob/token| (variant − original)"
    raise ValueError(f"unknown metric {metric!r}")


def fig_capacity_matrix(summary: dict, out: Path,
                        judge: str = "zscore", cot: str = "no_cot",
                        pool: str = "last", metric: str = "delta") -> None:
    """3x9 task-family × atomic-capability heatmap.

    Sparse by construction: only cells where the variant defining that
    capability is native to the task family carry signals. The
    off-diagonal NaN blocks make the failure-diagnosis structure visible.

    metric defaults to ``delta`` (|mean Δlogprob/token|), which is dense over
    every (capability, cot) cell. Pass ``metric="bacc"`` to plot probe
    balanced accuracy instead — only available for cells in linear.top.
    """
    M = _build_capacity_matrix(summary, judge, cot, pool, metric=metric)
    vmin, vmax, cbar_label = _matrix_color_range(metric)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    _annotate_matrix(ax, M)

    ax.set_yticks(range(len(FAMILY_NAMES)))
    ax.set_yticklabels(FAMILY_NAMES)
    ax.set_xticks(range(len(MATRIX_CAPS)))
    ax.set_xticklabels(MATRIX_CAPS)
    ax.set_xlabel("atomic capability")
    ax.set_ylabel("task family")

    for x in (2.5, 5.5):
        ax.axvline(x, color="black", linewidth=0.5, alpha=0.4)

    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.04, pad=0.02)
    title_metric = "Δlogprob magnitude" if metric == "delta" else f"probe bacc ({judge})"
    ax.set_title(
        f"Failure-diagnosis matrix — {_model_name(summary)} "
        f"({title_metric}, {cot}, {pool})\n"
        "cell = perturbation signal on its native task family",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_capacity_matrix_grid(summaries: list[dict], out: Path,
                             judge: str = "zscore", cot: str = "no_cot",
                             pool: str = "last", metric: str = "delta") -> None:
    """2x2 grid of capacity matrices, one per model."""
    n = len(summaries)
    if n == 0:
        return
    rows = 2 if n > 2 else 1
    cols = (n + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 3.0),
                             squeeze=False)
    vmin, vmax, cbar_label = _matrix_color_range(metric)
    last_im = None
    for idx, s in enumerate(summaries):
        ax = axes[idx // cols, idx % cols]
        M = _build_capacity_matrix(s, judge, cot, pool, metric=metric)
        last_im = ax.imshow(M, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        _annotate_matrix(ax, M)
        ax.set_yticks(range(len(FAMILY_NAMES))); ax.set_yticklabels(FAMILY_NAMES, fontsize=8)
        ax.set_xticks(range(len(MATRIX_CAPS))); ax.set_xticklabels(MATRIX_CAPS, fontsize=8)
        for x in (2.5, 5.5):
            ax.axvline(x, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(_model_name(s), fontsize=10)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     label=cbar_label, fraction=0.025, pad=0.02)
    title_metric = "Δlogprob magnitude" if metric == "delta" else f"probe bacc ({judge})"
    fig.suptitle(
        f"Failure-diagnosis matrix across models ({title_metric}, {cot}, {pool})",
        fontsize=11,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_capacity_scaling_lines(summaries: list[dict], out: Path,
                               judge: str = "zscore", cot: str = "no_cot",
                               pool: str = "last") -> None:
    """One axis, x=model size, y=balanced accuracy, one line per atomic
    capability. Base / instruct distinguished by marker."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cap_to_color = dict(zip(MATRIX_CAPS, plt.cm.tab10(np.linspace(0, 1, len(MATRIX_CAPS)))))
    plotted_any = False
    for cap in MATRIX_CAPS:
        for is_base, marker in [(False, "o"), (True, "s")]:
            xs, ys = [], []
            for s in summaries:
                name = _model_name(s)
                size = _model_size(name)
                if size is None: continue
                if ("base" in name.lower()) != is_base: continue
                v = _bacc_lookup(s, cap, judge=judge, cot=cot, pool=pool)
                if v is None: continue
                xs.append(size); ys.append(v)
            if xs:
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                style = "-" if not is_base else "--"
                ax.plot(xs, ys, marker=marker, color=cap_to_color[cap],
                        linestyle=style, markersize=6, linewidth=1.2,
                        label=f"{cap} ({'base' if is_base else 'inst'})")
                plotted_any = True
    if not plotted_any:
        plt.close(fig); return
    ax.set_xscale("log")
    ax.axhline(0.5, color="gray", ls="--", lw=0.6, label="chance")
    ax.set_xlabel("model size (B params, log scale)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"capability scaling ({judge}, {cot}, {pool})", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def cross_model_figures(summary_paths: list[Path], output_dir: Path) -> None:
    if len(summary_paths) < 2:
        return
    summaries = [json.load(open(p, encoding="utf-8")) for p in summary_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("plotting cross-model figures -> %s", output_dir)
    fig_scaling_taskfamily(summaries, output_dir / "fig_scaling_taskfamily.png")
    fig_scaling_capability(summaries, output_dir / "fig_scaling_capability.png")
    fig_capacity_matrix_grid(summaries, output_dir / "fig_capacity_matrix_grid.png")
    fig_capacity_scaling_lines(summaries, output_dir / "fig_capacity_scaling.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summaries", nargs="+", help="paths to summary.json files")
    p.add_argument("--output-dir", default="figures", help="root output directory")
    args = p.parse_args()

    paths = [Path(s) for s in args.summaries]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        per_model_figures(path, out)
    if len(paths) >= 2:
        cross_model_figures(paths, out / "_cross_model")


if __name__ == "__main__":
    main()
