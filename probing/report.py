"""Output saving utilities for probing experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import make_output_dir, save_json, save_csv


def save_run(
    base_output_dir: str,
    experiment: str,
    probe_target: str,
    split_mode: str,
    seed: int,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    coefficients: list | None,
    config_dict: dict[str, Any],
) -> Path:
    """Save all outputs for one probe run (one seed).

    Directory: {base}/{experiment}/{probe_target}/{split_mode}/seed_{seed}/

    Files:
        metrics.json
        predictions.csv
        coefficients.csv  (if provided)
        run_config.json
    """
    out_dir = make_output_dir(base_output_dir, experiment, probe_target, split_mode, seed)

    save_json(metrics, out_dir / "metrics.json")
    save_csv(predictions, out_dir / "predictions.csv")
    save_json(config_dict, out_dir / "run_config.json")

    if coefficients is not None:
        coef_arr = np.array(coefficients)
        if coef_arr.ndim == 1:
            coef_rows = [{"dim": i, "weight": float(w)} for i, w in enumerate(coef_arr)]
        else:
            coef_rows = [
                {"class": c, "dim": d, "weight": float(coef_arr[c, d])}
                for c in range(coef_arr.shape[0])
                for d in range(coef_arr.shape[1])
            ]
        save_csv(coef_rows, out_dir / "coefficients.csv")

    return out_dir


def save_per_layer_metrics(
    base_output_dir: str,
    experiment: str,
    probe_target: str,
    split_mode: str,
    layer_rows: list[dict[str, Any]],
) -> Path:
    """Save aggregated per-layer metrics CSV.

    Directory: {base}/{experiment}/{probe_target}/{split_mode}/
    File: per_layer_metrics.csv
    """
    out_dir = make_output_dir(base_output_dir, experiment, probe_target, split_mode)
    save_csv(layer_rows, out_dir / "per_layer_metrics.csv")
    return out_dir


def save_confusion_matrix(
    base_output_dir: str,
    experiment: str,
    probe_target: str,
    split_mode: str,
    cm: list[list[int]],
    class_names: list[str] | None = None,
) -> Path:
    """Save confusion matrix as CSV."""
    out_dir = make_output_dir(base_output_dir, experiment, probe_target, split_mode)
    n = len(cm)
    names = class_names or [str(i) for i in range(n)]
    rows = []
    for i, row in enumerate(cm):
        entry = {"true_class": names[i]}
        for j, val in enumerate(row):
            entry[f"pred_{names[j]}"] = val
        rows.append(entry)
    save_csv(rows, out_dir / "confusion_matrix.csv")
    return out_dir


def save_geometry_results(
    base_output_dir: str,
    rows: list[dict],
    summary: list[dict],
    layer: int | str = "all",
) -> Path:
    """Save geometry analysis results."""
    out_dir = Path(base_output_dir) / "geometry" / f"layer_{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, out_dir / "pairwise_distances.csv")
    save_csv(summary, out_dir / "distance_summary.csv")
    return out_dir


def save_transfer_comparison(
    base_output_dir: str,
    setup: str,
    rows: list[dict[str, Any]],
) -> Path:
    """Save transfer probe comparison table."""
    out_dir = Path(base_output_dir) / "transfer" / setup
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, out_dir / "transfer_comparison.csv")
    save_json(rows, out_dir / "transfer_comparison.json")
    return out_dir
