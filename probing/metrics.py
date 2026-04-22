"""Evaluation metrics for probing experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities [N, C]. Used for AUROC.

    Returns:
        Dict with accuracy, macro_f1, auroc (if binary + proba available),
        confusion_matrix, and per_class_f1.
    """
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        f1_score,
        confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()

    result: dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class,
        "confusion_matrix": cm,
        "n_samples": int(len(y_true)),
    }

    # AUROC: only for binary classification with probabilities
    classes = np.unique(y_true)
    if y_proba is not None and len(classes) == 2:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            # Use probability of the positive class (index 1)
            auroc = float(roc_auc_score(y_true, y_proba[:, 1]))
            result["auroc"] = auroc
        except Exception:
            result["auroc"] = None
    else:
        result["auroc"] = None

    return result


def aggregate_seeds(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed metric dicts into mean ± std.

    Args:
        results: List of dicts, each with a "metrics" key.

    Returns:
        Dict with mean and std for each scalar metric.
    """
    if not results:
        return {}

    scalar_keys = [
        k for k, v in results[0]["metrics"].items()
        if isinstance(v, (int, float)) and v is not None
    ]

    aggregated: dict[str, Any] = {}
    for key in scalar_keys:
        values = [r["metrics"][key] for r in results if r["metrics"].get(key) is not None]
        if not values:
            continue
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    # Include the last seed's confusion matrix as a representative
    aggregated["confusion_matrix"] = results[-1]["metrics"].get("confusion_matrix")
    aggregated["n_seeds"] = len(results)
    return aggregated


def per_layer_metrics(
    layer_results: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate seed results per layer into a flat list of dicts for CSV export.

    Args:
        layer_results: Dict mapping layer_index → list of per-seed result dicts.

    Returns:
        List of dicts, one per layer, with mean/std metrics.
    """
    rows = []
    for layer_idx in sorted(layer_results.keys()):
        seed_results = layer_results[layer_idx]
        agg = aggregate_seeds(seed_results)
        row = {"layer": layer_idx}
        row.update(agg)
        rows.append(row)
    return rows
