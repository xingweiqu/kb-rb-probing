"""Representation geometry analysis: pairwise distances between variant hidden states."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import ItemFamily
from .hidden_states import HiddenStateStore


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity(a, b)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


_DISTANCE_FNS = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
}

# Default variant pairs to analyze
DEFAULT_VARIANT_PAIRS = [
    ("original", "paraphrase"),
    ("original", "premise"),
    ("original", "context_scaffolding"),
    ("original", "counterfactual"),
    ("original", "structure_substitution"),
    ("original", "symbol_substitution"),
    ("original", "hint"),
    ("original", "premise_removal"),
]


def pairwise_distances(
    families: dict[str, ItemFamily],
    store: HiddenStateStore,
    layer: int,
    variant_pairs: list[tuple[str, str]] | None = None,
    metric: str = "cosine",
) -> list[dict[str, Any]]:
    """Compute pairwise distances between variant hidden states for each family.

    Args:
        families: Dict of ItemFamily objects.
        store: HiddenStateStore with pre-saved hidden states.
        layer: Layer index to use.
        variant_pairs: List of (variant_a, variant_b) pairs. Defaults to DEFAULT_VARIANT_PAIRS.
        metric: "cosine" or "euclidean".

    Returns:
        List of dicts, one per (family, variant_pair) combination.
        Each dict has: family_id, task_family, sub_family, variant_a, variant_b, distance.
    """
    if variant_pairs is None:
        variant_pairs = DEFAULT_VARIANT_PAIRS

    dist_fn = _DISTANCE_FNS.get(metric)
    if dist_fn is None:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'cosine' or 'euclidean'.")

    rows: list[dict[str, Any]] = []
    for fid, family in families.items():
        for var_a, var_b in variant_pairs:
            if not family.has_variant(var_a) or not family.has_variant(var_b):
                continue
            try:
                h_a = store.get(family.row(var_a), layer)
                h_b = store.get(family.row(var_b), layer)
            except (FileNotFoundError, KeyError, ValueError):
                continue

            dist = dist_fn(h_a, h_b)
            rows.append({
                "family_id": fid,
                "task_family": family.task_family,
                "sub_family": family.sub_family,
                "variant_a": var_a,
                "variant_b": var_b,
                "distance": dist,
                "metric": metric,
                "layer": layer,
            })

    return rows


def pairwise_distances_all_layers(
    families: dict[str, ItemFamily],
    store: HiddenStateStore,
    layers: list[int],
    variant_pairs: list[tuple[str, str]] | None = None,
    metric: str = "cosine",
) -> list[dict[str, Any]]:
    """Run pairwise_distances across multiple layers."""
    all_rows: list[dict[str, Any]] = []
    for layer in layers:
        all_rows.extend(pairwise_distances(families, store, layer, variant_pairs, metric))
    return all_rows


def summarize_by_group(
    rows: list[dict[str, Any]],
    group_cols: list[str],
) -> list[dict[str, Any]]:
    """Group distance rows and compute mean ± std per group.

    Args:
        rows: Output of pairwise_distances.
        group_cols: Columns to group by, e.g. ["task_family", "variant_a", "variant_b"].

    Returns:
        List of dicts with group keys + distance_mean, distance_std, count.
    """
    from collections import defaultdict

    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(c, "") for c in group_cols)
        groups[key].append(row["distance"])

    summary = []
    for key, distances in sorted(groups.items()):
        entry = dict(zip(group_cols, key))
        entry["distance_mean"] = float(np.mean(distances))
        entry["distance_std"] = float(np.std(distances))
        entry["count"] = len(distances)
        summary.append(entry)

    return summary


def per_layer_distance_summary(
    families: dict[str, ItemFamily],
    store: HiddenStateStore,
    layers: list[int],
    variant_pairs: list[tuple[str, str]] | None = None,
    metric: str = "cosine",
    group_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convenience: compute and summarize distances across all layers.

    Returns a flat list suitable for CSV export.
    """
    if group_cols is None:
        group_cols = ["layer", "task_family", "variant_a", "variant_b"]

    all_rows = pairwise_distances_all_layers(families, store, layers, variant_pairs, metric)
    return summarize_by_group(all_rows, group_cols)
