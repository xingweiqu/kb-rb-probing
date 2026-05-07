"""Linear probes for the 2x3 capability taxonomy and the task-family axis.

Two probe targets are supported:

1. ``task_family`` — three-way classification over (KB, RB, Hybrid).
   Trained on `original`-variant items only (one row per family). The
   confusion matrix on this probe is the central artifact for the
   hidden-state geometry argument.

2. ``capability`` — binary classification per atomic capability label
   (KP, KD, KB, RP, RD, RB, CP, CB, CB-control). One probe per (capability,
   cot_state) pair, trained on the families where ``derive_labels.py``
   emitted a non-null judgment.

For both targets we sweep all transformer layers and report per-layer
accuracy + balanced accuracy + (for task_family) the confusion matrix.

Inputs come from ``extract_hidden_states.py``:
    hidden_<cot_state>_<pool>.npy        (N, n_layers, hidden_dim)
    item_index.jsonl                     row -> (family_id, variant, mode)

Plus the labels file from ``derive_labels.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _load_hidden(npy_path: Path) -> np.ndarray:
    arr = np.load(npy_path)
    return arr.astype(np.float32, copy=False)


def _kfold_layer_metrics(
    X_layer: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    rng: int,
) -> tuple[float, float, np.ndarray]:
    """Stratified k-fold logistic regression on one layer."""
    n_classes = len(np.unique(y))
    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        return float("nan"), float("nan"), np.zeros((n_classes, n_classes), dtype=int)
    cm_total = np.zeros((n_classes, n_classes), dtype=int)
    accs, baccs = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    for train_idx, test_idx in skf.split(X_layer, y):
        clf = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=rng,
        )
        clf.fit(X_layer[train_idx], y[train_idx])
        pred = clf.predict(X_layer[test_idx])
        accs.append((pred == y[test_idx]).mean())
        baccs.append(balanced_accuracy_score(y[test_idx], pred))
        cm_total += confusion_matrix(y[test_idx], pred, labels=np.arange(n_classes))
    return float(np.mean(accs)), float(np.mean(baccs)), cm_total


def probe_task_family(
    hidden_dir: Path,
    cot_state: str,
    pool: str,
    n_splits: int = 5,
    rng: int = 42,
) -> dict:
    """Three-way KB/RB/Hybrid probe on `original` variants only."""
    index = _load_index(hidden_dir / "item_index.jsonl")
    H = _load_hidden(hidden_dir / f"hidden_{cot_state}_{pool}.npy")  # (N, L, D)

    families_per_id = {}
    rows = []
    family_ids = []
    for r in index:
        if r["variant"] != "original" or r.get("mode", "natural") != "natural":
            continue
        # task_family is not in index; derive it from family_id prefix
        fid = r["family_id"]
        fam = fid.split("_")[0].upper()
        if fam == "HYBRID":
            fam = "Hybrid"
        elif fam == "KB":
            fam = "KB"
        elif fam == "RB":
            fam = "RB"
        else:
            continue
        families_per_id[fid] = fam
        rows.append(r["row"])
        family_ids.append(fid)

    label_names = ["KB", "RB", "Hybrid"]
    label_to_id = {name: i for i, name in enumerate(label_names)}
    y = np.array([label_to_id[families_per_id[fid]] for fid in family_ids])
    Xall = H[rows]  # (N_orig, L, D)

    n_layers = Xall.shape[1]
    layer_results = []
    for layer in range(n_layers):
        acc, bacc, cm = _kfold_layer_metrics(Xall[:, layer, :], y, n_splits, rng)
        layer_results.append({
            "layer": layer,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "confusion_matrix": cm.tolist(),
        })
    best = max(layer_results, key=lambda d: d["balanced_accuracy"] if not np.isnan(d["balanced_accuracy"]) else -1)
    logger.info(
        "task_family probe (%s/%s): best layer=%d bacc=%.3f acc=%.3f",
        cot_state, pool, best["layer"], best["balanced_accuracy"], best["accuracy"],
    )
    return {
        "target": "task_family",
        "cot_state": cot_state,
        "pool": pool,
        "label_names": label_names,
        "n_samples": int(len(y)),
        "class_counts": np.bincount(y, minlength=len(label_names)).tolist(),
        "by_layer": layer_results,
        "best_layer": best["layer"],
    }


def probe_capabilities(
    hidden_dir: Path,
    labels_path: Path,
    cot_state: str,
    pool: str,
    n_splits: int = 5,
    rng: int = 42,
    judge: str = "binary",
) -> list[dict]:
    """One binary probe per atomic capability (using `original` row's hidden state).

    `judge` selects which label column from derive_labels.py to read:
        "binary" : correctness-flip label (legacy, drops on strong models)
        "delta"  : Δlogprob-thresholded label (preferred when binary collapses)
    """
    index = _load_index(hidden_dir / "item_index.jsonl")
    H = _load_hidden(hidden_dir / f"hidden_{cot_state}_{pool}.npy")
    labels = json.load(open(labels_path, encoding="utf-8"))

    fid_to_orig_row = {}
    for r in index:
        if r["variant"] == "original" and r.get("mode", "natural") == "natural":
            fid_to_orig_row[r["family_id"]] = r["row"]

    cap_to_xy: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for fid, by_cot in labels.items():
        cell_root = by_cot.get(cot_state, {})
        # support both new schema {"binary": {...}, "delta": {...}} and old flat dict
        if isinstance(cell_root, dict) and judge in cell_root:
            cell = cell_root[judge]
        else:
            cell = cell_root  # legacy flat dict
        if fid not in fid_to_orig_row:
            continue
        for cap, val in cell.items():
            if val is None:
                continue
            cap_to_xy[cap].append((fid_to_orig_row[fid], bool(val)))

    results = []
    n_layers = H.shape[1]
    for cap in sorted(cap_to_xy):
        rows = [r for r, _ in cap_to_xy[cap]]
        y = np.array([1 if v else 0 for _, v in cap_to_xy[cap]])
        if len(np.unique(y)) < 2 or len(y) < 6:
            results.append({
                "target": "capability",
                "capability": cap,
                "cot_state": cot_state,
                "pool": pool,
                "skipped": True,
                "reason": f"insufficient samples (n={len(y)}, classes={len(np.unique(y))})",
                "n_samples": int(len(y)),
                "class_counts": np.bincount(y, minlength=2).tolist(),
            })
            continue
        Xcap = H[rows]
        layer_results = []
        for layer in range(n_layers):
            acc, bacc, cm = _kfold_layer_metrics(Xcap[:, layer, :], y, n_splits, rng)
            layer_results.append({
                "layer": layer,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "confusion_matrix": cm.tolist(),
            })
        best = max(layer_results, key=lambda d: d["balanced_accuracy"] if not np.isnan(d["balanced_accuracy"]) else -1)
        logger.info(
            "capability probe %s (%s/%s): n=%d pos=%d neg=%d best layer=%d bacc=%.3f",
            cap, cot_state, pool, len(y), int(y.sum()), int((1 - y).sum()),
            best["layer"], best["balanced_accuracy"],
        )
        results.append({
            "target": "capability",
            "capability": cap,
            "judge": judge,
            "cot_state": cot_state,
            "pool": pool,
            "n_samples": int(len(y)),
            "class_counts": np.bincount(y, minlength=2).tolist(),
            "by_layer": layer_results,
            "best_layer": best["layer"],
        })
    return results


def _skipped_capability_record(cap, cot_state, pool, judge, y_count):
    return {
        "target": "capability", "capability": cap, "judge": judge,
        "cot_state": cot_state, "pool": pool, "skipped": True,
        "reason": "insufficient samples", "n_samples": int(y_count),
    }


def run(
    hidden_dir: str,
    labels_path: str,
    output_path: str,
    cot_states: Iterable[str] = ("no_cot", "with_cot"),
    pools: Iterable[str] = ("last", "mean"),
    n_splits: int = 5,
    rng: int = 42,
    judges: Iterable[str] = ("binary", "delta", "zscore"),
) -> dict:
    hidden_dir_p = Path(hidden_dir)
    out: dict = {"task_family": [], "capability": []}
    for cot in cot_states:
        for pool in pools:
            npy = hidden_dir_p / f"hidden_{cot}_{pool}.npy"
            if not npy.exists():
                logger.warning("missing %s — skipping", npy)
                continue
            out["task_family"].append(probe_task_family(hidden_dir_p, cot, pool, n_splits, rng))
            for judge in judges:
                out["capability"].extend(
                    probe_capabilities(
                        hidden_dir_p, Path(labels_path), cot, pool, n_splits, rng, judge=judge,
                    )
                )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", output_path)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dir", required=True, help="dir containing hidden_*.npy and item_index.jsonl")
    p.add_argument("--labels", required=True, help="capability labels JSON from derive_labels.py")
    p.add_argument("--output", required=True, help="output probe results JSON")
    p.add_argument("--cot_states", nargs="+", default=["no_cot", "with_cot"])
    p.add_argument("--pools", nargs="+", default=["last", "mean"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--rng", type=int, default=42)
    p.add_argument("--judges", nargs="+", default=["binary", "delta", "zscore"],
                   choices=["binary", "delta", "zscore"])
    args = p.parse_args()

    run(
        hidden_dir=args.hidden_dir,
        labels_path=args.labels,
        output_path=args.output,
        cot_states=args.cot_states,
        pools=args.pools,
        n_splits=args.n_splits,
        rng=args.rng,
        judges=args.judges,
    )


if __name__ == "__main__":
    main()
