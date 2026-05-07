"""Hidden-state geometry analysis for the capability taxonomy.

Outputs a single JSON ``probe_geometry.json`` per (cot_state, pool) covering:

  centroid_distances    layer -> 3x3 matrix of L2 distances between class centroids
                         (KB, RB, Hybrid)
  silhouette_per_layer   per-layer silhouette score using task_family labels
  cka_pairs              layer -> {KB-RB, KB-Hybrid, RB-Hybrid} CKA values
  cot_shift              for each task_family, mean L2 displacement of a row
                         between (no_cot, with_cot) at each layer; only emitted
                         when both cot states are present
  pca_2d                 2D PCA coords for the highest-bacc layer (last layer
                         used as fallback) for plotting

The script also writes a small markdown summary alongside the JSON so the
results are scannable without loading numpy.

Designed to be run after extract_hidden_states.py. No model required.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fam_of(family_id: str) -> str:
    if family_id.startswith("hybrid"):
        return "Hybrid"
    if family_id.startswith("kb"):
        return "KB"
    if family_id.startswith("rb"):
        return "RB"
    return "?"


def _load_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _select_originals(index: list[dict]) -> tuple[list[int], list[str]]:
    rows, fams = [], []
    for r in index:
        if r["variant"] != "original" or r.get("mode", "natural") != "natural":
            continue
        rows.append(r["row"])
        fams.append(_fam_of(r["family_id"]))
    return rows, fams


def _centered_kernel(X: np.ndarray) -> np.ndarray:
    K = X @ X.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n, d) matrices."""
    Kx = _centered_kernel(X)
    Ky = _centered_kernel(Y)
    num = (Kx * Ky).sum()
    denom = np.linalg.norm(Kx, ord="fro") * np.linalg.norm(Ky, ord="fro")
    if denom < 1e-12:
        return float("nan")
    return float(num / denom)


def analyze_one(
    hidden_dir: Path,
    cot_state: str,
    pool: str,
) -> dict:
    index = _load_index(hidden_dir / "item_index.jsonl")
    H = np.load(hidden_dir / f"hidden_{cot_state}_{pool}.npy").astype(np.float32, copy=False)

    rows, fams = _select_originals(index)
    label_names = ["KB", "RB", "Hybrid"]
    fam_to_id = {n: i for i, n in enumerate(label_names)}
    y = np.array([fam_to_id[f] for f in fams])
    Xall = H[rows]  # (N, L, D)
    n_layers = Xall.shape[1]

    centroid_distances = []
    silhouettes = []
    cka_pairs = []

    for layer in range(n_layers):
        Xl = Xall[:, layer, :]
        # centroids
        centroids = np.stack([Xl[y == i].mean(axis=0) for i in range(3)])
        d = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                d[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))
        centroid_distances.append(d.tolist())
        # silhouette
        try:
            sil = float(silhouette_score(Xl, y))
        except ValueError:
            sil = float("nan")
        silhouettes.append(sil)
        # CKA pairwise
        Xkb, Xrb, Xhy = Xl[y == 0], Xl[y == 1], Xl[y == 2]
        cka_pairs.append({
            "KB_RB": _cka(Xkb, Xrb),
            "KB_Hybrid": _cka(Xkb, Xhy),
            "RB_Hybrid": _cka(Xrb, Xhy),
        })

    # 2D PCA at the layer with max silhouette (geometry signal proxy)
    if any(not np.isnan(s) for s in silhouettes):
        best_layer = int(np.nanargmax(silhouettes))
    else:
        best_layer = n_layers - 1
    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(Xall[:, best_layer, :])
    pca_2d = {
        "layer": best_layer,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "points": [
            {"family": fams[i], "x": float(pca_xy[i, 0]), "y": float(pca_xy[i, 1])}
            for i in range(len(fams))
        ],
    }

    return {
        "cot_state": cot_state,
        "pool": pool,
        "n_samples": int(len(y)),
        "class_counts": np.bincount(y, minlength=3).tolist(),
        "label_names": label_names,
        "centroid_distances_per_layer": centroid_distances,
        "silhouette_per_layer": silhouettes,
        "cka_pairs_per_layer": cka_pairs,
        "best_silhouette_layer": best_layer,
        "pca_2d": pca_2d,
    }


def cot_shift(
    hidden_dir: Path,
    pool: str,
) -> dict | None:
    """Per-class mean L2 displacement of the same row between no_cot and with_cot."""
    p_no = hidden_dir / f"hidden_no_cot_{pool}.npy"
    p_with = hidden_dir / f"hidden_with_cot_{pool}.npy"
    if not (p_no.exists() and p_with.exists()):
        return None
    Hn = np.load(p_no).astype(np.float32, copy=False)
    Hw = np.load(p_with).astype(np.float32, copy=False)
    if Hn.shape != Hw.shape:
        logger.warning("cot shift skipped: shape mismatch %s vs %s", Hn.shape, Hw.shape)
        return None

    index = _load_index(hidden_dir / "item_index.jsonl")
    rows, fams = _select_originals(index)
    label_names = ["KB", "RB", "Hybrid"]
    fam_to_id = {n: i for i, n in enumerate(label_names)}
    y = np.array([fam_to_id[f] for f in fams])
    diffs = Hw[rows] - Hn[rows]  # (N, L, D)
    norms = np.linalg.norm(diffs, axis=-1)  # (N, L)
    n_layers = norms.shape[1]
    by_class = {}
    for fam, idx in zip(label_names, range(3)):
        m = (y == idx)
        by_class[fam] = norms[m].mean(axis=0).tolist() if m.any() else [float("nan")] * n_layers

    return {"pool": pool, "label_names": label_names, "shift_norm_per_layer": by_class}


def _summary_md(results: dict) -> str:
    lines = ["# Geometry summary", ""]
    lines.append("## Best silhouette per (cot_state, pool)")
    lines.append("")
    lines.append("| cot_state | pool | best layer | best silhouette |")
    lines.append("|---|---|---|---|")
    for r in results["per_setting"]:
        sils = r["silhouette_per_layer"]
        best = r["best_silhouette_layer"]
        lines.append(f"| {r['cot_state']} | {r['pool']} | {best} | {sils[best]:.3f} |")
    if results.get("cot_shift"):
        lines.append("")
        lines.append("## CoT shift L2 norm (mean over items, per class, last layer)")
        lines.append("")
        lines.append("| pool | KB | RB | Hybrid |")
        lines.append("|---|---|---|---|")
        for s in results["cot_shift"]:
            kb, rb, hy = (s["shift_norm_per_layer"][k][-1] for k in ("KB", "RB", "Hybrid"))
            lines.append(f"| {s['pool']} | {kb:.3f} | {rb:.3f} | {hy:.3f} |")
    return "\n".join(lines) + "\n"


def run(
    hidden_dir: str,
    output_path: str,
    cot_states: Iterable[str] = ("no_cot", "with_cot"),
    pools: Iterable[str] = ("last", "mean"),
) -> dict:
    hidden_dir_p = Path(hidden_dir)
    out: dict = {"per_setting": [], "cot_shift": []}
    for cot in cot_states:
        for pool in pools:
            if not (hidden_dir_p / f"hidden_{cot}_{pool}.npy").exists():
                logger.warning("missing hidden_%s_%s.npy — skipping", cot, pool)
                continue
            out["per_setting"].append(analyze_one(hidden_dir_p, cot, pool))
    for pool in pools:
        shift = cot_shift(hidden_dir_p, pool)
        if shift is not None:
            out["cot_shift"].append(shift)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    md_path = Path(output_path).with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_summary_md(out))
    logger.info("wrote %s and %s", output_path, md_path)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cot_states", nargs="+", default=["no_cot", "with_cot"])
    p.add_argument("--pools", nargs="+", default=["last", "mean"])
    args = p.parse_args()
    run(args.hidden_dir, args.output, args.cot_states, args.pools)


if __name__ == "__main__":
    main()
