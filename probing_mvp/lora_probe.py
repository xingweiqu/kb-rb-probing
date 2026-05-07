"""Low-rank (LoRA-style) probes as a non-linear robustness check.

The probe applied at each layer is:

    h -> A @ h        # A: rank x hidden_dim
      -> ReLU
      -> B @ ReLU(...) # B: n_classes x rank

with class-balanced cross-entropy and weight decay for regularization. This
is the same "rank-r adapter" idea Allen-Zhu & Li used in the Physics of LMs
papers — meaningful expressivity beyond linear, but few enough parameters
that a high accuracy still implies the information is recoverable rather
than memorized by an over-parameterized head.

We deliberately mirror linear_probe.py's interface: same input directory
layout, same target axes (`task_family` and `capability`), same output JSON
schema with an extra ``probe_type`` field. Run it after linear_probe to
compare like-for-like.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LoRAProbe(nn.Module):
    def __init__(self, hidden_dim: int, n_classes: int, rank: int):
        super().__init__()
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, n_classes, bias=True)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(F.relu(self.A(x)))


def _train_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    rank: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    rng: int,
) -> tuple[float, float, np.ndarray]:
    torch.manual_seed(rng)
    n_classes = int(max(ytr.max(), yte.max())) + 1
    model = LoRAProbe(Xtr.shape[1], n_classes, rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    weight = weight / weight.mean()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = F.cross_entropy(logits, ytr_t, weight=weight)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(dim=-1).cpu().numpy()
    yte_np = yte_t.cpu().numpy()
    acc = float((pred == yte_np).mean())
    bacc = float(balanced_accuracy_score(yte_np, pred))
    cm = confusion_matrix(yte_np, pred, labels=np.arange(n_classes))
    return acc, bacc, cm


def _kfold_layer_metrics(
    X_layer: np.ndarray,
    y: np.ndarray,
    rank: int,
    n_splits: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    rng: int,
) -> tuple[float, float, np.ndarray]:
    n_classes = len(np.unique(y))
    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        return float("nan"), float("nan"), np.zeros((n_classes, n_classes), dtype=int)
    cm_total = np.zeros((n_classes, n_classes), dtype=int)
    accs, baccs = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    for train_idx, test_idx in skf.split(X_layer, y):
        a, b, cm = _train_eval(
            X_layer[train_idx], y[train_idx],
            X_layer[test_idx], y[test_idx],
            rank=rank, epochs=epochs, lr=lr, weight_decay=weight_decay,
            device=device, rng=rng,
        )
        accs.append(a); baccs.append(b); cm_total += cm
    return float(np.mean(accs)), float(np.mean(baccs)), cm_total


def _load_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def probe_task_family(
    hidden_dir: Path,
    cot_state: str,
    pool: str,
    rank: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_splits: int,
    device: str,
    rng: int,
) -> dict:
    index = _load_index(hidden_dir / "item_index.jsonl")
    H = np.load(hidden_dir / f"hidden_{cot_state}_{pool}.npy").astype(np.float32, copy=False)

    rows, fids = [], []
    for r in index:
        if r["variant"] != "original" or r.get("mode", "natural") != "natural":
            continue
        rows.append(r["row"]); fids.append(r["family_id"])
    label_names = ["KB", "RB", "Hybrid"]
    label_to_id = {"KB": 0, "RB": 1, "Hybrid": 2}
    fam_of = lambda fid: ("Hybrid" if fid.startswith("hybrid") else fid.split("_")[0].upper())
    y = np.array([label_to_id[fam_of(fid)] for fid in fids])
    Xall = H[rows]

    layer_results = []
    for layer in range(Xall.shape[1]):
        acc, bacc, cm = _kfold_layer_metrics(
            Xall[:, layer, :], y, rank, n_splits, epochs, lr, weight_decay, device, rng,
        )
        layer_results.append({
            "layer": layer, "accuracy": acc, "balanced_accuracy": bacc,
            "confusion_matrix": cm.tolist(),
        })
    best = max(layer_results, key=lambda d: d["balanced_accuracy"] if not np.isnan(d["balanced_accuracy"]) else -1)
    logger.info(
        "[lora-r%d] task_family (%s/%s): best layer=%d bacc=%.3f",
        rank, cot_state, pool, best["layer"], best["balanced_accuracy"],
    )
    return {
        "probe_type": "lora",
        "rank": rank,
        "target": "task_family",
        "cot_state": cot_state,
        "pool": pool,
        "label_names": label_names,
        "n_samples": int(len(y)),
        "class_counts": np.bincount(y, minlength=3).tolist(),
        "by_layer": layer_results,
        "best_layer": best["layer"],
    }


def probe_capabilities(
    hidden_dir: Path,
    labels_path: Path,
    cot_state: str,
    pool: str,
    rank: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_splits: int,
    device: str,
    rng: int,
) -> list[dict]:
    index = _load_index(hidden_dir / "item_index.jsonl")
    H = np.load(hidden_dir / f"hidden_{cot_state}_{pool}.npy").astype(np.float32, copy=False)
    labels = json.load(open(labels_path, encoding="utf-8"))

    fid_to_orig_row = {
        r["family_id"]: r["row"]
        for r in index
        if r["variant"] == "original" and r.get("mode", "natural") == "natural"
    }
    cap_to_xy: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for fid, by_cot in labels.items():
        cell = by_cot.get(cot_state, {})
        if fid not in fid_to_orig_row:
            continue
        for cap, val in cell.items():
            if val is None:
                continue
            cap_to_xy[cap].append((fid_to_orig_row[fid], bool(val)))

    out = []
    for cap in sorted(cap_to_xy):
        rows = [r for r, _ in cap_to_xy[cap]]
        y = np.array([1 if v else 0 for _, v in cap_to_xy[cap]])
        if len(np.unique(y)) < 2 or len(y) < 6:
            out.append({
                "probe_type": "lora", "rank": rank,
                "target": "capability", "capability": cap,
                "cot_state": cot_state, "pool": pool,
                "skipped": True,
                "reason": f"insufficient samples (n={len(y)})",
            })
            continue
        Xcap = H[rows]
        layer_results = []
        for layer in range(Xcap.shape[1]):
            acc, bacc, cm = _kfold_layer_metrics(
                Xcap[:, layer, :], y, rank, n_splits, epochs, lr, weight_decay, device, rng,
            )
            layer_results.append({
                "layer": layer, "accuracy": acc, "balanced_accuracy": bacc,
                "confusion_matrix": cm.tolist(),
            })
        best = max(layer_results, key=lambda d: d["balanced_accuracy"] if not np.isnan(d["balanced_accuracy"]) else -1)
        out.append({
            "probe_type": "lora", "rank": rank,
            "target": "capability", "capability": cap,
            "cot_state": cot_state, "pool": pool,
            "n_samples": int(len(y)),
            "class_counts": np.bincount(y, minlength=2).tolist(),
            "by_layer": layer_results,
            "best_layer": best["layer"],
        })
    return out


def run(
    hidden_dir: str,
    labels_path: str,
    output_path: str,
    cot_states: Iterable[str] = ("no_cot", "with_cot"),
    pools: Iterable[str] = ("last", "mean"),
    rank: int = 16,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    n_splits: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    rng: int = 42,
) -> dict:
    hidden_dir_p = Path(hidden_dir)
    out: dict = {"task_family": [], "capability": []}
    for cot in cot_states:
        for pool in pools:
            npy = hidden_dir_p / f"hidden_{cot}_{pool}.npy"
            if not npy.exists():
                logger.warning("missing %s — skipping", npy)
                continue
            out["task_family"].append(probe_task_family(
                hidden_dir_p, cot, pool, rank, epochs, lr, weight_decay, n_splits, device, rng,
            ))
            out["capability"].extend(probe_capabilities(
                hidden_dir_p, Path(labels_path), cot, pool, rank, epochs, lr, weight_decay,
                n_splits, device, rng,
            ))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", output_path)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cot_states", nargs="+", default=["no_cot", "with_cot"])
    p.add_argument("--pools", nargs="+", default=["last", "mean"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rng", type=int, default=42)
    args = p.parse_args()
    run(
        hidden_dir=args.hidden_dir,
        labels_path=args.labels,
        output_path=args.output,
        cot_states=args.cot_states,
        pools=args.pools,
        rank=args.rank,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_splits=args.n_splits,
        device=args.device,
        rng=args.rng,
    )


if __name__ == "__main__":
    main()
