"""Low-rank (LoRA-style) probes as a non-linear robustness check.

The probe applied at each layer is:

    h -> A @ h        # A: rank x hidden_dim
      -> ReLU
      -> B @ ReLU(...) # B: n_classes x rank

with class-balanced cross-entropy and weight decay for regularization. Same
"rank-r adapter" idea as Allen-Zhu & Li, Physics of LMs.

This implementation **batches all layers** into one forward pass. For each
(capability, fold) we train one ProbeBank module of shape (n_layers, ...)
in parallel — typically 36-37 layers fit in a single small GPU kernel. This
turned the bottleneck from per-layer launch overhead into actual compute,
giving a ~30x speedup at 4096-dim, rank=16 on a single A100.

We mirror linear_probe.py's interface: same input directory layout, same
target axes (`task_family` and `capability`), same output JSON schema with
extra ``probe_type`` and ``judge`` fields.
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


class ProbeBank(nn.Module):
    """A bank of independent rank-r probes, one per layer, trained in parallel.

    Parameter shapes:
        A : (n_layers, hidden_dim, rank)
        B : (n_layers, rank, n_classes)
        b : (n_layers, n_classes)

    Input  : (n_layers, batch, hidden_dim)
    Output : (n_layers, batch, n_classes)
    """

    def __init__(self, n_layers: int, hidden_dim: int, n_classes: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_layers, hidden_dim, rank))
        self.B = nn.Parameter(torch.zeros(n_layers, rank, n_classes))
        self.b = nn.Parameter(torch.zeros(n_layers, n_classes))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, B, H)  -> (L, B, R)  via per-layer matmul
        h = torch.einsum("lbh,lhr->lbr", x, self.A)
        h = F.relu(h)
        out = torch.einsum("lbr,lrc->lbc", h, self.B) + self.b.unsqueeze(1)
        return out


def _train_bank(
    X_train: torch.Tensor,    # (L, B_train, H)
    y_train: torch.Tensor,    # (B_train,)
    X_test: torch.Tensor,     # (L, B_test, H)
    y_test: torch.Tensor,     # (B_test,)
    n_classes: int,
    rank: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    rng: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train ProbeBank covering all L layers, return per-layer (acc, bacc, cm)."""
    device = X_train.device
    n_layers = X_train.shape[0]

    torch.manual_seed(rng)
    bank = ProbeBank(n_layers, X_train.shape[2], n_classes, rank).to(device)
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=weight_decay)

    counts = torch.bincount(y_train, minlength=n_classes).float()
    weight = counts.sum() / counts.clamp(min=1.0)
    weight = (weight / weight.mean()).to(device)

    bank.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = bank(X_train)                        # (L, B, C)
        # cross-entropy across (L*B) rows; targets repeat across L
        loss = F.cross_entropy(
            logits.reshape(-1, n_classes),
            y_train.repeat(n_layers),
            weight=weight,
        )
        loss.backward()
        opt.step()

    bank.eval()
    with torch.no_grad():
        pred = bank(X_test).argmax(dim=-1).cpu().numpy()  # (L, B_test)
    y_np = y_test.cpu().numpy()
    accs, baccs, cms = [], [], []
    for li in range(n_layers):
        p = pred[li]
        accs.append(float((p == y_np).mean()))
        baccs.append(float(balanced_accuracy_score(y_np, p)))
        cms.append(confusion_matrix(y_np, p, labels=np.arange(n_classes)))
    return np.array(accs), np.array(baccs), np.stack(cms)


def _kfold_bank(
    X: np.ndarray,            # (N, L, H)
    y: np.ndarray,            # (N,)
    rank: int,
    n_splits: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    rng: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-fold across samples, return per-layer mean (acc, bacc, summed cm)."""
    n_classes = len(np.unique(y))
    n_splits = min(n_splits, int(np.bincount(y).min()))
    n_layers = X.shape[1]
    if n_splits < 2:
        nan = np.full(n_layers, np.nan)
        return nan, nan, np.zeros((n_layers, n_classes, n_classes), dtype=int)

    # move full tensor once
    X_t = torch.tensor(X, dtype=torch.float32, device=device).permute(1, 0, 2)  # (L, N, H)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    accs_all, baccs_all = [], []
    cm_total = np.zeros((n_layers, n_classes, n_classes), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    for train_idx, test_idx in skf.split(X[:, 0, :], y):
        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
        test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)
        accs, baccs, cms = _train_bank(
            X_t.index_select(1, train_idx_t),
            y_t.index_select(0, train_idx_t),
            X_t.index_select(1, test_idx_t),
            y_t.index_select(0, test_idx_t),
            n_classes, rank, epochs, lr, weight_decay, rng,
        )
        accs_all.append(accs); baccs_all.append(baccs)
        cm_total += cms
    return np.mean(accs_all, axis=0), np.mean(baccs_all, axis=0), cm_total


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
    Xall = H[rows]  # (N, L, H)

    accs, baccs, cms = _kfold_bank(
        Xall, y, rank, n_splits, epochs, lr, weight_decay, device, rng,
    )
    n_layers = Xall.shape[1]
    layer_results = [
        {
            "layer": li,
            "accuracy": float(accs[li]) if not np.isnan(accs[li]) else float("nan"),
            "balanced_accuracy": float(baccs[li]) if not np.isnan(baccs[li]) else float("nan"),
            "confusion_matrix": cms[li].tolist(),
        }
        for li in range(n_layers)
    ]
    valid = [(li, baccs[li]) for li in range(n_layers) if not np.isnan(baccs[li])]
    best_layer = max(valid, key=lambda t: t[1])[0] if valid else 0
    logger.info(
        "[lora-r%d] task_family (%s/%s): best layer=%d bacc=%.3f",
        rank, cot_state, pool, best_layer, layer_results[best_layer]["balanced_accuracy"],
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
        "best_layer": best_layer,
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
    judge: str = "binary",
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
        cell_root = by_cot.get(cot_state, {})
        if isinstance(cell_root, dict) and judge in cell_root:
            cell = cell_root[judge]
        else:
            cell = cell_root
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
                "probe_type": "lora", "rank": rank, "judge": judge,
                "target": "capability", "capability": cap,
                "cot_state": cot_state, "pool": pool,
                "skipped": True,
                "reason": f"insufficient samples (n={len(y)})",
            })
            continue
        Xcap = H[rows]
        accs, baccs, cms = _kfold_bank(
            Xcap, y, rank, n_splits, epochs, lr, weight_decay, device, rng,
        )
        n_layers = Xcap.shape[1]
        layer_results = [
            {
                "layer": li,
                "accuracy": float(accs[li]) if not np.isnan(accs[li]) else float("nan"),
                "balanced_accuracy": float(baccs[li]) if not np.isnan(baccs[li]) else float("nan"),
                "confusion_matrix": cms[li].tolist(),
            }
            for li in range(n_layers)
        ]
        valid = [(li, baccs[li]) for li in range(n_layers) if not np.isnan(baccs[li])]
        best_layer = max(valid, key=lambda t: t[1])[0] if valid else 0
        out.append({
            "probe_type": "lora", "rank": rank, "judge": judge,
            "target": "capability", "capability": cap,
            "cot_state": cot_state, "pool": pool,
            "n_samples": int(len(y)),
            "class_counts": np.bincount(y, minlength=2).tolist(),
            "by_layer": layer_results,
            "best_layer": best_layer,
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
            out["task_family"].append(probe_task_family(
                hidden_dir_p, cot, pool, rank, epochs, lr, weight_decay, n_splits, device, rng,
            ))
            for judge in judges:
                out["capability"].extend(probe_capabilities(
                    hidden_dir_p, Path(labels_path), cot, pool, rank, epochs, lr, weight_decay,
                    n_splits, device, rng, judge=judge,
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
    p.add_argument("--judges", nargs="+", default=["binary", "delta", "zscore"],
                   choices=["binary", "delta", "zscore"])
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
        judges=args.judges,
    )


if __name__ == "__main__":
    main()
