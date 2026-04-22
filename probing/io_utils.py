"""I/O utilities: JSONL, tensor files, output directory management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np


# ── JSONL ──────────────────────────────────────────────────────────────────────

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lazily iterate over a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc


# ── Tensor / array files ───────────────────────────────────────────────────────

def load_tensor(path: str | Path, key: str | None = None) -> np.ndarray:
    """Load a hidden-state tensor from a .pt or .npy file.

    Args:
        path: Path to the file.
        key: If the file stores a dict (e.g. torch .pt with multiple tensors),
             use this key to select the right tensor.

    Returns:
        numpy array of shape [L, H] or [H].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return arr.astype(np.float32)

    if suffix in {".pt", ".pth"}:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError("PyTorch is required to load .pt files: pip install torch") from exc
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            if key is None:
                if len(obj) == 1:
                    obj = next(iter(obj.values()))
                else:
                    raise ValueError(
                        f"File {path} contains multiple keys {list(obj.keys())}; "
                        "specify hidden_state_key in metadata."
                    )
            else:
                if key not in obj:
                    raise KeyError(f"Key '{key}' not found in {path}. Available: {list(obj.keys())}")
                obj = obj[key]
        if hasattr(obj, "numpy"):
            return obj.float().numpy()
        return np.array(obj, dtype=np.float32)

    raise ValueError(f"Unsupported tensor file format: {suffix}. Expected .pt, .pth, or .npy")


# ── Output directory management ────────────────────────────────────────────────

def make_output_dir(
    base: str | Path,
    experiment: str,
    probe_target: str,
    split_mode: str,
    seed: int | None = None,
) -> Path:
    """Create and return a structured output directory.

    Structure: {base}/{experiment}/{probe_target}/{split_mode}/[seed_{seed}/]
    """
    parts = [str(base), experiment, probe_target, split_mode]
    if seed is not None:
        parts.append(f"seed_{seed}")
    out = Path(os.path.join(*parts))
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of dicts as CSV using the csv module (no pandas dependency)."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
