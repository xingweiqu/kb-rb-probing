"""Feature matrix assembly: hidden states → (X, y) arrays for probing."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore

from .config import ProbingConfig
from .data import ItemFamily, SCAFFOLD_VARIANTS
from .hidden_states import HiddenStateStore, ModelExtractor


def _get_vector(
    store: HiddenStateStore | ModelExtractor,
    family: ItemFamily,
    variant: str,
    layer: int | None,
    delta_mode: bool,
) -> np.ndarray | None:
    """Return the feature vector for one (family, variant) pair.

    Returns None if the variant is missing or the hidden state is unavailable.
    """
    if not family.has_variant(variant):
        return None

    try:
        if isinstance(store, ModelExtractor):
            # Extract on-the-fly: not yet supported in batch here; caller handles it
            raise NotImplementedError(
                "ModelExtractor batch extraction should be called via extract_all_families()."
            )
        if delta_mode:
            return store.get_delta(family, variant, layer)
        else:
            return store.get(family.row(variant), layer)
    except (KeyError, FileNotFoundError, ValueError):
        return None


def build_feature_matrix(
    families: dict[str, ItemFamily],
    labels: dict[str, Any],
    store: HiddenStateStore | ModelExtractor,
    layer: int | None,
    config: ProbingConfig,
    variant: str = "original",
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Build (X, y) arrays for a linear probe.

    Args:
        families: Dict of ItemFamily objects.
        labels: Dict mapping family_id → label value (int or str).
        store: HiddenStateStore or ModelExtractor.
        layer: Layer index to extract. None = use row.layer_index.
        config: ProbingConfig (for delta_mode, variant_for_delta).
        variant: Which variant's hidden state to use as features.
                 For delta mode, this is the variant subtracted from original.
        scaler: Pre-fitted StandardScaler (for test sets). If None, fit a new one.
        fit_scaler: Whether to fit the scaler on this data.

    Returns:
        (X, y, family_ids, scaler)
        X: [N, H] float32
        y: [N] (int or encoded)
        family_ids: list of family_id strings in the same order as rows
        scaler: fitted StandardScaler
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[Any] = []
    fids: list[str] = []

    delta_mode = config.delta_mode
    delta_variant = config.variant_for_delta or variant

    for fid, family in families.items():
        if fid not in labels:
            continue
        label = labels[fid]
        if label is None:
            continue  # skip families with unavailable labels

        vec = _get_vector(store, family, delta_variant if delta_mode else variant, layer, delta_mode)
        if vec is None:
            continue

        X_rows.append(vec)
        y_rows.append(label)
        fids.append(fid)

    if not X_rows:
        raise ValueError(
            f"No valid feature vectors found for layer={layer}, variant='{variant}'. "
            "Check that hidden_state_path is set in metadata and files exist."
        )

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.array(y_rows)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, fids, scaler


def encode_task_family_labels(
    families: dict[str, ItemFamily],
) -> tuple[dict[str, int], dict[int, str]]:
    """Return label encoding for task_family (KB/RB/Hybrid).

    Returns:
        (labels_dict, id_to_name) where labels_dict maps family_id → int label.
    """
    unique = sorted({fam.task_family for fam in families.values()})
    name_to_id = {name: i for i, name in enumerate(unique)}
    id_to_name = {i: name for name, i in name_to_id.items()}
    labels = {fid: name_to_id[fam.task_family] for fid, fam in families.items()}
    return labels, id_to_name


def extract_all_families(
    families: dict[str, ItemFamily],
    extractor: ModelExtractor,
    variant: str = "original",
    layer: int | None = None,
) -> dict[str, np.ndarray]:
    """Batch-extract hidden states for all families using a ModelExtractor.

    Args:
        families: Dict of ItemFamily objects.
        extractor: ModelExtractor instance.
        variant: Which variant to extract.
        layer: Layer index. None = return all layers.

    Returns:
        Dict mapping family_id → numpy array [H] (or [L, H] if layer is None).
    """
    fids = []
    texts = []
    for fid, family in families.items():
        if family.has_variant(variant):
            row = family.row(variant)
            text = row.extra.get("prompt", "")
            if not text:
                continue
            fids.append(fid)
            texts.append(text)

    if not texts:
        raise ValueError(
            f"No prompts found for variant '{variant}'. "
            "Ensure metadata rows have a 'prompt' field."
        )

    # Extract in batches of 16 to avoid OOM
    batch_size = 16
    all_hidden: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        hidden = extractor.extract(batch)  # [B, L, H]
        all_hidden.append(hidden)

    hidden_all = np.concatenate(all_hidden, axis=0)  # [N, L, H]

    result: dict[str, np.ndarray] = {}
    for i, fid in enumerate(fids):
        if layer is not None:
            result[fid] = hidden_all[i, layer, :]
        else:
            result[fid] = hidden_all[i]  # [L, H]

    return result
