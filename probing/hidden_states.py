"""Hidden state loading and extraction.

Two modes:
1. File mode: load pre-saved .pt / .npy files referenced in metadata.
2. Model mode: extract hidden states on-the-fly from a HuggingFace model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import ProbingConfig
from .data import ItemFamily, ItemRow
from .io_utils import load_tensor


class HiddenStateStore:
    """Loads hidden states from pre-saved files.

    Each file is expected to contain a tensor of shape:
        [L, H]  — all layers, one position
        [H]     — single layer / position

    The layer to extract is selected via the `layer` argument.
    """

    def __init__(self, root_dir: str, default_position_type: str | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self.default_position_type = default_position_type
        self._cache: dict[str, np.ndarray] = {}

    def _load_file(self, path: str, key: str | None) -> np.ndarray:
        """Load and cache a tensor file."""
        cache_key = f"{path}::{key}"
        if cache_key not in self._cache:
            full_path = Path(path) if Path(path).is_absolute() else self.root_dir / path
            self._cache[cache_key] = load_tensor(full_path, key)
        return self._cache[cache_key]

    def get(
        self,
        row: ItemRow,
        layer: int | None = None,
    ) -> np.ndarray:
        """Return the hidden state vector for a single row.

        Args:
            row: ItemRow with hidden_state_path and optional hidden_state_key.
            layer: Layer index to extract. If None, uses row.layer_index.
                   If the tensor is 1-D (already a single vector), returns as-is.

        Returns:
            1-D numpy array of shape [H].
        """
        if not row.hidden_state_path:
            raise ValueError(
                f"Row '{row.uid}' has no hidden_state_path. "
                "Either provide pre-saved files or use ModelExtractor."
            )
        arr = self._load_file(row.hidden_state_path, row.hidden_state_key)

        # arr shape: [L, H] or [H]
        if arr.ndim == 1:
            return arr

        if arr.ndim == 2:
            target_layer = layer if layer is not None else row.layer_index
            if target_layer is None:
                raise ValueError(
                    f"Tensor for '{row.uid}' has shape {arr.shape} but no layer index specified. "
                    "Set layer in config or provide layer_index in metadata."
                )
            if target_layer >= arr.shape[0]:
                raise IndexError(
                    f"Layer {target_layer} out of range for tensor shape {arr.shape} "
                    f"(row '{row.uid}')."
                )
            return arr[target_layer]

        raise ValueError(
            f"Unexpected tensor shape {arr.shape} for row '{row.uid}'. Expected [L, H] or [H]."
        )

    def get_delta(
        self,
        family: ItemFamily,
        variant: str,
        layer: int | None = None,
    ) -> np.ndarray:
        """Return h_variant - h_original for a family.

        Args:
            family: ItemFamily containing both 'original' and the target variant.
            variant: The variant to subtract original from.
            layer: Layer index.

        Returns:
            1-D numpy array [H].
        """
        if not family.has_variant("original"):
            raise KeyError(
                f"Family '{family.family_id}' has no 'original' variant; "
                "cannot compute delta state."
            )
        if not family.has_variant(variant):
            raise KeyError(
                f"Family '{family.family_id}' has no variant '{variant}'; "
                "cannot compute delta state."
            )
        h_orig = self.get(family.row("original"), layer)
        h_var = self.get(family.row(variant), layer)
        return h_var - h_orig

    def available_layers(self, row: ItemRow) -> list[int]:
        """Return the list of available layer indices for a row's tensor file."""
        arr = self._load_file(row.hidden_state_path, row.hidden_state_key)
        if arr.ndim == 1:
            return [0]
        return list(range(arr.shape[0]))


class ModelExtractor:
    """Extract hidden states on-the-fly from a HuggingFace model.

    Reuses the same logic as probe.py's get_last_token_hidden.
    """

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 256) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for model extraction."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def extract(self, texts: list[str]) -> np.ndarray:
        """Extract last-token hidden states for a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            numpy array of shape [B, L, H] where L includes the embedding layer.
        """
        import torch

        self._load()
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each [B, T, H]
        hidden = torch.stack(outputs.hidden_states, dim=1)  # [B, L, T, H]

        # Extract the last non-pad token for each example
        attention_mask = inputs["attention_mask"]  # [B, T]
        last_token_idx = attention_mask.sum(dim=1) - 1  # [B]

        result = []
        for b in range(hidden.shape[0]):
            idx = last_token_idx[b].item()
            result.append(hidden[b, :, idx, :].float().cpu().numpy())  # [L, H]

        return np.stack(result, axis=0)  # [B, L, H]

    def num_layers(self) -> int:
        """Return the number of transformer layers (excluding embedding)."""
        self._load()
        return len(self._model.model.layers)

    def unload(self) -> None:
        """Free GPU memory."""
        import gc
        import torch
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_store(config: ProbingConfig) -> HiddenStateStore | ModelExtractor:
    """Return the appropriate hidden state provider based on config."""
    if config.model_name:
        return ModelExtractor(
            model_name=config.model_name,
            device=config.device,
            max_length=config.max_length,
        )
    return HiddenStateStore(
        root_dir=config.hidden_state_root,
        default_position_type=config.position_type,
    )
