"""Configuration dataclass and loader for the probing toolkit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProbingConfig:
    # ── paths ──────────────────────────────────────────────────────────────────
    metadata_path: str = ""
    hidden_state_root: str = ""
    output_dir: str = "probing_outputs"

    # ── field name mapping ─────────────────────────────────────────────────────
    # Maps expected internal field names to actual field names in the JSONL.
    # E.g. {"family_id": "id", "task_family": "category"} for items.jsonl.
    field_map: dict[str, str] = field(default_factory=dict)

    # ── scoring ────────────────────────────────────────────────────────────────
    score_field: str = "score"

    # ── behavioral label thresholds ────────────────────────────────────────────
    # Keys: tau_p (premise), tau_s (scaffold), tau_r (removal),
    #       tau_u (substitution), tau_w (wrong_claim)
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "tau_p": 0.1,
        "tau_s": 0.1,
        "tau_r": 0.1,
        "tau_u": 0.1,
        "tau_w": 0.1,
    })
    # "fixed" uses thresholds directly; "quantile" treats them as quantile cutoffs
    threshold_mode: str = "fixed"

    # ── probe settings ─────────────────────────────────────────────────────────
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # List of layer indices, or "all" to use every available layer
    layers: list[int] | str = "all"
    position_type: str | None = None   # e.g. "final_input", "pre_answer", "span_mean"
    probe_type: str = "logistic"       # "logistic" | "ridge"
    class_weight: str | None = "balanced"

    # ── split settings ─────────────────────────────────────────────────────────
    split_mode: str = "random"         # "random" | "held_out_subfamily" | "cross_variant" | "cross_model"
    test_size: float = 0.3

    # ── hidden state mode ──────────────────────────────────────────────────────
    delta_mode: bool = False           # use h_variant - h_original
    variant_for_delta: str | None = None  # which variant to subtract from original
    variants_to_include: list[str] = field(default_factory=list)  # empty = all

    # ── optional: extract hidden states from a model directly ──────────────────
    model_name: str | None = None      # HuggingFace model path or name
    device: str = "cuda"
    max_length: int = 256

    def resolve_field(self, name: str) -> str:
        """Return the actual field name in the JSONL for an internal field name."""
        return self.field_map.get(name, name)


def load_config(path: str | Path) -> ProbingConfig:
    """Load a ProbingConfig from a JSON or YAML file.

    YAML support requires PyYAML. Falls back to JSON if YAML is unavailable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any]
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML configs: pip install pyyaml"
            ) from exc
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

    cfg = ProbingConfig()
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: '{key}'")
        setattr(cfg, key, value)
    return cfg


def default_config_for_items_jsonl(
    metadata_path: str,
    hidden_state_root: str,
    output_dir: str = "probing_outputs",
) -> ProbingConfig:
    """Return a config pre-wired for the items.jsonl format produced by generate_items.py."""
    return ProbingConfig(
        metadata_path=metadata_path,
        hidden_state_root=hidden_state_root,
        output_dir=output_dir,
        # items.jsonl uses "id" and "category" instead of "family_id" / "task_family"
        field_map={
            "family_id": "id",
            "task_family": "category",
        },
    )
