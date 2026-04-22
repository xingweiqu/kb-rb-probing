"""Data loading: metadata JSONL → ItemFamily objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ProbingConfig
from .io_utils import read_jsonl

# Variants produced by generate_items.py
KNOWN_VARIANTS = {
    "original",
    "paraphrase",
    "counterfactual",
    "hint",
    "premise",
    "context_scaffolding",
    "premise_removal",
    "structure_substitution",
    "symbol_substitution",
    # scaffold levels (probe_v2.py may emit these)
    "scaffold_0",
    "scaffold_1",
    "scaffold_2",
    "scaffold_3",
}

# Canonical scaffold variant names in priority order
SCAFFOLD_VARIANTS = ["scaffold_0", "scaffold_1", "scaffold_2", "scaffold_3", "context_scaffolding"]


@dataclass
class ItemRow:
    """One row from the metadata JSONL — one (family, variant) pair."""
    uid: str
    family_id: str
    task_family: str          # KB / RB / Hybrid
    sub_family: str           # e.g. factual_entity, arithmetic, logic, multihop
    variant: str
    score: float
    model_name: str
    split: str
    hidden_state_path: str
    hidden_state_key: str | None
    layer_index: int | None
    position_type: str | None
    extra: dict[str, Any]


@dataclass
class ItemFamily:
    """All rows belonging to one item family (same family_id).

    Provides convenient accessors for scores and metadata.
    """
    family_id: str
    task_family: str
    sub_family: str
    model_name: str
    rows: dict[str, ItemRow] = field(default_factory=dict)  # variant → row

    @property
    def variants(self) -> list[str]:
        return list(self.rows.keys())

    def has_variant(self, variant: str) -> bool:
        return variant in self.rows

    def score(self, variant: str) -> float:
        """Return the score for a variant. Raises KeyError if missing."""
        if variant not in self.rows:
            raise KeyError(
                f"Family '{self.family_id}' has no variant '{variant}'. "
                f"Available: {self.variants}"
            )
        return self.rows[variant].score

    def score_or_none(self, variant: str) -> float | None:
        """Return the score for a variant, or None if missing."""
        row = self.rows.get(variant)
        return row.score if row is not None else None

    def scaffold_scores(self) -> list[float]:
        """Return scores for all scaffold variants present, in order."""
        scores = []
        for v in SCAFFOLD_VARIANTS:
            s = self.score_or_none(v)
            if s is not None:
                scores.append(s)
        return scores

    def row(self, variant: str) -> ItemRow:
        if variant not in self.rows:
            raise KeyError(f"Family '{self.family_id}' has no variant '{variant}'.")
        return self.rows[variant]


def _apply_field_map(raw: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    """Rename keys in raw according to field_map (internal_name → actual_name)."""
    result = dict(raw)
    for internal, actual in field_map.items():
        if actual in result and internal not in result:
            result[internal] = result[actual]
    return result


def _flatten_items_jsonl_row(raw: dict[str, Any], field_map: dict[str, str]) -> list[dict[str, Any]]:
    """Expand an items.jsonl row (one family with a 'variants' dict) into per-variant rows.

    items.jsonl format (from generate_items.py):
        {"id": "kb_001", "category": "KB", "gold_answer": "...", "variants": {...}}

    This is different from the probe_v2.py metadata format where each row is one variant.
    We detect the items.jsonl format by the presence of a "variants" dict.
    """
    mapped = _apply_field_map(raw, field_map)
    family_id = mapped.get("family_id") or mapped.get("id", "")
    task_family = mapped.get("task_family") or mapped.get("category", "")
    gold_answer = mapped.get("gold_answer", "")
    variants_dict = mapped.get("variants", {})

    if not isinstance(variants_dict, dict):
        return []

    rows = []
    for variant, prompt in variants_dict.items():
        rows.append({
            "uid": f"{family_id}__{variant}",
            "family_id": family_id,
            "task_family": task_family,
            "sub_family": mapped.get("sub_family", ""),
            "variant": variant,
            "score": mapped.get("score", float("nan")),  # scores filled in by probe_v2.py
            "model_name": mapped.get("model_name", ""),
            "split": mapped.get("split", ""),
            "hidden_state_path": mapped.get("hidden_state_path", ""),
            "hidden_state_key": mapped.get("hidden_state_key"),
            "layer_index": mapped.get("layer_index"),
            "position_type": mapped.get("position_type"),
            "gold_answer": gold_answer,
            "prompt": prompt,
            "extra": {k: v for k, v in mapped.items()
                      if k not in {"uid", "family_id", "task_family", "sub_family",
                                   "variant", "score", "model_name", "split",
                                   "hidden_state_path", "hidden_state_key",
                                   "layer_index", "position_type", "variants"}},
        })
    return rows


def load_metadata(path: str, field_map: dict[str, str]) -> list[dict[str, Any]]:
    """Load metadata JSONL and normalize to per-variant rows.

    Handles two formats:
    1. items.jsonl (generate_items.py): one row per family with a 'variants' dict
    2. probe_v2.py output: one row per (family, variant) pair
    """
    raw_rows = read_jsonl(path)
    if not raw_rows:
        raise ValueError(f"Metadata file is empty: {path}")

    # Detect format by checking the first row
    first = raw_rows[0]
    mapped_first = _apply_field_map(first, field_map)
    is_items_jsonl = isinstance(mapped_first.get("variants"), dict)

    normalized: list[dict[str, Any]] = []
    for raw in raw_rows:
        if is_items_jsonl:
            normalized.extend(_flatten_items_jsonl_row(raw, field_map))
        else:
            mapped = _apply_field_map(raw, field_map)
            normalized.append(mapped)

    return normalized


def _parse_row(raw: dict[str, Any]) -> ItemRow:
    """Parse a normalized metadata row into an ItemRow."""
    required = ["uid", "family_id", "task_family", "variant"]
    for f in required:
        if not raw.get(f):
            raise ValueError(
                f"Metadata row is missing required field '{f}'. "
                f"Row keys: {list(raw.keys())}. "
                "Check your field_map in config."
            )
    score_raw = raw.get("score", float("nan"))
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = float("nan")

    return ItemRow(
        uid=str(raw["uid"]),
        family_id=str(raw["family_id"]),
        task_family=str(raw["task_family"]),
        sub_family=str(raw.get("sub_family", "") or ""),
        variant=str(raw["variant"]),
        score=score,
        model_name=str(raw.get("model_name", "") or ""),
        split=str(raw.get("split", "") or ""),
        hidden_state_path=str(raw.get("hidden_state_path", "") or ""),
        hidden_state_key=raw.get("hidden_state_key"),
        layer_index=raw.get("layer_index"),
        position_type=raw.get("position_type"),
        extra={k: v for k, v in raw.items()
               if k not in {"uid", "family_id", "task_family", "sub_family",
                             "variant", "score", "model_name", "split",
                             "hidden_state_path", "hidden_state_key",
                             "layer_index", "position_type"}},
    )


def group_by_family(rows: list[dict[str, Any]]) -> dict[str, ItemFamily]:
    """Group normalized metadata rows into ItemFamily objects keyed by family_id."""
    families: dict[str, ItemFamily] = {}
    for raw in rows:
        item = _parse_row(raw)
        if item.family_id not in families:
            families[item.family_id] = ItemFamily(
                family_id=item.family_id,
                task_family=item.task_family,
                sub_family=item.sub_family,
                model_name=item.model_name,
            )
        fam = families[item.family_id]
        if item.variant in fam.rows:
            # Keep the row with the higher score if duplicates exist
            if item.score > fam.rows[item.variant].score:
                fam.rows[item.variant] = item
        else:
            fam.rows[item.variant] = item
    return families


def load_families(config: ProbingConfig) -> dict[str, ItemFamily]:
    """Top-level loader: read metadata and return grouped ItemFamily dict."""
    rows = load_metadata(config.metadata_path, config.field_map)
    families = group_by_family(rows)
    if not families:
        raise ValueError(f"No item families found in {config.metadata_path}")
    return families
