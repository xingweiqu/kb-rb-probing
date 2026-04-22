"""Train/test split strategies for probing experiments."""

from __future__ import annotations

import random
from typing import Any

from .data import ItemFamily


class SplitFactory:
    """Generates train/test splits over item families.

    All methods return (train_ids, test_ids) as lists of family_id strings.
    """

    @staticmethod
    def random_split(
        families: dict[str, ItemFamily],
        test_size: float = 0.3,
        seed: int = 0,
    ) -> tuple[list[str], list[str]]:
        """Random stratified split by task_family."""
        from collections import defaultdict

        by_class: dict[str, list[str]] = defaultdict(list)
        for fid, fam in families.items():
            by_class[fam.task_family].append(fid)

        rng = random.Random(seed)
        train_ids: list[str] = []
        test_ids: list[str] = []

        for cls_ids in by_class.values():
            shuffled = list(cls_ids)
            rng.shuffle(shuffled)
            n_test = max(1, int(len(shuffled) * test_size))
            test_ids.extend(shuffled[:n_test])
            train_ids.extend(shuffled[n_test:])

        return train_ids, test_ids

    @staticmethod
    def held_out_subfamily(
        families: dict[str, ItemFamily],
        held_out: list[str],
    ) -> tuple[list[str], list[str]]:
        """Hold out specific sub_family values as the test set.

        Args:
            held_out: List of sub_family strings to use as test.
        """
        held_out_set = set(held_out)
        train_ids = [fid for fid, fam in families.items() if fam.sub_family not in held_out_set]
        test_ids = [fid for fid, fam in families.items() if fam.sub_family in held_out_set]

        if not test_ids:
            available = sorted({fam.sub_family for fam in families.values()})
            raise ValueError(
                f"No families found with sub_family in {held_out}. "
                f"Available sub_families: {available}"
            )
        return train_ids, test_ids

    @staticmethod
    def cross_variant(
        families: dict[str, ItemFamily],
        train_variants: list[str],
        test_variants: list[str],
    ) -> tuple[list[str], list[str]]:
        """Split by which variants are present.

        Families that have at least one train_variant go to train;
        families that have at least one test_variant go to test.
        A family can appear in both if it has both.

        This is most useful when combined with feature extraction that
        selects a specific variant's hidden state.
        """
        train_set = set(train_variants)
        test_set = set(test_variants)

        train_ids = [
            fid for fid, fam in families.items()
            if any(fam.has_variant(v) for v in train_set)
        ]
        test_ids = [
            fid for fid, fam in families.items()
            if any(fam.has_variant(v) for v in test_set)
        ]
        return train_ids, test_ids

    @staticmethod
    def cross_model(
        families: dict[str, ItemFamily],
        train_model: str,
        test_model: str,
    ) -> tuple[list[str], list[str]]:
        """Split by model_name.

        NOTE: This is only meaningful when hidden states from both models
        are in the same representational space (same architecture, same dim).
        Raises a warning if the models differ in architecture.
        """
        import warnings

        train_ids = [fid for fid, fam in families.items() if fam.model_name == train_model]
        test_ids = [fid for fid, fam in families.items() if fam.model_name == test_model]

        if not train_ids:
            available = sorted({fam.model_name for fam in families.values()})
            raise ValueError(
                f"No families found with model_name='{train_model}'. "
                f"Available: {available}"
            )
        if not test_ids:
            available = sorted({fam.model_name for fam in families.values()})
            raise ValueError(
                f"No families found with model_name='{test_model}'. "
                f"Available: {available}"
            )

        warnings.warn(
            "cross_model split requires that both models have the same hidden state "
            "dimensionality and compatible representations. Verify before interpreting results.",
            UserWarning,
            stacklevel=2,
        )
        return train_ids, test_ids

    @staticmethod
    def kb_rb_to_hybrid(
        families: dict[str, ItemFamily],
    ) -> tuple[list[str], list[str]]:
        """Train on KB + RB, test on Hybrid."""
        train_ids = [fid for fid, fam in families.items() if fam.task_family in {"KB", "RB"}]
        test_ids = [fid for fid, fam in families.items() if fam.task_family == "Hybrid"]
        if not test_ids:
            raise ValueError("No Hybrid families found for kb_rb_to_hybrid split.")
        return train_ids, test_ids

    @staticmethod
    def natural_to_symbolic(
        families: dict[str, ItemFamily],
    ) -> tuple[list[str], list[str]]:
        """Train on natural-language sub_families, test on symbolic ones.

        Symbolic sub_families are identified by the 'symbolic' substring in sub_family.
        """
        train_ids = [fid for fid, fam in families.items() if "symbolic" not in fam.sub_family.lower()]
        test_ids = [fid for fid, fam in families.items() if "symbolic" in fam.sub_family.lower()]
        if not test_ids:
            # Fall back: use symbol_substitution variant presence as proxy
            test_ids = [
                fid for fid, fam in families.items()
                if fam.has_variant("symbol_substitution")
            ]
        return train_ids, test_ids


SPLIT_REGISTRY: dict[str, Any] = {
    "random": SplitFactory.random_split,
    "held_out_subfamily": SplitFactory.held_out_subfamily,
    "cross_variant": SplitFactory.cross_variant,
    "cross_model": SplitFactory.cross_model,
    "kb_rb_to_hybrid": SplitFactory.kb_rb_to_hybrid,
    "natural_to_symbolic": SplitFactory.natural_to_symbolic,
}
