"""Behavioral signature label builder.

Derives 5 binary labels from item-family scores:
  - premise_sensitive
  - scaffold_sensitive
  - removal_dependent
  - substitution_robust
  - wrong_claim_susceptible
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import ProbingConfig
from .data import ItemFamily, SCAFFOLD_VARIANTS

# Default threshold keys
TAU_PREMISE = "tau_p"
TAU_SCAFFOLD = "tau_s"
TAU_REMOVAL = "tau_r"
TAU_SUBSTITUTION = "tau_u"
TAU_WRONG_CLAIM = "tau_w"

# Variant names used for each signal
PREMISE_VARIANT = "premise"
REMOVAL_VARIANT = "premise_removal"
SUBSTITUTION_VARIANT = "structure_substitution"
WRONG_CLAIM_VARIANT = "counterfactual"


class FamilySignals:
    """Intermediate delta signals for one item family."""

    def __init__(
        self,
        family_id: str,
        task_family: str,
        sub_family: str,
        premise_gain: float | None,
        scaffold_gain: float | None,
        removal_drop: float | None,
        substitution_drop: float | None,
        wrong_claim_drop: float | None,
    ) -> None:
        self.family_id = family_id
        self.task_family = task_family
        self.sub_family = sub_family
        self.premise_gain = premise_gain
        self.scaffold_gain = scaffold_gain
        self.removal_drop = removal_drop
        self.substitution_drop = substitution_drop
        self.wrong_claim_drop = wrong_claim_drop

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "task_family": self.task_family,
            "sub_family": self.sub_family,
            "premise_gain": self.premise_gain,
            "scaffold_gain": self.scaffold_gain,
            "removal_drop": self.removal_drop,
            "substitution_drop": self.substitution_drop,
            "wrong_claim_drop": self.wrong_claim_drop,
        }


def _safe_delta(a: float | None, b: float | None) -> float | None:
    """Return a - b, or None if either is NaN or missing."""
    if a is None or b is None:
        return None
    if math.isnan(a) or math.isnan(b):
        return None
    return a - b


def compute_signals(family: ItemFamily) -> FamilySignals:
    """Compute raw delta signals for one item family."""
    orig = family.score_or_none("original")

    # premise_gain = score(premise) - score(original)
    premise_gain = _safe_delta(family.score_or_none(PREMISE_VARIANT), orig)

    # scaffold_gain = max(score(scaffold_k)) - score(original)
    scaffold_scores = family.scaffold_scores()
    if scaffold_scores and orig is not None and not math.isnan(orig):
        scaffold_gain = max(scaffold_scores) - orig
    else:
        scaffold_gain = None

    # removal_drop = score(original) - score(premise_removal)
    removal_drop = _safe_delta(orig, family.score_or_none(REMOVAL_VARIANT))

    # substitution_drop = score(original) - score(structure_substitution)
    substitution_drop = _safe_delta(orig, family.score_or_none(SUBSTITUTION_VARIANT))

    # wrong_claim_drop = score(original) - score(counterfactual)
    wrong_claim_drop = _safe_delta(orig, family.score_or_none(WRONG_CLAIM_VARIANT))

    return FamilySignals(
        family_id=family.family_id,
        task_family=family.task_family,
        sub_family=family.sub_family,
        premise_gain=premise_gain,
        scaffold_gain=scaffold_gain,
        removal_drop=removal_drop,
        substitution_drop=substitution_drop,
        wrong_claim_drop=wrong_claim_drop,
    )


class BehavioralLabelBuilder:
    """Derives binary behavioral signature labels from item-family scores.

    Args:
        config: ProbingConfig with thresholds and threshold_mode.
    """

    def __init__(self, config: ProbingConfig) -> None:
        self.config = config
        self._thresholds = dict(config.thresholds)
        self._mode = config.threshold_mode

    def build(
        self, families: dict[str, ItemFamily]
    ) -> dict[str, dict[str, int | None]]:
        """Compute 5 binary labels for each family.

        Returns:
            Dict mapping family_id → {label_name: 0/1/None}.
            None means the signal was unavailable (missing variant or NaN score).
        """
        all_signals = {fid: compute_signals(fam) for fid, fam in families.items()}

        if self._mode == "quantile":
            thresholds = self._compute_quantile_thresholds(all_signals)
        else:
            thresholds = self._thresholds

        labels: dict[str, dict[str, int | None]] = {}
        for fid, sig in all_signals.items():
            labels[fid] = self._apply_thresholds(sig, thresholds)
        return labels

    def build_with_signals(
        self, families: dict[str, ItemFamily]
    ) -> tuple[dict[str, dict[str, int | None]], dict[str, FamilySignals]]:
        """Like build(), but also returns the raw signals for inspection."""
        all_signals = {fid: compute_signals(fam) for fid, fam in families.items()}

        if self._mode == "quantile":
            thresholds = self._compute_quantile_thresholds(all_signals)
        else:
            thresholds = self._thresholds

        labels: dict[str, dict[str, int | None]] = {}
        for fid, sig in all_signals.items():
            labels[fid] = self._apply_thresholds(sig, thresholds)
        return labels, all_signals

    def _apply_thresholds(
        self, sig: FamilySignals, thresholds: dict[str, float]
    ) -> dict[str, int | None]:
        tau_p = thresholds.get(TAU_PREMISE, 0.1)
        tau_s = thresholds.get(TAU_SCAFFOLD, 0.1)
        tau_r = thresholds.get(TAU_REMOVAL, 0.1)
        tau_u = thresholds.get(TAU_SUBSTITUTION, 0.1)
        tau_w = thresholds.get(TAU_WRONG_CLAIM, 0.1)

        def _bin(value: float | None, threshold: float, direction: str = "gt") -> int | None:
            if value is None:
                return None
            if direction == "gt":
                return 1 if value > threshold else 0
            else:  # "lt"
                return 1 if value < threshold else 0

        return {
            "premise_sensitive": _bin(sig.premise_gain, tau_p, "gt"),
            "scaffold_sensitive": _bin(sig.scaffold_gain, tau_s, "gt"),
            "removal_dependent": _bin(sig.removal_drop, tau_r, "gt"),
            # substitution_robust = 1 if drop is SMALL (model is robust)
            "substitution_robust": _bin(sig.substitution_drop, tau_u, "lt"),
            "wrong_claim_susceptible": _bin(sig.wrong_claim_drop, tau_w, "gt"),
        }

    def _compute_quantile_thresholds(
        self, all_signals: dict[str, FamilySignals]
    ) -> dict[str, float]:
        """Compute thresholds as quantiles of the observed signal distributions."""
        def _quantile(values: list[float], q: float) -> float:
            arr = np.array([v for v in values if not math.isnan(v)])
            if len(arr) == 0:
                return 0.0
            return float(np.quantile(arr, q))

        premise_gains = [s.premise_gain for s in all_signals.values() if s.premise_gain is not None]
        scaffold_gains = [s.scaffold_gain for s in all_signals.values() if s.scaffold_gain is not None]
        removal_drops = [s.removal_drop for s in all_signals.values() if s.removal_drop is not None]
        sub_drops = [s.substitution_drop for s in all_signals.values() if s.substitution_drop is not None]
        wc_drops = [s.wrong_claim_drop for s in all_signals.values() if s.wrong_claim_drop is not None]

        return {
            TAU_PREMISE: _quantile(premise_gains, self._thresholds.get(TAU_PREMISE, 0.5)),
            TAU_SCAFFOLD: _quantile(scaffold_gains, self._thresholds.get(TAU_SCAFFOLD, 0.5)),
            TAU_REMOVAL: _quantile(removal_drops, self._thresholds.get(TAU_REMOVAL, 0.5)),
            TAU_SUBSTITUTION: _quantile(sub_drops, self._thresholds.get(TAU_SUBSTITUTION, 0.5)),
            TAU_WRONG_CLAIM: _quantile(wc_drops, self._thresholds.get(TAU_WRONG_CLAIM, 0.5)),
        }
