"""Linear probe wrappers (sklearn-based)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


class LinearProbe:
    """Thin wrapper around sklearn LogisticRegression or Ridge.

    Args:
        probe_type: "logistic" for classification, "ridge" for regression.
        class_weight: Passed to LogisticRegression. Use "balanced" for imbalanced labels.
        seed: Random state for reproducibility.
        max_iter: Max iterations for LogisticRegression solver.
        C: Regularization strength (inverse) for LogisticRegression.
        alpha: Regularization strength for Ridge.
    """

    def __init__(
        self,
        probe_type: str = "logistic",
        class_weight: str | None = "balanced",
        seed: int = 0,
        max_iter: int = 1000,
        C: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        self.probe_type = probe_type
        self.seed = seed

        if probe_type == "logistic":
            self._model = LogisticRegression(
                class_weight=class_weight,
                random_state=seed,
                max_iter=max_iter,
                C=C,
                solver="lbfgs",
                multi_class="auto",
            )
        elif probe_type == "ridge":
            self._model = Ridge(alpha=alpha, random_state=seed)
        else:
            raise ValueError(f"Unknown probe_type '{probe_type}'. Expected 'logistic' or 'ridge'.")

        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """Fit the probe on training data."""
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Training labels contain only one class: {np.unique(y)}. "
                "Cannot fit a probe with a single class."
            )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted labels."""
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return predicted probabilities (logistic only)."""
        self._check_fitted()
        if self.probe_type == "logistic":
            return self._model.predict_proba(X)
        return None

    def coefficients(self) -> np.ndarray:
        """Return the probe weight matrix.

        Shape: [n_classes, H] for logistic, [H] for ridge.
        """
        self._check_fitted()
        if self.probe_type == "logistic":
            return self._model.coef_
        return self._model.coef_

    def classes(self) -> np.ndarray | None:
        """Return class labels (logistic only)."""
        if self.probe_type == "logistic":
            return self._model.classes_
        return None

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Probe has not been fitted. Call fit() first.")


def run_probe_with_seeds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seeds: list[int],
    probe_type: str = "logistic",
    class_weight: str | None = "balanced",
) -> list[dict]:
    """Run a probe across multiple seeds and return per-seed results.

    Args:
        X_train, y_train: Training features and labels.
        X_test, y_test: Test features and labels.
        seeds: List of random seeds.
        probe_type: "logistic" or "ridge".
        class_weight: Passed to LogisticRegression.

    Returns:
        List of dicts, one per seed, each containing predictions and probabilities.
    """
    from .metrics import compute_metrics

    results = []
    for seed in seeds:
        probe = LinearProbe(probe_type=probe_type, class_weight=class_weight, seed=seed)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        y_proba = probe.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        results.append({
            "seed": seed,
            "metrics": metrics,
            "y_pred": y_pred.tolist(),
            "y_true": y_test.tolist(),
            "coefficients": probe.coefficients().tolist(),
        })
    return results
