"""Train linear probes for atomic capabilities."""

import json
import logging
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_linear_probe(
    hidden_states_path: str,
    labels_path: str,
    output_path: str,
    split: str = "random"
) -> dict[str, dict[int, float]]:
    """
    Train linear classifier for each atomic capability.

    Args:
        hidden_states_path: Pickle file with hidden states
        labels_path: JSON file with atomic labels
        output_path: Output JSON file
        split: "random" | "transfer" | "metadata_controlled"

    Returns:
        results: dict[atomic_capability] -> dict[layer] -> accuracy
    """
    # Load hidden states
    logger.info(f"Loading hidden states from {hidden_states_path}...")
    with open(hidden_states_path, "rb") as f:
        states = pickle.load(f)

    # Load labels
    logger.info(f"Loading labels from {labels_path}...")
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    # Get all capabilities
    all_capabilities = set()
    for family_labels in labels.values():
        all_capabilities.update(family_labels.keys())

    results = {}

    for capability in sorted(all_capabilities):
        logger.info(f"\nProbing {capability}...")

        # Prepare data
        X_list = []
        y_list = []
        family_ids = []

        for family_id, family_labels in labels.items():
            if capability not in family_labels or family_labels[capability] is None:
                continue

            # Get hidden state for original variant
            key = f"{family_id}_original_natural"
            if key not in states:
                continue

            X_list.append(states[key])  # [n_layers, hidden_dim]
            y_list.append(family_labels[capability])
            family_ids.append(family_id)

        if len(set(y_list)) < 2:
            logger.warning(f"  Skipping {capability}: only one class")
            continue

        X = np.array(X_list)  # [n_samples, n_layers, hidden_dim]
        y = np.array(y_list)  # [n_samples]

        logger.info(f"  Data: {len(X)} samples, pos={y.sum()}, neg={(~y).sum()}")

        # Split
        if split == "random":
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
        else:
            # For now, just use random split
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train probe for each layer
        layer_results = {}
        n_layers = X.shape[1]

        for layer in range(n_layers):
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train[:, layer, :], y_train)
            acc = clf.score(X_test[:, layer, :], y_test)
            layer_results[layer] = float(acc)

        best_layer = max(layer_results, key=layer_results.get)
        best_acc = layer_results[best_layer]

        logger.info(f"  Best layer: {best_layer}, accuracy: {best_acc:.3f}")

        results[capability] = layer_results

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved results to {output_path}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states", required=True, help="Hidden states pickle")
    parser.add_argument("--labels", required=True, help="Atomic labels JSON")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--split", default="random", choices=["random", "transfer", "metadata_controlled"])

    args = parser.parse_args()

    train_linear_probe(
        hidden_states_path=args.hidden_states,
        labels_path=args.labels,
        output_path=args.output,
        split=args.split
    )


if __name__ == "__main__":
    main()
