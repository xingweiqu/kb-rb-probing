"""Experiment B: Signature probe.

Probes hidden representations for each behavioral signature label:
  - premise_sensitive
  - scaffold_sensitive
  - removal_dependent
  - substitution_robust
  - wrong_claim_susceptible

Supports both absolute-state and delta-state modes.

Usage:
    python -m probing.cli.run_signature_probe --config config.yaml --target all
    python -m probing.cli.run_signature_probe --config config.yaml --target premise_sensitive --delta
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ..config import load_config, ProbingConfig
from ..data import load_families
from ..features import build_feature_matrix
from ..hidden_states import build_store, HiddenStateStore
from ..labels import BehavioralLabelBuilder
from ..metrics import per_layer_metrics, aggregate_seeds
from ..probes import run_probe_with_seeds
from ..report import save_run, save_per_layer_metrics, save_confusion_matrix
from ..splits import SplitFactory
from ..io_utils import save_json

ALL_SIGNATURE_LABELS = [
    "premise_sensitive",
    "scaffold_sensitive",
    "removal_dependent",
    "substitution_robust",
    "wrong_claim_susceptible",
]

# Which variant's delta to use for each signature label
DELTA_VARIANT_FOR_LABEL = {
    "premise_sensitive": "premise",
    "scaffold_sensitive": "context_scaffolding",
    "removal_dependent": "premise_removal",
    "substitution_robust": "structure_substitution",
    "wrong_claim_susceptible": "counterfactual",
}


def _resolve_layers(config: ProbingConfig, store: HiddenStateStore, families) -> list[int]:
    if config.layers == "all":
        for fam in families.values():
            if fam.has_variant("original"):
                row = fam.row("original")
                if row.hidden_state_path:
                    return store.available_layers(row)
        raise ValueError("Cannot determine available layers. Set 'layers' in config.")
    return list(config.layers)


def run_signature_probe_for_target(
    target: str,
    families,
    all_labels: dict,
    store: HiddenStateStore,
    layers: list[int],
    config: ProbingConfig,
    delta_mode: bool,
) -> list[dict]:
    """Run the signature probe for one target label across all layers.

    Returns per-layer aggregated metric rows.
    """
    # Extract binary labels for this target
    target_labels = {}
    for fid, label_dict in all_labels.items():
        val = label_dict.get(target)
        if val is not None:
            target_labels[fid] = int(val)

    if len(target_labels) < 4:
        print(f"  [{target}] Only {len(target_labels)} families have this label — skipping.")
        return []

    n_pos = sum(v for v in target_labels.values())
    n_neg = len(target_labels) - n_pos
    print(f"  [{target}] {len(target_labels)} families: {n_pos} positive, {n_neg} negative")

    if n_pos == 0 or n_neg == 0:
        print(f"  [{target}] All labels are the same class — skipping.")
        return []

    # Determine which variant to use for delta
    delta_variant = DELTA_VARIANT_FOR_LABEL.get(target, "original")

    # Override config delta settings for this target
    import copy
    probe_config = copy.copy(config)
    probe_config.delta_mode = delta_mode
    probe_config.variant_for_delta = delta_variant if delta_mode else None

    layer_results: dict[int, list[dict]] = defaultdict(list)

    for layer in layers:
        # Split
        label_families = {fid: families[fid] for fid in target_labels if fid in families}
        if config.split_mode == "random":
            train_ids, test_ids = SplitFactory.random_split(
                label_families, test_size=config.test_size, seed=config.seeds[0]
            )
        else:
            train_ids, test_ids = SplitFactory.random_split(
                label_families, test_size=config.test_size, seed=config.seeds[0]
            )

        train_fams = {fid: label_families[fid] for fid in train_ids}
        test_fams = {fid: label_families[fid] for fid in test_ids}

        variant = delta_variant if delta_mode else "original"

        try:
            X_train, y_train, _, scaler = build_feature_matrix(
                train_fams, target_labels, store, layer, probe_config,
                variant=variant, fit_scaler=True
            )
            X_test, y_test, test_fids, _ = build_feature_matrix(
                test_fams, target_labels, store, layer, probe_config,
                variant=variant, scaler=scaler, fit_scaler=False
            )
        except ValueError:
            continue

        seed_results = run_probe_with_seeds(
            X_train, y_train, X_test, y_test,
            seeds=config.seeds,
            probe_type=config.probe_type,
            class_weight=config.class_weight,
        )

        for sr in seed_results:
            sr["layer"] = layer
            layer_results[layer].append(sr)

            predictions = [
                {"family_id": fid, "y_true": int(yt), "y_pred": int(yp)}
                for fid, yt, yp in zip(test_fids, sr["y_true"], sr["y_pred"])
            ]
            mode_tag = "delta" if delta_mode else "absolute"
            save_run(
                config.output_dir,
                experiment="signature_probe",
                probe_target=f"{target}__{mode_tag}",
                split_mode=config.split_mode,
                seed=sr["seed"],
                metrics=sr["metrics"],
                predictions=predictions,
                coefficients=sr["coefficients"],
                config_dict={"layer": layer, "target": target, "delta_mode": delta_mode},
            )

    layer_rows = per_layer_metrics(layer_results)
    mode_tag = "delta" if delta_mode else "absolute"
    save_per_layer_metrics(
        config.output_dir, "signature_probe", f"{target}__{mode_tag}", config.split_mode, layer_rows
    )
    return layer_rows


def run_signature_probe(config: ProbingConfig, targets: list[str], delta_mode: bool) -> None:
    print(f"[signature_probe] Loading families from {config.metadata_path}")
    families = load_families(config)
    print(f"  {len(families)} families loaded.")

    store = build_store(config)
    if not isinstance(store, HiddenStateStore):
        raise NotImplementedError("ModelExtractor not yet supported here. Pre-extract hidden states.")

    layers = _resolve_layers(config, store, families)

    # Build behavioral labels
    builder = BehavioralLabelBuilder(config)
    all_labels, signals = builder.build_with_signals(families)

    # Save signals for inspection
    signals_rows = [s.to_dict() for s in signals.values()]
    save_json(signals_rows, Path(config.output_dir) / "signature_probe" / "behavioral_signals.json")

    all_layer_rows: dict[str, list[dict]] = {}
    for target in targets:
        print(f"\n  Probing target: {target} (delta={delta_mode})")
        layer_rows = run_signature_probe_for_target(
            target, families, all_labels, store, layers, config, delta_mode
        )
        all_layer_rows[target] = layer_rows

    # Save comparison summary
    summary = {
        "experiment": "signature_probe",
        "delta_mode": delta_mode,
        "targets": targets,
        "n_families": len(families),
        "n_layers": len(layers),
        "per_target_per_layer": all_layer_rows,
    }
    mode_tag = "delta" if delta_mode else "absolute"
    save_json(
        summary,
        Path(config.output_dir) / "signature_probe" / f"summary_{mode_tag}.json"
    )
    print(f"\n[signature_probe] Done. Results in {config.output_dir}/signature_probe/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B: Signature probe.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file.")
    parser.add_argument(
        "--target", default="all",
        help="Signature label to probe. One of: " + ", ".join(ALL_SIGNATURE_LABELS) + ", or 'all'."
    )
    parser.add_argument(
        "--delta", action="store_true",
        help="Use delta-state mode (h_variant - h_original) instead of absolute hidden state."
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Override layer indices."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.layers is not None:
        config.layers = args.layers

    if args.target == "all":
        targets = ALL_SIGNATURE_LABELS
    elif args.target in ALL_SIGNATURE_LABELS:
        targets = [args.target]
    else:
        raise ValueError(
            f"Unknown target '{args.target}'. "
            f"Choose from: {ALL_SIGNATURE_LABELS} or 'all'."
        )

    run_signature_probe(config, targets, delta_mode=args.delta)


if __name__ == "__main__":
    main()
