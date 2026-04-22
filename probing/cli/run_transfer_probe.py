"""Experiment C: Cross-generalization probe.

Tests transfer performance for both label probe and signature probe across:
  - kb_rb_to_hybrid: train on KB+RB, test on Hybrid
  - natural_to_symbolic: train on natural-language items, test on symbolic
  - variant_transfer: train on one variant mode, test on another
  - subfamily_transfer: train on selected sub_families, test on held-out ones

Usage:
    python -m probing.cli.run_transfer_probe --config config.yaml --setup kb_rb_to_hybrid
    python -m probing.cli.run_transfer_probe --config config.yaml --setup all
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import load_config, ProbingConfig
from ..data import load_families, ItemFamily
from ..features import build_feature_matrix, encode_task_family_labels
from ..hidden_states import build_store, HiddenStateStore
from ..labels import BehavioralLabelBuilder
from ..metrics import aggregate_seeds
from ..probes import run_probe_with_seeds
from ..report import save_transfer_comparison
from ..splits import SplitFactory
from ..io_utils import save_json

TRANSFER_SETUPS = ["kb_rb_to_hybrid", "natural_to_symbolic", "variant_transfer", "subfamily_transfer"]


def _probe_split(
    train_fams: dict[str, ItemFamily],
    test_fams: dict[str, ItemFamily],
    labels: dict[str, Any],
    store: HiddenStateStore,
    layer: int,
    config: ProbingConfig,
    variant: str = "original",
) -> dict[str, Any] | None:
    """Fit probe on train, evaluate on test. Returns aggregated metrics or None on failure."""
    try:
        X_train, y_train, _, scaler = build_feature_matrix(
            train_fams, labels, store, layer, config, variant=variant, fit_scaler=True
        )
        X_test, y_test, _, _ = build_feature_matrix(
            test_fams, labels, store, layer, config, variant=variant,
            scaler=scaler, fit_scaler=False
        )
    except ValueError:
        return None

    seed_results = run_probe_with_seeds(
        X_train, y_train, X_test, y_test,
        seeds=config.seeds,
        probe_type=config.probe_type,
        class_weight=config.class_weight,
    )
    return aggregate_seeds(seed_results)


def _in_domain_metrics(
    families: dict[str, ItemFamily],
    labels: dict[str, Any],
    store: HiddenStateStore,
    layer: int,
    config: ProbingConfig,
) -> dict[str, Any] | None:
    """Random split in-domain performance for comparison."""
    train_ids, test_ids = SplitFactory.random_split(families, test_size=config.test_size, seed=0)
    train_fams = {fid: families[fid] for fid in train_ids if fid in families}
    test_fams = {fid: families[fid] for fid in test_ids if fid in families}
    return _probe_split(train_fams, test_fams, labels, store, layer, config)


def run_transfer_setup(
    setup: str,
    families: dict[str, ItemFamily],
    store: HiddenStateStore,
    layers: list[int],
    config: ProbingConfig,
    subfamily_held_out: list[str] | None = None,
    train_variants: list[str] | None = None,
    test_variants: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run transfer evaluation for one setup across all layers.

    Returns a list of comparison rows (one per layer × probe_type).
    """
    # Build labels for both label probe and signature probe
    task_labels, id_to_name = encode_task_family_labels(families)
    builder = BehavioralLabelBuilder(config)
    sig_labels = builder.build(families)

    # Get train/test split for this setup
    if setup == "kb_rb_to_hybrid":
        train_ids, test_ids = SplitFactory.kb_rb_to_hybrid(families)
    elif setup == "natural_to_symbolic":
        train_ids, test_ids = SplitFactory.natural_to_symbolic(families)
    elif setup == "subfamily_transfer":
        held_out = subfamily_held_out or []
        if not held_out:
            # Auto-select: hold out the last sub_family alphabetically
            all_subs = sorted({fam.sub_family for fam in families.values() if fam.sub_family})
            held_out = [all_subs[-1]] if all_subs else []
        train_ids, test_ids = SplitFactory.held_out_subfamily(families, held_out)
    elif setup == "variant_transfer":
        tv = train_variants or ["original", "paraphrase"]
        ev = test_variants or ["counterfactual", "symbol_substitution"]
        train_ids, test_ids = SplitFactory.cross_variant(families, tv, ev)
    else:
        raise ValueError(f"Unknown setup '{setup}'.")

    if not train_ids or not test_ids:
        print(f"  [{setup}] Empty train or test split — skipping.")
        return []

    train_fams = {fid: families[fid] for fid in train_ids if fid in families}
    test_fams = {fid: families[fid] for fid in test_ids if fid in families}

    rows: list[dict[str, Any]] = []

    for layer in layers:
        # Label probe transfer
        label_transfer = _probe_split(train_fams, test_fams, task_labels, store, layer, config)
        label_indomain = _in_domain_metrics(families, task_labels, store, layer, config)

        # Signature probe transfer (one per signature label)
        for sig_name in ["premise_sensitive", "scaffold_sensitive", "removal_dependent",
                         "substitution_robust", "wrong_claim_susceptible"]:
            sig_target = {fid: d.get(sig_name) for fid, d in sig_labels.items() if d.get(sig_name) is not None}
            if len(sig_target) < 4:
                continue

            sig_train = {fid: train_fams[fid] for fid in train_ids if fid in train_fams and fid in sig_target}
            sig_test = {fid: test_fams[fid] for fid in test_ids if fid in test_fams and fid in sig_target}

            sig_transfer = _probe_split(sig_train, sig_test, sig_target, store, layer, config)
            sig_indomain = _in_domain_metrics(
                {fid: families[fid] for fid in sig_target if fid in families},
                sig_target, store, layer, config
            )

            row: dict[str, Any] = {
                "setup": setup,
                "layer": layer,
                "probe": f"signature__{sig_name}",
            }
            if sig_indomain:
                row["indomain_accuracy"] = sig_indomain.get("accuracy_mean")
                row["indomain_f1"] = sig_indomain.get("macro_f1_mean")
            if sig_transfer:
                row["transfer_accuracy"] = sig_transfer.get("accuracy_mean")
                row["transfer_f1"] = sig_transfer.get("macro_f1_mean")
            if sig_indomain and sig_transfer:
                row["transfer_drop_accuracy"] = (
                    (sig_indomain.get("accuracy_mean") or 0) - (sig_transfer.get("accuracy_mean") or 0)
                )
                row["transfer_drop_f1"] = (
                    (sig_indomain.get("macro_f1_mean") or 0) - (sig_transfer.get("macro_f1_mean") or 0)
                )
            rows.append(row)

        # Add label probe row
        label_row: dict[str, Any] = {
            "setup": setup,
            "layer": layer,
            "probe": "label__task_family",
        }
        if label_indomain:
            label_row["indomain_accuracy"] = label_indomain.get("accuracy_mean")
            label_row["indomain_f1"] = label_indomain.get("macro_f1_mean")
        if label_transfer:
            label_row["transfer_accuracy"] = label_transfer.get("accuracy_mean")
            label_row["transfer_f1"] = label_transfer.get("macro_f1_mean")
        if label_indomain and label_transfer:
            label_row["transfer_drop_accuracy"] = (
                (label_indomain.get("accuracy_mean") or 0) - (label_transfer.get("accuracy_mean") or 0)
            )
            label_row["transfer_drop_f1"] = (
                (label_indomain.get("macro_f1_mean") or 0) - (label_transfer.get("macro_f1_mean") or 0)
            )
        rows.append(label_row)

    return rows


def run_transfer_probe(
    config: ProbingConfig,
    setups: list[str],
    subfamily_held_out: list[str] | None = None,
    train_variants: list[str] | None = None,
    test_variants: list[str] | None = None,
) -> None:
    print(f"[transfer_probe] Loading families from {config.metadata_path}")
    families = load_families(config)
    print(f"  {len(families)} families loaded.")

    store = build_store(config)
    if not isinstance(store, HiddenStateStore):
        raise NotImplementedError("ModelExtractor not yet supported here.")

    layers: list[int]
    if config.layers == "all":
        for fam in families.values():
            if fam.has_variant("original"):
                row = fam.row("original")
                if row.hidden_state_path:
                    layers = store.available_layers(row)
                    break
        else:
            raise ValueError("Cannot determine layers. Set 'layers' in config.")
    else:
        layers = list(config.layers)

    for setup in setups:
        print(f"\n  Setup: {setup}")
        try:
            rows = run_transfer_setup(
                setup, families, store, layers, config,
                subfamily_held_out=subfamily_held_out,
                train_variants=train_variants,
                test_variants=test_variants,
            )
        except ValueError as exc:
            print(f"  SKIP: {exc}")
            continue

        if rows:
            save_transfer_comparison(config.output_dir, setup, rows)
            print(f"  Saved {len(rows)} rows for setup '{setup}'.")

    print(f"\n[transfer_probe] Done. Results in {config.output_dir}/transfer/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment C: Cross-generalization probe.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file.")
    parser.add_argument(
        "--setup", default="all",
        help="Transfer setup. One of: " + ", ".join(TRANSFER_SETUPS) + ", or 'all'."
    )
    parser.add_argument(
        "--held-out-subfamily", nargs="+", default=None,
        help="Sub-family names to hold out (for subfamily_transfer setup)."
    )
    parser.add_argument(
        "--train-variants", nargs="+", default=None,
        help="Variants to train on (for variant_transfer setup)."
    )
    parser.add_argument(
        "--test-variants", nargs="+", default=None,
        help="Variants to test on (for variant_transfer setup)."
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Override layer indices."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.layers is not None:
        config.layers = args.layers

    if args.setup == "all":
        setups = TRANSFER_SETUPS
    elif args.setup in TRANSFER_SETUPS:
        setups = [args.setup]
    else:
        raise ValueError(f"Unknown setup '{args.setup}'. Choose from: {TRANSFER_SETUPS} or 'all'.")

    run_transfer_probe(
        config, setups,
        subfamily_held_out=args.held_out_subfamily,
        train_variants=args.train_variants,
        test_variants=args.test_variants,
    )


if __name__ == "__main__":
    main()
