"""Experiment A: Label probe baseline.

Probes hidden representations for task_family (KB / RB / Hybrid).
This is the baseline against which signature probes are compared.

Usage:
    python -m probing.cli.run_label_probe --config config.yaml [--layers 0 8 16 24 32]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..config import load_config, ProbingConfig
from ..data import load_families
from ..features import build_feature_matrix, encode_task_family_labels
from ..hidden_states import build_store, HiddenStateStore
from ..metrics import per_layer_metrics, aggregate_seeds
from ..probes import run_probe_with_seeds
from ..report import save_run, save_per_layer_metrics, save_confusion_matrix
from ..splits import SplitFactory
from ..io_utils import save_json


def _resolve_layers(config: ProbingConfig, store: HiddenStateStore, families) -> list[int]:
    """Determine which layers to probe."""
    if config.layers == "all":
        # Infer from the first available hidden state file
        for fam in families.values():
            if fam.has_variant("original"):
                row = fam.row("original")
                if row.hidden_state_path:
                    return store.available_layers(row)
        raise ValueError(
            "Cannot determine available layers: no hidden state files found. "
            "Set 'layers' explicitly in config."
        )
    return list(config.layers)


def run_label_probe(config: ProbingConfig) -> None:
    print(f"[label_probe] Loading families from {config.metadata_path}")
    families = load_families(config)
    print(f"  {len(families)} families loaded.")

    store = build_store(config)
    if not isinstance(store, HiddenStateStore):
        raise NotImplementedError(
            "ModelExtractor is not yet supported in run_label_probe. "
            "Pre-extract hidden states and set hidden_state_root in config."
        )

    layers = _resolve_layers(config, store, families)
    print(f"  Probing {len(layers)} layers: {layers[:5]}{'...' if len(layers) > 5 else ''}")

    # Encode task_family labels
    task_labels, id_to_name = encode_task_family_labels(families)
    class_names = [id_to_name[i] for i in sorted(id_to_name)]
    print(f"  Classes: {class_names}")

    layer_results: dict[int, list[dict]] = defaultdict(list)

    for layer in layers:
        print(f"  Layer {layer}...", end=" ", flush=True)

        # Build split (same split across seeds for comparability)
        if config.split_mode == "random":
            train_ids, test_ids = SplitFactory.random_split(
                families, test_size=config.test_size, seed=config.seeds[0]
            )
        elif config.split_mode == "kb_rb_to_hybrid":
            train_ids, test_ids = SplitFactory.kb_rb_to_hybrid(families)
        else:
            raise ValueError(
                f"Unsupported split_mode '{config.split_mode}' for label probe. "
                "Use 'random' or 'kb_rb_to_hybrid'."
            )

        train_fams = {fid: families[fid] for fid in train_ids if fid in families}
        test_fams = {fid: families[fid] for fid in test_ids if fid in families}

        try:
            X_train, y_train, _, scaler = build_feature_matrix(
                train_fams, task_labels, store, layer, config, fit_scaler=True
            )
            X_test, y_test, test_fids, _ = build_feature_matrix(
                test_fams, task_labels, store, layer, config,
                scaler=scaler, fit_scaler=False
            )
        except ValueError as exc:
            print(f"SKIP ({exc})")
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

            # Save per-seed run
            predictions = [
                {"family_id": fid, "y_true": int(yt), "y_pred": int(yp)}
                for fid, yt, yp in zip(test_fids, sr["y_true"], sr["y_pred"])
            ]
            save_run(
                config.output_dir,
                experiment="label_probe",
                probe_target="task_family",
                split_mode=config.split_mode,
                seed=sr["seed"],
                metrics=sr["metrics"],
                predictions=predictions,
                coefficients=sr["coefficients"],
                config_dict={"layer": layer, "split_mode": config.split_mode, "seed": sr["seed"]},
            )

        agg = aggregate_seeds(seed_results)
        print(f"acc={agg.get('accuracy_mean', 0):.3f}±{agg.get('accuracy_std', 0):.3f}  "
              f"f1={agg.get('macro_f1_mean', 0):.3f}±{agg.get('macro_f1_std', 0):.3f}")

    # Save per-layer summary
    layer_rows = per_layer_metrics(layer_results)
    save_per_layer_metrics(
        config.output_dir, "label_probe", "task_family", config.split_mode, layer_rows
    )

    # Save confusion matrix from last layer
    if layer_results:
        last_layer = max(layer_results.keys())
        last_agg = aggregate_seeds(layer_results[last_layer])
        cm = last_agg.get("confusion_matrix")
        if cm:
            save_confusion_matrix(
                config.output_dir, "label_probe", "task_family",
                config.split_mode, cm, class_names
            )

    # Save summary JSON
    summary = {
        "experiment": "label_probe",
        "probe_target": "task_family",
        "split_mode": config.split_mode,
        "n_families": len(families),
        "n_layers": len(layers),
        "class_names": class_names,
        "per_layer": layer_rows,
    }
    from ..io_utils import save_json
    save_json(summary, Path(config.output_dir) / "label_probe" / "task_family" / config.split_mode / "summary.json")
    print(f"\n[label_probe] Done. Results in {config.output_dir}/label_probe/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A: Label probe baseline (KB/RB/Hybrid).")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file.")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Override layer indices to probe (e.g. --layers 0 8 16 24 32)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.layers is not None:
        config.layers = args.layers

    run_label_probe(config)


if __name__ == "__main__":
    main()
