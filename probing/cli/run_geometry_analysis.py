"""Geometry analysis: pairwise distances between variant hidden states.

Usage:
    python -m probing.cli.run_geometry_analysis --config config.yaml --layer 16
    python -m probing.cli.run_geometry_analysis --config config.yaml --layers 0 8 16 24 32
    python -m probing.cli.run_geometry_analysis --config config.yaml --all-layers
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config, ProbingConfig
from ..data import load_families
from ..geometry import (
    pairwise_distances,
    pairwise_distances_all_layers,
    summarize_by_group,
    DEFAULT_VARIANT_PAIRS,
)
from ..hidden_states import build_store, HiddenStateStore
from ..report import save_geometry_results
from ..io_utils import save_csv


def run_geometry_analysis(
    config: ProbingConfig,
    layers: list[int],
    metric: str = "cosine",
    variant_pairs: list[tuple[str, str]] | None = None,
) -> None:
    print(f"[geometry] Loading families from {config.metadata_path}")
    families = load_families(config)
    print(f"  {len(families)} families loaded.")

    store = build_store(config)
    if not isinstance(store, HiddenStateStore):
        raise NotImplementedError("ModelExtractor not yet supported for geometry analysis.")

    pairs = variant_pairs or DEFAULT_VARIANT_PAIRS
    print(f"  Variant pairs: {pairs}")
    print(f"  Layers: {layers}")
    print(f"  Metric: {metric}")

    if len(layers) == 1:
        layer = layers[0]
        rows = pairwise_distances(families, store, layer, pairs, metric)
        summary = summarize_by_group(rows, ["task_family", "variant_a", "variant_b"])
        save_geometry_results(config.output_dir, rows, summary, layer=layer)
        print(f"  {len(rows)} distance pairs computed for layer {layer}.")
    else:
        rows = pairwise_distances_all_layers(families, store, layers, pairs, metric)
        # Summary grouped by layer + task_family + variant pair
        summary_by_layer = summarize_by_group(rows, ["layer", "task_family", "variant_a", "variant_b"])
        # Summary collapsed across layers (for a quick overview)
        summary_overall = summarize_by_group(rows, ["task_family", "variant_a", "variant_b"])

        save_geometry_results(config.output_dir, rows, summary_by_layer, layer="all")
        save_csv(
            summary_overall,
            Path(config.output_dir) / "geometry" / "layer_all" / "distance_summary_overall.csv"
        )
        print(f"  {len(rows)} distance pairs computed across {len(layers)} layers.")

    print(f"\n[geometry] Done. Results in {config.output_dir}/geometry/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry analysis: pairwise variant distances.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file.")

    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int, help="Single layer index to analyze.")
    layer_group.add_argument(
        "--layers", nargs="+", type=int, help="Multiple layer indices."
    )
    layer_group.add_argument(
        "--all-layers", action="store_true", help="Use all available layers."
    )

    parser.add_argument(
        "--metric", default="cosine", choices=["cosine", "euclidean"],
        help="Distance metric."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.layer is not None:
        layers = [args.layer]
    elif args.layers is not None:
        layers = args.layers
    else:
        # all-layers: resolve from config or infer
        if config.layers == "all":
            # Will be resolved inside run_geometry_analysis via store
            from ..data import load_families as _lf
            from ..hidden_states import build_store as _bs, HiddenStateStore
            fams = _lf(config)
            st = _bs(config)
            if isinstance(st, HiddenStateStore):
                for fam in fams.values():
                    if fam.has_variant("original"):
                        row = fam.row("original")
                        if row.hidden_state_path:
                            layers = st.available_layers(row)
                            break
                else:
                    raise ValueError("Cannot determine layers. Set 'layers' in config.")
            else:
                raise ValueError("Cannot determine layers with ModelExtractor. Set 'layers' in config.")
        else:
            layers = list(config.layers)

    run_geometry_analysis(config, layers, metric=args.metric)


if __name__ == "__main__":
    main()
