# Probing Toolkit

Tests whether hidden representations are better organized by **coarse task labels** (KB / RB / Hybrid) or by **intervention-defined behavioral signatures** (premise-sensitive, scaffold-sensitive, etc.).

## Setup

```bash
pip install scikit-learn numpy torch transformers pyyaml
```

## Input Format

The toolkit reads a JSONL metadata file. Two formats are supported:

**Format 1 — items.jsonl** (from `generate_items.py`):
```json
{"id": "kb_001", "category": "KB", "gold_answer": "Paris", "variants": {"original": "...", "paraphrase": "...", ...}}
```
Set `field_map: {family_id: id, task_family: category}` in config.

**Format 2 — probe_v2.py output** (one row per variant):
```json
{"uid": "kb_001__original", "family_id": "kb_001", "task_family": "KB", "variant": "original", "score": 1.0, "hidden_state_path": "hidden_states/kb_001__original__layer16.pt"}
```

Hidden states are `.pt` (torch) or `.npy` (numpy) files, shape `[L, H]` or `[H]`.

## Configuration

Copy `probing/config_example.yaml` and edit paths and thresholds.

## Running Experiments

### Experiment A — Label probe baseline

Probes for KB / RB / Hybrid task labels.

```bash
python -m probing.cli.run_label_probe --config config.yaml
python -m probing.cli.run_label_probe --config config.yaml --layers 0 8 16 24 32
```

### Experiment B — Signature probe

Probes for behavioral signature labels derived from intervention scores.

```bash
# All 5 signature labels, absolute hidden state
python -m probing.cli.run_signature_probe --config config.yaml --target all

# One label, delta-state mode (h_variant - h_original)
python -m probing.cli.run_signature_probe --config config.yaml --target premise_sensitive --delta
```

Available targets: `premise_sensitive`, `scaffold_sensitive`, `removal_dependent`, `substitution_robust`, `wrong_claim_susceptible`.

### Experiment C — Cross-generalization probe

Transfer evaluation comparing label probe vs signature probe.

```bash
python -m probing.cli.run_transfer_probe --config config.yaml --setup kb_rb_to_hybrid
python -m probing.cli.run_transfer_probe --config config.yaml --setup all
python -m probing.cli.run_transfer_probe --config config.yaml --setup variant_transfer \
    --train-variants original paraphrase --test-variants counterfactual symbol_substitution
```

Available setups: `kb_rb_to_hybrid`, `natural_to_symbolic`, `variant_transfer`, `subfamily_transfer`.

### Geometry analysis

Pairwise cosine/euclidean distances between variant hidden states.

```bash
python -m probing.cli.run_geometry_analysis --config config.yaml --layer 16
python -m probing.cli.run_geometry_analysis --config config.yaml --layers 0 8 16 24 32 --metric cosine
python -m probing.cli.run_geometry_analysis --config config.yaml --all-layers
```

## Output Structure

```
probing_outputs/
  label_probe/task_family/{split_mode}/
    per_layer_metrics.csv
    confusion_matrix.csv
    summary.json
    seed_0/  metrics.json  predictions.csv  coefficients.csv  run_config.json
    seed_1/  ...
  signature_probe/
    behavioral_signals.json
    premise_sensitive__absolute/{split_mode}/  ...
    premise_sensitive__delta/{split_mode}/     ...
  transfer/
    kb_rb_to_hybrid/  transfer_comparison.csv  transfer_comparison.json
    natural_to_symbolic/  ...
  geometry/
    layer_16/  pairwise_distances.csv  distance_summary.csv
    layer_all/ pairwise_distances.csv  distance_summary.csv  distance_summary_overall.csv
```

## Behavioral Signature Labels

Labels are derived at the item-family level from per-variant scores:

| Label | Formula | Interpretation |
|---|---|---|
| `premise_sensitive` | `score(premise) - score(original) > τ_p` | Model benefits from explicit premises |
| `scaffold_sensitive` | `max(score(scaffold_k)) - score(original) > τ_s` | Model benefits from reasoning scaffolds |
| `removal_dependent` | `score(original) - score(premise_removal) > τ_r` | Model degrades when premises are removed |
| `substitution_robust` | `score(original) - score(structure_substitution) < τ_u` | Model is robust to surface token substitution |
| `wrong_claim_susceptible` | `score(original) - score(counterfactual) > τ_w` | Model is misled by injected wrong claims |
