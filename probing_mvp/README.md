# Probing pipeline

End-to-end pipeline for probing the 2x3 atomic capability taxonomy
(see `docs/taxonomy.md`) on a frozen causal LM.

## Quick start (server with GPU)

```bash
# 1. install deps
pip install -r requirements_mvp.txt

# 2. one-shot run with default settings (Qwen3-8B at the path you choose)
./probing_mvp/run_probing.sh /opt/tiger/Flame/Qwen3-8B

# results land in runs/Qwen3-8B/
```

The shell script chains four stages in order. Each is also a standalone module:

| Stage | Module | What it produces |
|---|---|---|
| 1 | `extract_hidden_states.py` | `hidden_<cot>_<pool>.npy`, `item_index.jsonl`, `model_outputs.jsonl` |
| 2 | `derive_labels.py` | `capability_labels.json` (per family x cot_state x capability) |
| 3 | `linear_probe.py` + `lora_probe.py` | `linear_probe.json`, `lora_probe.json` (per layer x target) |
| 4 | `geometry_analysis.py` | `probe_geometry.json` + `probe_geometry.md` |

Stage 1 needs a GPU. Stages 2–4 are CPU and fast.

## Knobs

Set environment variables before invoking the script:

```bash
DEVICE=cuda DTYPE=bfloat16 \
COT_STATES="no_cot with_cot" POOLS="last mean" \
./probing_mvp/run_probing.sh /opt/tiger/Flame/Qwen3-8B
```

To skip CoT entirely (drops second prompt pass): `COT_STATES=no_cot`.
To skip mean pooling: `POOLS=last`.

## Output files explained

- `hidden_<cot>_<pool>.npy` — `(N_items, n_layers+1, hidden_dim)` float16. Layer 0 is the embedding output; layers 1..n are transformer blocks.
- `item_index.jsonl` — `{row, family_id, variant, mode}` for each row of the `.npy`.
- `model_outputs.jsonl` — `{row, family_id, variant, mode, cot_state, prompt, generation, gold_answer, correct}`. `correct` is a soft check (gold substring in generation, case-insensitive); use it as a coarse signal, not as the ground truth answer label.
- `capability_labels.json` — `{family_id: {cot_state: {capability: True | False | null}}}`. Null means the judgment was uninformative for that family under that CoT state.
- `linear_probe.json` / `lora_probe.json` — per-layer `accuracy`, `balanced_accuracy`, `confusion_matrix` for both targets:
  - `task_family` — three-way KB/RB/Hybrid probe trained on `original` items
  - `capability` — one binary probe per (capability, cot_state)
- `probe_geometry.json` — centroid distances, silhouette, CKA, PCA-2D coordinates per (cot_state, pool); plus `cot_shift` showing per-class mean displacement between with-CoT and no-CoT representations.

## Re-runs and caching

Every stage overwrites its output file. To rerun only the probes after tweaking labels, point the modules directly at the existing `runs/<run_name>/` directory:

```bash
python -m probing_mvp.linear_probe \
  --hidden_dir runs/Qwen3-8B \
  --labels runs/Qwen3-8B/capability_labels.json \
  --output runs/Qwen3-8B/linear_probe.json
```

## Notes on the taxonomy

Capability labels follow the 2x3 mechanism × perturbation grid (see `docs/taxonomy.md`). One probe per (capability, cot_state) means up to `9 capabilities × 2 cot_states = 18` probes. Some cells will be `skipped` if too few non-null judgments survive. The `task_family` probe is the headline number; capability probes are the disaggregated story.
