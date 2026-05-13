# Atomic-Capacity Figure 2: Cross-Model Replication

One-click pipeline to reproduce **Figure 2** of *Atomic Capacities, Not
Task Labels: Diagnosing Language Model Failures* on any HuggingFace
causal LM.

Figure 2 is the 3 × 9 atomic-capacity matrix: rows are task families
(Knowledge / Reasoning / Bridge), columns are atomic capacities grouped
by exposure mode (Provide / Block / Distract). Each cell is the
`|mean Δlog p / token|` between the original item and its capacity
assay, over 25 backbones per family. All nine cells are populated.

## Quick start

```bash
git clone -b figure2-replication git@github.com:xingweiqu/kb-rb-probing.git
cd kb-rb-probing

pip install torch transformers numpy matplotlib

# Local weights:
./run.sh /path/to/Qwen3-8B

# Or HuggingFace ID (auto-downloads):
./run.sh Qwen/Qwen3-8B
```

Outputs:
```
runs/<model_name>/summary.json
figures/<model_name>/fig_capacity_matrix.png
```

## What it runs

1. `probing_mvp/extract_hidden_states.py` — loads the model once, runs a
   forward pass on every item in the 650-item dataset under both no-CoT
   and zero-shot CoT conditions, records per-item gold-answer
   per-token log-probabilities and per-layer hidden states.
2. `probing_mvp/make_summary.py` — rolls up per-cell mean / median /
   stdev / min / max of Δlog p (variant − original) into
   `summary.json`.
3. `probing_mvp/make_plots.py` — renders the 3 × 9 capacity matrix.

## Dataset

`runs/full_25/output/dataset.jsonl` — 650 items:
- 75 backbones (25 Knowledge + 25 Reasoning + 25 Bridge)
- per backbone: 1 original + 8 paired capacity assays + 1 lower-bound
  control (`both_blocked`, appendix-only)
- ×2 surface forms (natural + symbolic)
- = 600 from the initial release + 50 wrong-bridge items added later
  to populate the CD cell

| Variant | Cell | Family |
|---|---|---|
| `hint` | KP — Knowledge Surfacing | KB |
| `paraphrase` | KB — Paraphrase Robustness | KB |
| `wrongclaim` | KD — Wrong-Claim Robustness | KB |
| `scaffold` | RP — Scaffold Activation | RB |
| `rule_removal` | RB — Rule Grounding | RB |
| `wrong_intermediate` | RD — Intermediate-Step Robustness | RB |
| `explicit_fact` | CP — Bridge-Fact Integration | Hybrid |
| `retrieval_blocked` | CB — Retrieval-Block Compensation | Hybrid |
| `wrong_bridge` | CD — Wrong-Bridge Robustness | Hybrid |
| `both_blocked` | (control) | Hybrid |

## Hardware

Single GPU. Roughly:
- 0.6B: ~2 min
- 8B: ~5 min
- 35B-A3B: ~15 min

H100 / A100 recommended; smaller models fit on a 24 GB consumer card.

## Files in this branch

```
run.sh                                # one-click entry point
probing_mvp/
  extract_hidden_states.py            # forward pass + logp extraction
  make_summary.py                     # aggregate to summary.json
  make_plots.py                       # render fig_capacity_matrix.png
  derive_labels.py                    # capacity / family labels
runs/full_25/output/dataset.jsonl     # the 650-item dataset
```

This branch is intentionally trimmed. The full pipeline (probes,
bootstrap CIs, margin decomposition, paper figures) lives on
`feat/probing-pipeline`.

## Citation

```bibtex
@inproceedings{atomic_capacity,
  title={Atomic Capacities, Not Task Labels: Diagnosing Language Model Failures},
  author={Qu, Xingwei},
  year={2026}
}
```
