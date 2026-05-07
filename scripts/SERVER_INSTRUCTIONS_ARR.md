# Server-side instructions: ARR revision Phase 2 + Phase 3

This file lists the experiments that must run on the GPU box for the ARR
revision. Phase 1 already shipped on the dev mac (see EXPERIMENT_PLAN_ARR_REVISION.md
top-level). Phase 2 (data + bootstrap) is required before the paper rewrite.
Phase 3 (non-Qwen + causal sanity) is appendix-quality and can lag.

## Phase 2A — Generate Wrong-Bridge Drop (CD) variant data

The dataset currently has 8 hybrid items per backbone (original +
explicit_fact + retrieval_blocked + both_blocked, two surface modes). We add
a ninth diagnostic cell, `wrong_bridge`, by appending a plausible-but-false
bridge claim to the original question.

```
git pull origin feat/probing-pipeline
python -m scripts.generate_cd_wrong_bridge \
    --structures runs/full_25/checkpoints/01_structures.json \
    --base_items runs/full_25/checkpoints/02_base_items.json \
    --existing_dataset runs/full_25/output/dataset.jsonl \
    --output_jsonl runs/full_25/output/dataset_with_cd.jsonl
```

Expected wall time: ~4 minutes (25 backbones, concurrency=8, ridgerzhu
Sonnet API). Output: `dataset_with_cd.jsonl` with 600 + 50 = 650 items.

After it finishes, **replace the canonical dataset path** so downstream
extraction picks it up:

```
mv runs/full_25/output/dataset.jsonl runs/full_25/output/dataset_pre_cd.jsonl
mv runs/full_25/output/dataset_with_cd.jsonl runs/full_25/output/dataset.jsonl
git add runs/full_25/output/dataset.jsonl
git commit -m "Add Wrong-Bridge Drop (CD) variant: 50 hybrid items"
git push origin feat/probing-pipeline
```

## Phase 2B — Re-extract hidden states for the new wrong_bridge items only

`extract_hidden_states.py` does not have a `--resume` flag yet. Two options:

**Option 1: rerun the whole pipeline per model (clean, ~10 min/model on H100)**

```
./probing_mvp/run_all_models.sh \
    /opt/tiger/ouro2/Qwen3-0.6B \
    /opt/tiger/ouro2/Qwen3-0.6B-Base \
    /opt/tiger/ouro2/Qwen3-1.7B \
    /opt/tiger/ouro2/Qwen3-1.7B-Base \
    /opt/tiger/ouro2/Qwen3-4B-Base \
    /opt/tiger/ouro2/Qwen3-8B \
    /opt/tiger/ouro2/Qwen3-8B-Base \
    /opt/tiger/ouro2/Qwen3.5-0.8B \
    /opt/tiger/ouro2/Qwen3.5-2B \
    /opt/tiger/ouro2/Qwen3.5-2B-Base \
    /opt/tiger/ouro2/Qwen3.5-4B \
    /opt/tiger/ouro2/Qwen3.5-4B-Base \
    /opt/tiger/ouro2/Qwen3.5-9B \
    /opt/tiger/ouro2/Qwen3.5-9B-Base \
    /opt/tiger/ouro2/Qwen3.5-35B-A3B \
    /opt/tiger/ouro2/Qwen3.5-35B-A3B-Base
```

The dispatcher already pins one model per GPU. With 8 GPUs this batch is
~2 hours wall time.

**Option 2: only extract the new 50 wrong_bridge items per model**

Quicker, but requires a small patch to `extract_hidden_states.py` to filter
the dataset by variant. Skip this option unless time is tight.

After all runs finish, push only the per-model `summary.json` files:

```
git add runs/Qwen3-*/summary.json runs/Qwen3.5-*/summary.json
git commit -m "Refresh summaries with Wrong-Bridge Drop (CD) cell"
git push origin feat/probing-pipeline
```

## Phase 2C — Bootstrap CI on existing model_outputs.jsonl

This is the most data-cheap step but requires `model_outputs.jsonl` files
which are NOT pushed to GitHub (.gitignore excludes them). Run locally on
the server. The script reads each `runs/<model>/model_outputs.jsonl` and
writes CSVs.

```
python -m scripts.bootstrap_ci \
    --runs runs/Qwen3-0.6B runs/Qwen3-0.6B-Base \
           runs/Qwen3-1.7B runs/Qwen3-1.7B-Base runs/Qwen3-4B-Base \
           runs/Qwen3-8B runs/Qwen3-8B-Base \
           runs/Qwen3.5-0.8B runs/Qwen3.5-2B runs/Qwen3.5-2B-Base \
           runs/Qwen3.5-4B runs/Qwen3.5-4B-Base \
           runs/Qwen3.5-9B runs/Qwen3.5-9B-Base \
           runs/Qwen3.5-35B-A3B runs/Qwen3.5-35B-A3B-Base \
    --bootstrap 10000
```

Then push the two CSVs (they're small):

```
git add reports/arr_revision/behavior_matrix_with_ci.csv \
        reports/arr_revision/headline_trend_tests.csv
git commit -m "Bootstrap CIs for intervention signature matrix"
git push origin feat/probing-pipeline
```

Wall time: ~2 minutes.

## Phase 3A — One non-Qwen model family (Llama-3.1-8B + Instruct)

If Llama-3.1-8B and Llama-3.1-8B-Instruct (or equivalent local paths) are
on the box, run the existing pipeline:

```
./probing_mvp/run_all_models.sh \
    /opt/tiger/ouro2/Llama-3.1-8B \
    /opt/tiger/ouro2/Llama-3.1-8B-Instruct
```

Then push their summary.json. This unblocks the §6.4 "Cross-family check"
subsection of the paper rewrite.

## Phase 3B — Causal sanity check (appendix)

`scripts/causal_sanity_direction.py` is NOT yet written; it is the next
script the dev mac will produce. Skip Phase 3B until it lands. The plan
calls for activation-direction add at one layer for two cells only
(Wrong-Claim Drop and Scaffold Gain) with random-direction and
shuffled-label controls.

## Sanity checks before pushing summary

- summary.task_family.linear[0].best_balanced_accuracy >= 0.9 on instruct
  models.
- summary.capability.label_distribution has a `wrong_bridge` entry on
  hybrid items (after Phase 2B).
- summary.logits_diagnostics.hint.no_cot.top1_match_rate is a real float
  (confirms per-token logprobs were captured).
- bootstrap_ci.py CSVs have non-empty rows for each (model, capability,
  cot) cell.

If any check fails, include the failure in the commit message rather than
silently fixing.
