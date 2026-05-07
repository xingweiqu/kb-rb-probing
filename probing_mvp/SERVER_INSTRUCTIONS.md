# Server-side instructions

You are running probing on a GPU machine. Read this whole file before doing anything.

## Context (what changed since last run)

Three things since the previous Qwen3-8B summary:

1. **LoRA probe is much faster.** Previously each (layer, capability, fold) ran
   sequentially with constant tensor-creation overhead. The new ``ProbeBank``
   trains all transformer layers in parallel as one batched module — about
   30× faster on small datasets like ours.

2. **Per-token logits diagnostics.** ``extract_hidden_states.py`` now records
   for each (item, cot_state):
   - ``gold_per_token_logprob`` — list[float], one per gold sub-token
   - ``gold_token_strs`` — decoded gold tokens
   - ``gold_first_token_rank`` — vocab rank of the first gold token (0 = top-1)
   - ``top_k_tokens`` / ``top_k_logprobs`` — the top-5 competitor tokens at
     each gold position
   These let downstream analysis tell apart "model genuinely confident in
   gold" vs "gold rises in a flat distribution" — useful for explaining
   anomalies like RD-with_cot Δlogprob > 0.

3. **z-score judge.** ``derive_labels.py`` now emits a third capability label
   axis: ``zscore`` — Δlogprob z-normalized within each (capability,
   cot_state) group across families. Removes per-capability scale, so
   thresholds are comparable across the 2x3 grid. Default |z| > 1.0.

   Output schema is now ``{binary, delta, zscore, delta_value, zscore_value}``.

The summary roll-up also includes a new ``logits_diagnostics`` block:
top-1/top-5 match rate per (variant, cot_state), median gold rank, and
sample mismatched-but-high-lp families for manual inspection.

## What to run

Pull the latest ``feat/probing-pipeline`` branch:

```bash
git pull origin feat/probing-pipeline
```

### Parallel run across 8 GPUs (recommended)

The user has 8 GPUs. Use the dispatcher to run all models in parallel,
one model per GPU:

```bash
./probing_mvp/run_all_models.sh \
    /opt/tiger/ouro2/Qwen3-0.5B \
    /opt/tiger/ouro2/Qwen3-1.7B \
    /opt/tiger/ouro2/Qwen3-4B \
    /opt/tiger/ouro2/Qwen3-8B \
    /opt/tiger/ouro2/Qwen3-8B-Base
```

Each model gets pinned to one GPU via ``CUDA_VISIBLE_DEVICES``. Logs go
to ``logs/<model_name>.log``. The dispatcher waits for all jobs and prints a
final summary listing the ``summary.json`` files to push.

The dispatcher passes ``GRABGPU_ENABLE=0`` to each child because the
keepalive logic assumes single-script ownership of all 8 cards. **Start
GrabGPU manually before the dispatcher and kill it manually after, if you
need keepalive.**

### Sequential run (one model at a time)

If you only want to run one model:

```bash
./probing_mvp/run_probing.sh /opt/tiger/ouro2/Qwen3-8B
```

This produces ``runs/Qwen3-8B/summary.json``.

## Models to run, in priority order

The user is downloading these to ``/opt/tiger/ouro2/`` (or similar):

1. ``Qwen3-8B`` (instruct) — rerun since the schema is new
2. ``Qwen3-0.5B``
3. ``Qwen3-1.7B``
4. ``Qwen3-4B``
5. ``Qwen3-8B-Base`` (base, not instruct — for base vs instruct contrast)

Pick whichever are downloaded and run them in any order. Use the model
directory name as the run_name (the script auto-derives this).

## What to push back

For each model, commit and push only the summary file:

```bash
git add runs/<run_name>/summary.json
git commit -m "Add probing summary for <run_name>"
git push origin feat/probing-pipeline
```

Do **not** push:
- ``hidden_*.npy`` (each is hundreds of MB, .gitignored)
- ``model_outputs.jsonl`` (.gitignored)
- ``capability_labels.json`` / ``linear_probe.json`` / ``lora_probe.json`` /
  ``probe_geometry.json`` — these are local intermediate artifacts; the
  summary.json contains the rolled-up version of each (.gitignored)

The .gitignore already excludes those. Just push summary.json.

## What summary.json now contains

Top-level keys:

- ``model_path``, ``run_dir``, ``generated_at``
- ``task_family.linear`` and ``task_family.lora``: list of records, one per
  (cot_state, pool). Each has ``best_layer``, ``best_balanced_accuracy``,
  ``best_accuracy``, ``confusion_matrix``.
- ``capability.label_distribution``: per-capability per-judge per-cot_state
  ``(pos, neg, drop, total)``. Look here first to see if delta-judge actually
  produces non-zero pos/neg counts.
- ``capability.delta_value_stats``: per-capability per-cot_state stats of
  Δlogprob (mean, median, stdev, min, max). Useful even when the threshold
  produces all-pos or all-neg labels — the raw distribution is informative.
- ``capability.linear`` and ``capability.lora``: ``total / skipped / kept``
  counts plus ``top`` (best 10 capability probes by balanced accuracy).
- ``geometry.per_setting``: best silhouette per (cot_state, pool).
- ``geometry.cot_shift_last_layer``: per-class L2 displacement at the final
  layer between with-CoT and no-CoT representations.

## Sanity checks before pushing summary

After the run finishes, eyeball these in summary.json:

1. ``task_family.linear[0].best_balanced_accuracy`` should be ≥ 0.9 for any
   reasonably trained Qwen-family model. If it isn't, the extract step is
   broken.
2. ``capability.label_distribution.<any-capability>.delta.<cot>`` should
   show **non-zero pos+neg** for at least a few capabilities. If everything
   is still drop=75 even on the delta judge, the logprob extraction
   probably failed — check ``model_outputs.jsonl`` for ``gold_logprob_mean``
   field; it should be a real float, not null/NaN.
3. ``capability.delta_value_stats.<any-capability>.<cot>.stdev`` should be
   > 0. Zero stdev means logprobs are identical across variants, which
   means extraction is buggy.

If any of these fails, post the failure in the commit message rather than
"fixing" it silently — we want to know whether the issue is data, model,
or code.

## Optional knobs

```bash
# different cot or pooling
COT_STATES="no_cot" POOLS="last" ./probing_mvp/run_probing.sh /path/to/model

# different delta threshold (default 0.1 nats per token)
TAU=0.05 ./probing_mvp/run_probing.sh /path/to/model
# (requires manual passthrough; or run derive_labels manually with --tau)
```

## When in doubt

Push what you have plus a short comment in the commit message about what
looked wrong. We do the analysis offline.
