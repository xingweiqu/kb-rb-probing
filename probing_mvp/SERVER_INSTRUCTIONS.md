# Server-side instructions

You are running probing on a GPU machine. Read this whole file before doing anything.

## Context (what changed since last run)

The previous run on Qwen3-8B produced near-perfect task_family results (98.7%
balanced accuracy) but **all 18 capability probes were skipped** because the
binary correctness flag never flipped — Qwen3-8B is too strong for the data
and gets ~everything right under both the original and variant prompts.

To recover capability signal, the pipeline now records **gold-answer logprob**
at extract time and derives capability labels from **Δlogprob** (continuous,
sensitive when both items are correct). Binary judgments are still emitted
side-by-side as a sanity check.

There is also a new ``make_summary.py`` step. The whole run still ends with
one ``summary.json`` per run, which is what you should push back.

## What to run

Pull the latest ``feat/probing-pipeline`` branch and run as before. The shell
script handles the new logprob extraction and the new summary step
automatically:

```bash
git pull origin feat/probing-pipeline
./probing_mvp/run_probing.sh /opt/tiger/ouro2/Qwen3-8B
```

This produces ``runs/Qwen3-8B/summary.json``. Repeat per model.

## Models to run, in priority order

1. ``Qwen3-8B`` (rerun, since the schema is new)
2. ``Qwen3-1.5B`` (or ``Qwen3-1.7B`` if 1.5B is unavailable; whichever
   smaller Qwen3 you have locally) — important: smaller model is more
   likely to produce non-zero capability signal under either judge
3. ``Qwen3-8B-Base`` (base, not instruct) — to compare base vs instruct

If any of these aren't available locally, run whichever Qwen-family models
you do have. Skip non-Qwen for now.

Use the model directory name as the run_name. The shell script auto-derives
``runs/<basename>`` if you don't pass one.

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
