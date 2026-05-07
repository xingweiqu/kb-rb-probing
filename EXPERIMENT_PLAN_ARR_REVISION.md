# ARR Revision Experiment Plan

**Status:** drafted 2026-05-07. This document tracks what experiments are needed to move the paper from a technical-report-style draft to an ACL ARR-suitable diagnostic paper. **Do not invent results.** Each section lists what to run, where it runs (local vs. server), inputs, outputs, acceptance criteria, and what the paper update should say after the result is in hand. Sections marked `BLOCKED` cannot start until a dependency clears.

## Motivation

Reviewer-facing weaknesses in the current draft:
1. The paper claims a 3×9 diagnostic matrix but the ninth cell (Cross-Interference / CD) is reserved.
2. Only Qwen3 models are reported; the result may be Qwen-family-specific.
3. The dataset is synthetic and teacher-generated.
4. Probe results may be explainable by prompt-length or surface-template artefacts.
5. Per-cell sample sizes are small; no confidence intervals or significance tests.
6. No causal sanity check; mechanism language must remain qualified.

## Headline thesis (do not relitigate)

> Task-family labels (Knowledge / Reasoning / Hybrid) are observation-level. They do not identify why a model succeeds or fails. The diagnostic unit is an *intervention-defined response signature*: how the model's gold-answer probability changes when we add evidence, remove information, paraphrase, scaffold, or distract.

Title: **Task Labels Are Not Mechanisms: Intervention-Based Diagnostics for Language Models**.

The 3×9 matrix is a *visualization* of intervention signatures, not the headline contribution.

---

## Execution map (read this first)

| # | Part | Verdict | Owner | Blocking? |
|---|------|---------|-------|-----------|
| 1 | Generate CD / Wrong-Bridge variant + add to dataset + rerun extraction | SERVER + API | Server (new variant generator + hidden-state extraction) | Blocks 9-cell paper claim |
| 2 | Prompt-artefact + split controls (length / TF-IDF / random-label / random-feature) | LOCAL for length+TF-IDF+random-label; SERVER for random-feature | Mostly local | No |
| 3 | Bootstrap CIs and significance tests on Δlogprob signatures | SERVER (needs raw `model_outputs.jsonl`) | Server | No (degrade to point estimates if blocked) |
| 4 | Add one non-Qwen model family (Llama / Gemma / Mistral) | SERVER (model not on dev mac) | Server | No (paper can ship with footnote if missing) |
| 5 | Manual audit sheet for 90 sampled pairs | LOCAL | Dev mac builds sheet, user/RA fills it | No |
| 6 | External / human-written subset infra | LOCAL infra; data fill later | Dev mac builds template + loader | No |
| 7 | Causal sanity check (activation direction add at one layer) | SERVER (GPU forward hooks) | Server | No (appendix only) |
| 8 | Reorganize results / paper narrative | LOCAL once 1–7 land | Dev mac (subagent) | Final step |
| 9 | Deliverables file map | LOCAL | Dev mac | Now |

**Probe split safety**: a feasibility audit confirmed that the existing `linear_probe.py` / `lora_probe.py` use one labeled sample per backbone (one `original` row per family × 25 families). Item-level k-fold == backbone-level k-fold by construction. Symbolic-mode items are excluded from probe splits. **No leakage fix is needed**; the draft prose just needs a sentence stating this. (Marked done in PART 2 below.)

---

## PART 1 — Implement the ninth cell (CD / Wrong-Bridge Drop)

**Status:** BLOCKED on dataset generation.

### What CD is

Hybrid items require a bridge fact `B` and a rule `R` to produce gold answer `A`. The CD variant injects a *plausible but wrong* bridge fact `B'` while leaving `R` intact. Gold answer is unchanged at `A`. The wrong answer `A'` is what `B' + R` would produce. Diagnostic signal: does the model get pulled toward `A'`?

Example: original asks "What is the currency of the country where Mount Everest is located?" Bridge `B = Nepal`, rule `R = currency_of`, gold `A = Nepali rupee`. CD variant prepends "Mount Everest is located in Bhutan." (false). Wrong answer `A' = Bhutanese ngultrum`.

### Move both_blocked to auxiliary

`both_blocked` becomes the appendix lower-bound control. The main matrix cell at the (Hybrid, Distract) position is **Wrong-Bridge Drop (CD)**.

### Files to add / modify (server side)

| File | Change |
|---|---|
| `dataset_synthesis_mvp/config.py` | Replace `"both_blocked"` with `"wrong_bridge"` in `ATOMIC_VARIANTS["Hybrid"]`; add `wrong_bridge` to `VARIANT_DEFINITIONS` and `ATOMIC_CAPABILITY_MAP`. Keep `both_blocked` as `aux_double_block` for appendix only. |
| `dataset_synthesis_mvp/variants.py` | Add a `wrong_bridge` generator path that uses the structure's `nodes` to pick an alternative entity of the same `role=intermediate`, then asks Claude to phrase a one-sentence false bridge claim that uses the alternative. Must pass leakage validator (no gold answer leak) and a new `wrong_bridge_validator` that checks the alt-entity is different from the true bridge and same type. |
| `dataset_synthesis_mvp/structures.py` | When generating Hybrid structures, sample one alt-entity per backbone (same `intermediate` role; different label). Store as `alt_bridge_entity` field. |
| `probing_mvp/derive_labels.py` | Update VARIANT_TO_CAPABILITY: replace `("both_blocked", "CB-control", "block")` with `("wrong_bridge", "CD", "distract")`. Add a `_judge_*` path for distract on Composition (same logic as KD/RD). Keep `both_blocked` parsing path with new label `aux_double_block` for appendix. |
| `runs/full_25/output/dataset.jsonl` | Regenerate: keeps existing 8 variants × 25 backbones × 2 modes = 400 items; replaces 50 `both_blocked` items with 50 `wrong_bridge` items; appends 50 `both_blocked` items into a separate `aux/dataset_aux.jsonl` for the appendix. Total mainline = 600 (unchanged); aux = 50. |

### Server execution

1. `python -m dataset_synthesis_mvp.generate --output_dir runs/full_25_v2 --kb 25 --rb 25 --hybrid 25` after the variant code lands. ~20 min API call.
2. Smoke-test 5 hybrid items by hand from the output to confirm `wrong_bridge` items satisfy the spec.
3. Re-run hidden-state extraction on the 50 new wrong-bridge items only (don't redo the rest): `python -m probing_mvp.extract_hidden_states --resume_from runs/full_25_v2/output/dataset.jsonl ...` (the resume flag does not exist yet — add it, or just rerun the whole thing per model; ~10 min/model on H100).
4. `python -m probing_mvp.derive_labels` on the new outputs.
5. `python -m probing_mvp.make_summary --run_dir runs/<model>` per model. Push summary.json files back.

### Acceptance criteria

- `runs/full_25_v2/output/dataset.jsonl` has exactly 25 hybrid items per `wrong_bridge` mode (50 total inc. symbolic).
- `summary.json` shows `Wrong-Bridge Drop (CD)` non-null for at least one model with a Δlogprob magnitude visibly different from zero on Hybrid items.
- `ATOMIC_VARIANTS["Hybrid"]` no longer lists `both_blocked`.
- Manual audit (PART 5) includes 10 wrong-bridge pairs.

### Paper update

- Tables 1 and 2 captions and grid: rename CD from "reserved" to "Wrong-Bridge Drop"; mark `both_blocked` as `aux_double_block` (appendix).
- Abstract: change "operationalises eight of nine grid cells" to "operationalises all nine cells".
- §6.2 (cross-model) adds Wrong-Bridge Drop trends to the four headline trends.

---

## PART 2 — Prompt-artefact and split controls

**Status:** Length / TF-IDF / random-label baselines = LOCAL. Random-feature baseline = SERVER (needs raw hidden-state shape). Probe-split audit = already done (no leakage; just write it up).

### Files to add

| File | What it does | Where it runs |
|---|---|---|
| `scripts/run_prompt_artifact_controls.py` | For each (model, target ∈ {task_family, capability×judge}): train a logistic regression on (a) length features, (b) TF-IDF n-grams of prompt only, (c) random labels (shuffled within fold), (d) hidden state stays as-is from existing summary. Use `GroupKFold` on `family_id`. Output a long CSV with one row per (target, baseline, fold). | Local (length / TF-IDF / random-label). |
| `scripts/run_random_feature_baseline.py` | Replace hidden states with N(0, I) of matching dimension and rerun probes. Needs `hidden_*.npy` from server. | Server. |

### Length features (compute from `runs/full_25/output/dataset.jsonl`)

For each item: `prompt_token_len` (tiktoken or len(question.split())), `gold_token_len`, `digit_count`, `entity_count` (regex on capitalized words; cheap proxy), `punct_count`. Total 5 numeric features.

### TF-IDF baseline

Standard sklearn `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)` on prompt strings; logistic regression head. Same `GroupKFold(n_splits=5)` on `family_id`.

### Random-label control

Shuffle the labels within each fold's train set; retrain hidden-state probe; report. This is implemented by adding a `--shuffle_labels` flag to `linear_probe.py` and rerunning. Local-feasible if hidden states are local; otherwise server.

### Random-feature control

In `extract_hidden_states.py`-shaped output, replace `hidden_no_cot_last.npy` with `np.random.randn(N, n_layers, hidden_dim)`. Rerun `linear_probe.py`. Server-only since `.npy` files aren't synced.

### Outputs

- `reports/arr_revision/prompt_artifact_controls.csv` — one row per (model, target, baseline, fold)
- `reports/arr_revision/prompt_artifact_controls.md` — Markdown table with mean ± std comparing hidden-state probe vs. baselines
- Appendix table in paper

### Acceptance criteria

- Length-only and TF-IDF baseline accuracies for task_family three-way are well below the hidden-state probe (paper claim is that 0.987 is not a length/lexical artefact).
- Random-label and random-feature controls are at chance.
- Probe-split audit sentence appears in §5 (already drafted in current §5 Validation and Quality Control).

### Paper update

New §6.6 "Prompt-artefact and split controls". Reference the appendix table. Frame as robustness, not a contribution.

---

## PART 3 — Bootstrap CIs and significance tests

**Status:** SERVER (needs `model_outputs.jsonl` for Δlogprob per-item rows).

### What to compute

For each (model, family, cell, cot_state):
- Bootstrap by `family_id` (the backbone), not by item, to preserve pairing. 10000 resamples (or 1000 if compute is tight).
- Report: mean Δlogp/token, mean |Δlogp/token|, z-scored cell magnitude, 95% CI for each.

For each headline trend (paired-bootstrap two-sample test):
- Wrong-Claim Drop (KD): Qwen3-8B (instruct) vs Qwen3-8B-Base
- Bridge-Fact Gain (CP): Qwen3-1.7B-Base vs Qwen3-8B-Base vs Qwen3-8B
- Scaffold Gain (RP): Qwen3-8B vs Qwen3-8B-Base
- Retrieval-Block Drop (CB): Qwen3-8B vs Qwen3-8B-Base

### Files

| File | What |
|---|---|
| `scripts/bootstrap_ci.py` | Reads `model_outputs.jsonl` for each model; computes per-(family, variant, cot) Δlogprob; resamples by family_id; writes CIs and trend-test p-values. |
| `probing_mvp/make_summary.py` | Augment to optionally call into bootstrap_ci.py and embed CI fields into summary.json so plots can render error bars without re-running. |

### Outputs

- `reports/arr_revision/behavior_matrix_with_ci.csv`
- `reports/arr_revision/headline_trend_tests.csv`
- Updated heatmap with cell ± CI (or appendix CI table)

### Acceptance criteria

- Every cell-magnitude claim in §6.2 has either 95% CI or paired-bootstrap p < 0.05 support.
- Trends with overlapping CIs are described as "consistent with" rather than "shows that".

### Paper update

Update §6.2 and §6.5 prose to cite CIs. If a trend is no longer significant under bootstrap, remove or downgrade the claim.

---

## PART 4 — Cross-family check (non-Qwen)

**Status:** SERVER (model files not on dev mac).

### Target

At minimum: one non-Qwen instruct model at ~7–8B for behavior-only matrix.
Preferred: one base + one instruct from {Llama-3.1-8B / Llama-3.1-8B-Instruct, Mistral-7B-v0.3 / Mistral-7B-Instruct-v0.3, Gemma-2-9B / Gemma-2-9B-it}, depending on local availability on user's GPU box.

### Server execution

1. User downloads chosen model(s) to `/opt/tiger/...` or similar.
2. Codex runs `./probing_mvp/run_probing.sh /path/to/<non_qwen> <run_name>`. Uses existing pipeline; no code change needed.
3. After run completes, push `summary.json` to `runs/<run_name>/`.

### Files

| File | What |
|---|---|
| `scripts/run_non_qwen_models.py` | Optional convenience wrapper that just dispatches `run_probing.sh` for a list of non-Qwen models — most-likely never used (existing `run_all_models.sh` already does this). Skip unless asked. |

### Outputs

- `runs/<NonQwen_model>/summary.json`
- `reports/arr_revision/non_qwen_behavior_matrix.csv` — cells from new model
- `reports/arr_revision/non_qwen_ci.csv` — bootstrap CIs (depends on PART 3)
- Figure: `figures/_cross_model/fig_qwen_vs_nonqwen.png` — selected cells side-by-side

### Acceptance criteria

- At least one non-Qwen `summary.json` lands in `runs/`.
- The cross-family figure compares ≥4 cells: Wrong-Claim Drop, Scaffold Gain, Bridge-Fact Gain, Retrieval-Block Drop, Wrong-Bridge Drop (the latter requires PART 1 to land first).

### Paper update

New §6.4 "Cross-family check". Phrase as "These signatures are not unique to the Qwen3 family in this initial cross-family check"; do NOT claim general universality.

---

## PART 5 — Human audit / manual validation

**Status:** LOCAL infra; user / RA fills the sheet.

### Sampling

- 10 paired items per cell × 9 cells = 90 pairs.
- Half natural, half symbolic per cell.
- All 10 of the (Hybrid, CD) pairs included once PART 1 lands.
- All items the validator flagged borderline are also included.

### Files

| File | What |
|---|---|
| `scripts/create_manual_audit_sheet.py` | Pivots `runs/full_25/output/dataset.jsonl` to (orig, variant) pairs; samples 10 per cell with seed=42; writes a CSV with the schema specified in the user's spec. |
| `scripts/summarize_manual_audit.py` | Reads filled-in CSV; computes pass rate per cell, per family; computes inter-annotator agreement (Cohen's kappa or raw if only one annotator); reports failure-mode breakdown. |

### Audit sheet schema

```
item_id, backbone_id, family, cell_name, surface_mode,
original_prompt, variant_prompt, gold_answer, expected_wrong_answer,
validator_flags,
annotator_gold_same_answer,
annotator_perturbation_valid,
annotator_wrong_claim_plausible,
annotator_no_gold_leakage,
annotator_surface_artifact,
annotator_overall_pass,
comments
```

Annotator columns are blank in the generated sheet; user / RA fills them.

### Outputs

- `reports/arr_revision/manual_audit_sample.csv` (90 pairs, blank annotator columns)
- `reports/arr_revision/manual_audit_summary.csv` (after fill-in)
- `reports/arr_revision/manual_audit_summary.md`

### Acceptance criteria

- At least one annotator pass produces a populated `manual_audit_summary.csv`.
- Overall pass rate per cell is ≥ 80% (if not, the failures inform a targeted regeneration pass).

### Paper update

§6.7 (or appendix) "Human validation of intervention quality". Crucial because the dataset is synthetic.

---

## PART 6 — External / human-written subset (infra now, data later)

**Status:** LOCAL infra. Data is open-ended; do not block the paper on it.

### Files

| File | What |
|---|---|
| `data/external_subset_template.jsonl` | One example row per cell with the schema below; comment header explaining each field. |
| `scripts/eval_external_subset.py` | Reads `data/external_subset.jsonl` (if exists), runs the same Δlogprob diagnostic against any model with a `summary.json` in `runs/`, writes `reports/arr_revision/external_subset_results.csv`. |

### Schema

```
item_id, family, cell_name, original_prompt, variant_prompt,
gold_answer, expected_wrong_answer, source, notes
```

### Acceptance criteria

- Template + loader land before submission.
- If actual data exists: paper §6.7 reports results; if not: paper has a one-sentence release note ("we publish a template for external robustness data; populating it is a community task").

---

## PART 7 — Lightweight causal sanity check

**Status:** SERVER (GPU + new forward hooks). APPENDIX-LEVEL ONLY.

### Scope

Two cells only: **Wrong-Claim Drop (KD)** and **Scaffold Gain (RP)**. The two with the strongest scaling effects, so a sanity-check intervention is most likely to land.

### Method

For each cell:
1. Pick the best layer `L*` from the existing capability probe summary.
2. Compute direction `d = mean(h_pos) - mean(h_neg)` where positive = capability-active items, negative = capability-inactive items, all from the held-out fold (or use the linear probe weight vector).
3. During forward pass, add `α * d` at layer `L*` via a forward hook. Sweep `α ∈ {-2, -1, -0.5, 0.5, 1, 2}`.
4. Measure: log p(gold), margin log p(gold) − log p(wrong), and (if generation enabled) final correctness.

### Controls

- Random-direction baseline with same norm.
- Shuffled-label direction (computed on shuffled labels).
- Nearby non-best layer (`L* ± 2`).

### Files

| File | What |
|---|---|
| `scripts/causal_sanity_direction.py` | New — implements forward hook, direction computation, alpha sweep, all controls. Server-side only. |

### Outputs

- `reports/arr_revision/causal_sanity_wrong_claim.csv`
- `reports/arr_revision/causal_sanity_scaffold.csv`
- Appendix figure: α vs. log p(gold) for target direction, random direction, shuffled-label direction.

### Acceptance criteria

- Target direction's effect is monotonically increasing/decreasing in α and exceeds random-direction effect by visible margin.
- If not, language is downgraded to "we report the result without claiming the direction is mechanistically privileged".

### Paper update

Appendix-only subsection "Causal sanity check on intervention directions". Language must remain: "behaviorally relevant", "consistent with", NEVER "proves the mechanism".

---

## PART 8 — Reorganize results / paper narrative

**Status:** LOCAL after PARTS 1–7 land. Do this last.

### New §6 order

1. §6.1 Intervention signatures reveal distinct response profiles
2. §6.2 A completed 3×9 matrix: Wrong-Bridge Drop closes the Hybrid Distract cell  *(blocked on PART 1)*
3. §6.3 Scaling and instruction tuning shift specific signatures (with bootstrap CIs from PART 3)
4. §6.4 Cross-family check beyond Qwen *(blocked on PART 4)*
5. §6.5 Frozen hidden states encode intervention signatures
6. §6.6 Prompt-artefact and split controls *(PART 2)*
7. §6.7 Human validation and external robustness *(PARTS 5, 6)*
8. §6.8 Task-family read-out as a sanity check (the 0.987 number, demoted)

### Title (locked)

`Task Labels Are Not Mechanisms: Intervention-Based Diagnostics for Language Models`

### Naming (locked)

Human-readable in figures/tables: Hint Gain, Paraphrase Drop, Wrong-Claim Drop, Scaffold Gain, Rule-Removal Drop, Wrong-Step Drop, Bridge-Fact Gain, Retrieval-Block Drop, **Wrong-Bridge Drop** (replaces Cross-Interference / "reserved"). Short codes (KP, KB, KD, RP, RB, RD, CP, CB, CD) used in parentheses on first mention; retained in Tables 1 and 2.

### Banned phrases

- "minimal complete factorisation" → "compact operational intervention grid"
- "mechanism-level proof" → "mechanism-facing diagnostic signature"
- "recoverable mechanism" → "behavior-derived intervention profile"
- "we propose / we built / we made" → "we report / we construct / this paper"

---

## PART 9 — Deliverables file map

After all experiments land, the repo should contain:

```
EXPERIMENT_PLAN_ARR_REVISION.md            # this file
scripts/
  generate_cd_wrong_bridge.py              # PART 1 (server-runnable; calls into dataset_synthesis_mvp)
  run_behavior_matrix.py                   # PART 3 helper (build behavior matrix from model_outputs.jsonl)
  bootstrap_ci.py                          # PART 3
  run_prompt_artifact_controls.py          # PART 2 (length / TF-IDF / random-label)
  run_random_feature_baseline.py           # PART 2 (server)
  run_non_qwen_models.py                   # PART 4 (or skip if run_all_models.sh suffices)
  create_manual_audit_sheet.py             # PART 5
  summarize_manual_audit.py                # PART 5
  eval_external_subset.py                  # PART 6
  causal_sanity_direction.py               # PART 7 (server)
data/
  external_subset_template.jsonl           # PART 6
reports/arr_revision/
  prompt_artifact_controls.{csv,md}        # PART 2
  behavior_matrix_with_ci.csv              # PART 3
  headline_trend_tests.csv                 # PART 3
  non_qwen_behavior_matrix.csv             # PART 4
  non_qwen_ci.csv                          # PART 4
  manual_audit_sample.csv                  # PART 5
  manual_audit_summary.{csv,md}            # PART 5
  external_subset_results.csv              # PART 6 (when data exists)
  causal_sanity_wrong_claim.csv            # PART 7
  causal_sanity_scaffold.csv               # PART 7
figures/
  _cross_model/fig_qwen_vs_nonqwen.png     # PART 4
  appendix/fig_causal_sanity_alpha.png     # PART 7
```

---

## Suggested phasing (orchestrator)

**Phase 1 (now, dev mac, no GPU needed)**:
1. `create_manual_audit_sheet.py` (PART 5)
2. `external_subset_template.jsonl` + `eval_external_subset.py` (PART 6)
3. Length + TF-IDF + random-label baselines for current 5 models (PART 2 partial)
4. Probe-split audit one-paragraph note in paper §5 (PART 2 partial; doable now)

**Phase 2 (server, parallel)**:
1. Implement Wrong-Bridge variant generator + regenerate hybrid CD items (PART 1)
2. `bootstrap_ci.py` over existing `model_outputs.jsonl` (PART 3)
3. Random-feature baseline rerun (PART 2 remainder)

**Phase 3 (server, sequential after Phase 2)**:
1. Add non-Qwen model run (PART 4)
2. Causal sanity check on Wrong-Claim Drop and Scaffold Gain (PART 7)

**Phase 4 (after all data lands)**:
1. Manual audit (user/RA fills the sheet)
2. `summarize_manual_audit.py` reports
3. Subagent rewrites §6 in the new order, updates Tables 1+2 captions, updates Wrong-Bridge mentions

---

## What NOT to do

- Do not pre-write paper sections referencing experiment numbers that don't yet exist. Each `§6.X` paragraph waits until the corresponding experiment has produced output files in `reports/arr_revision/`.
- Do not invent CIs, p-values, or model accuracies.
- Do not strengthen claims past what the data supports.
- Do not skip controls because they are inconvenient.
- Do not regenerate the entire dataset from scratch when only the Hybrid Distract cell needs new items.
