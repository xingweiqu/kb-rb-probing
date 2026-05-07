# Atomic-Capacity Revision Plan

## 1. Paper thesis

**Atomic capacities are the diagnostic unit.** Task-family labels (Knowledge / Reasoning / Hybrid) are observation-level; they describe the *kind* of question, not the capacity that failed. **Interventions are assays, not the object of study.** A single task-family failure dissolves into several atomic-capacity failures, which imply different model-improvement strategies.

This revision recasts the paper so that *atomic capacity* is the headline noun. *Intervention signature* becomes a method-level term used only in §3 and §6 prose, never in the title, abstract, or section headers.

## 2. The nine atomic capacities

Three-level hierarchy:

- **Level 1 — Task domain**: Knowledge / Reasoning / Bridge (the question type).
- **Level 2 — Exposure mode (assay type)**: Provide / Block / Distract (the controlled operation we apply to the prompt).
- **Level 3 — Atomic capacity**: the unique combination of domain × exposure-mode (nine cells).

Capacity names, assays, and required validation:

| Code | Domain | Exposure | Capacity (Level 3) | Assay variant | Expected failure mode | Metric |
|---|---|---|---|---|---|---|
| KP | Knowledge | Provide | **Knowledge Surfacing** | hint | model cannot use a brief cue to surface stored knowledge | Δlogp, margin |
| KB | Knowledge | Block | **Paraphrase Robustness** | paraphrase | model is fragile to surface rewording of the same query | Δlogp, margin |
| KD | Knowledge | Distract | **Wrong-Claim Robustness** | wrongclaim | model is overridden by a planted contradicting claim | Δlogp, margin |
| RP | Reasoning | Provide | **Scaffold Activation** | scaffold | model cannot leverage a step decomposition that is offered to it | Δlogp, margin |
| RB | Reasoning | Block | **Rule Grounding** (a.k.a. Rule Dependence) | rule_removal | model cannot recover when a rule is stripped from the prompt | Δlogp, margin |
| RD | Reasoning | Distract | **Intermediate-Step Robustness** | wrong_intermediate | model is derailed by a planted wrong intermediate value | Δlogp, margin |
| CP | Bridge | Provide | **Bridge-Fact Integration** | explicit_fact | model cannot integrate an offered bridge fact into composition | Δlogp, margin |
| CB | Bridge | Block | **Retrieval-Block Compensation** | retrieval_blocked | model cannot fill a missing bridge by reasoning from clues | Δlogp, margin |
| CD | Bridge | Distract | **Wrong-Bridge Robustness** (a.k.a. Cross-Axis Interference) | wrong_bridge | model adopts a planted wrong bridge and produces the wrong composed answer | Δlogp, margin |

Required validation, per capacity:
- gold-answer leakage check (every assay variant);
- answer preservation: gold answer is unchanged across original / variant pair (every assay);
- type-matched distractors (KD, RD, CD only — same answer type as gold);
- structural removal (RB, CB only — the rule or bridge is actually removed, not decoratively);
- family-level k-fold split for any read-out (every probe);
- z-normalisation within (capacity, cot_state) so binary correctness collapse on strong models is handled.

## 3. Missing experiments and TODOs

| TODO | Status | Owner | Notes |
|---|---|---|---|
| **CD completion in dataset** | DONE locally; awaiting server re-extraction | dev mac (data) + server (hidden states) | 50 wrong_bridge items already in `runs/full_25/output/dataset.jsonl` (commit `85e09dd`); 22 of 25 natural items satisfy `wrong_implied_answer ≠ gold`. Server must rerun `run_all_models.sh` so summary.json picks up the CD cell. |
| **Both-blocked demoted to appendix** | DONE in code; pending paper prose | paper rewrite | The wrong_bridge variant occupies the CD cell. `both_blocked` items remain in dataset_pre_cd.jsonl and are referenced as the appendix lower-bound control. |
| **Bootstrap confidence intervals** | INFRA DONE; awaiting server execution | server | `scripts/bootstrap_ci.py` reads `model_outputs.jsonl` and resamples by `family_id`. Pre-registered tests: KD instruct vs base (8B), CP scaling (1.7B-Base→8B), RP instruct vs base (8B), CB instruct vs base (8B), and CD once data lands. |
| **Prompt-artefact controls** | DONE for length / TF-IDF / random-label; random-feature pending server | partly local, partly server | `scripts/run_prompt_artifact_controls.py` shipped Phase 1 results (TF-IDF=0.914, length=0.864, random-label=0.229, random-feature=0.247 vs hidden-state probe=0.987). The TF-IDF baseline is *high* — the paper acknowledges that the three-way task-family read-out is not a clean representation finding and lives in §6.7 as a sanity check. |
| **Grouped backbone split** | DONE | dev mac | `scripts/run_prompt_artifact_controls.py` uses `GroupKFold` on `family_id`. Existing capability probes train one labeled sample per backbone, so item-level == backbone-level by construction; symbolic items are excluded from probe splits. Paper §5 documents this. |
| **Human audit** | INFRA DONE; awaiting human pass | dev mac infra; user / RA fills | `scripts/create_manual_audit_sheet.py` produces `reports/arr_revision/manual_audit_sample.csv` (10 pairs × 9 capacities). Annotator columns are blank pending fill-in. `scripts/summarize_manual_audit.py` reports pass rate by capacity (not just overall). The CD column is real after the local CD generation; previous AWAITING_GENERATION placeholders are replaced. |
| **Non-Qwen model family** | PENDING | server (Llama-3.1-8B base + instruct preferred) | Run `./probing_mvp/run_all_models.sh` on the Llama paths if locally available. Behaviour-only matrix is sufficient if hidden-state extraction is too expensive on the larger model. |
| **External / human-written subset** | INFRA DONE; data fill optional | dev mac infra | `data/external_subset_template.jsonl` and `scripts/eval_external_subset.py` ready. The paper notes this as a release artefact; results section conditional on data being filled before submission. |
| **Causal sanity check (optional)** | NOT STARTED | server (forward hooks) | Two capacities only: Wrong-Claim Robustness and Scaffold Activation. Mean-difference direction at the best probe layer; α-sweep with random-direction and shuffled-label controls. Appendix-only; language stays "behavioural relevance sanity check". |

## 4. Acceptance criteria

The revision is acceptable if and only if:

- All nine capacity cells have real generated data and real per-cell results in summary.json.
- `both_blocked` appears only as an appendix lower-bound control; no main matrix or main figure shows it as a tenth main cell.
- Every main capacity claim in the paper is paired with a 95% bootstrap CI (`reports/arr_revision/behavior_matrix_with_ci.csv` populated).
- All capacity probes use grouped backbone splits (already true; documented in §5 prose).
- Prompt-artefact controls (length-only, TF-IDF, random-label, random-feature) are reported in §6.6 with the existing numbers from Phase 1.
- Human audit pass rate is reported **per capacity**, not only overall, in §6.7. The current sheet has 90 rows (10 per capacity, including 10 real CD pairs).
- Non-Qwen model family is either reported in §6.4 or explicitly listed as future work (do not fake numbers).
- The paper's title, abstract, §3 header, and §6 header use "atomic capacit…" as the headline noun. "Intervention signature" appears only as a method-level term in §3 and §6 prose, never in titles or the abstract.
- No paragraph contains the phrase "minimal complete factorisation". Capacity claims are explicitly framed as "compact operational diagnostic grid", not exhaustive.
- Causal sanity check, if included, sits in the appendix and uses "behavioural relevance" / "consistent with" / "supports" language only — never "causal proof".

## 5. Renaming map (paper-wide)

To be applied by the rewrite pass:

| Old | New |
|---|---|
| Title: "...Intervention-Based Diagnostics for Language Models" | "Atomic Capacities, Not Task Labels: Diagnosing Language Model Failures" |
| Abstract / intro: "intervention signature" as the headline noun | "atomic capacity" as the headline noun |
| §3 header "Interventions and signatures" | "Atomic Capacities and Capacity Assays" |
| §6.1 header "Intervention signatures reveal distinct response profiles" | "Behavioural capacity profiles across the diagnostic grid" |
| §6.5 header "Frozen hidden states encode intervention signatures" | "Atomic-capacity decodability from frozen hidden states" |
| Cell labels in figure axes (e.g. "Hint Gain (KP, knowledge augmentation use)") | "Hint Gain (KP, Knowledge Surfacing)" — keep the readable Δlogp name as the axis label, the noun phrase in parentheses |
| "capacity-named perturbation modes" | "exposure-mode assay types (Provide / Block / Distract)" |
| "minimal complete factorisation" / "complete decomposition" | "compact operational diagnostic grid" |

## 6. Out of scope for this revision

- Re-generating the entire dataset (only 50 wrong_bridge items added; rest unchanged).
- Re-training probes (server reruns hidden-state extraction so the new `wrong_bridge` items show up; no probe-training change).
- Adding a tenth or eleventh capacity. The grid stays at nine.
- Cross-architecture coverage beyond one non-Qwen family. We do not promise universality.
