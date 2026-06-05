# Atomic-Repair v1 — Known-Component Repair

v1 changes the question from v0. v0 asked "can a model repair an answer whose facts
it was never given and never knew?" (answer: no, by construction — `final_answer`≈0).
v1 asks the sharper question:

> A model may **know** two atomic facts yet fail to **chain** them.
> *"What language do people in Beijing speak?"* needs *Beijing→China* and
> *China→Mandarin*. If the model knows both but can't surface the bridge, it fails.

So v1 first **injects** every atomic fact, then tests whether the model can **use**
those known facts under **novel question/corruption wording**. The generalization
axis is **facts seen, repair forms unseen** — not entity-OOD.

## The causal chain (4 conditions)

| id | model | the question it answers |
|----|-------|--------------------------|
| **A** zero-shot (direct / CoT / Skill+CoT) | base Instruct | Can prompting alone repair? Does a skill prompt help with no training? |
| **B** Fact-only | Instruct + single-hop fact SFT | Does it now *know* the facts (gate)? Knowing facts, can it chain two hops? |
| **C** Fact→CoT | relay from B, CoT trajectory (no skill labels) | Does trajectory teach it to *use* known facts? |
| **D** Fact→Skill+CoT | relay from the **same** B, trajectory **with** skill labels | Does the skill label add over plain CoT? |

C and D start from the **same** Fact-only checkpoint and have **byte-identical
inputs** — only the output supervision differs, so C-vs-D isolates the skill label.

## Two gates (must pass before reading repair numbers)

1. **Cleanliness** — base/Instruct fact-QA accuracy must be ~0 (≤5%). Proves the
   synthetic facts are *not* in pretraining, so anything B learns was injected by us.
   (`run_v1_09_sanity_base.sh` → `--sanity` in the evaluator; leaked facts are listed.)
2. **Learned** — Fact-only fact-QA accuracy must be high (≥80%). If B didn't learn
   the facts, repair results are not interpretable.

## Data (all in `data_v1/`)

- `fact_train.jsonl` / `fact_eval.jsonl` — all **345** atomic facts, **disjoint
  phrasings** train vs eval (same facts, different question wording).
- `repair_train.jsonl` / `repair_eval.jsonl` — 5 cells (H-Aug, H-Abl, H-Cor, K-Cor,
  Clean), **shared entities**, **held-out forms**: question template, corruption
  wording (wrong-bridge / wrong-claim), and trace wording are all disjoint train/eval.

Guarantees enforced by `validate_v1.py` (hard, exit 1 on fail):
fact coverage (`repair gold triples ⊆ fact_train`), entity **overlap required**
(not disjoint), three form-id axes disjoint, fact phrasing disjoint, CoT==Skill+CoT
input, every eval item uses eval-side forms with facts ⊆ fact_train.

## Run order

**Local (this machine):**
```bash
bash scripts/run_v1_all.sh        # generate -> validate -> convert -> scriptcheck
                                  # then prints the server checklist
```
**Server (GPU box):** set `LF_DIR`, edit `model_name_or_path` in `configs/v1/*.yaml`
to your Qwen3-8B-Instruct, then run, in order:
`run_v1_09_sanity_base` → `run_v1_10_train_fact` →
`run_v1_11_train_fact_then_cot` & `run_v1_12_train_fact_then_skillcot` →
the `run_v1_2x_predict_*` scripts.

**Local scoring** (after pulling `output_v1/predict_*/generated_predictions.jsonl`):
```bash
bash scripts/run_v1_all.sh --score   # evaluate all + build comparison_v1.md
```

## Output

`data_v1/comparison_v1.md` — two-gate banner, final-answer accuracy by condition,
the four contrasts with bootstrap CIs on the deltas, and a failure-mode table
(Clean over-repair, H-Cor wrong-bridge accept, K-Cor wrong-claim accept).

## Design note

v1 does **not** preset "skill is useful." If Fact→Skill+CoT ≈ Fact→CoT, the gain is
from trajectory and skill is mainly for interpretability / future tool interfaces.
If Skill+CoT clearly wins, skill labels carry extra signal as a repair state/action
space. The data lets the result decide.

v0 files are untouched; v1 lives alongside (`*_v1.py`, `configs/v1/`, `data_v1/`,
`scripts/run_v1_*`).
