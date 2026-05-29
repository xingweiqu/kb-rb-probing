# Atomic-repair v0 — handoff

**Status (2026-05-29):** v0 shipped. Branch `atomic-repair-data-v0`, commit `73cafdf`. Validator PASS, 0 failures. Train 2,100 / eval 550. Server-side training scripts written but not run.

This file exists so a fresh Claude Code session can pick up the project without re-reading the full chat history. Long-lived design context is in the auto-memory note `project_atomic_repair_v0` (the chat agent loads it automatically); this file is the **today-list**.

---

## What's done

- [x] `generate_repair_data.py`, `validate_repair_data.py`, `convert_to_llamafactory.py`, README, configs, scripts.
- [x] Generated + validated `data/repair_raw_{train,eval}.jsonl` and `data/repair_lf_{train,eval}.json`.
- [x] Pushed `atomic-repair-data-v0` to origin.

## What's open (pick one to start the next session with)

1. **v0.1 family-level OOD.** Add a 7th relation family that lives only in eval. Currently 6 families appear in both splits — README §8 caveat. This is the next-biggest signal-vs-cost win.
2. **Server training.** Configs target Qwen3-8B-Base with LoRA r=16. Need to set `model_name_or_path` and run `scripts/run_03_train_llamafactory.sh` on a GPU box, then `run_04_predict_llamafactory.sh` on the held-out eval. Not greenlit yet.
3. **Per-cell difficulty / bootstrap CI report.** Cheap post-validation add; emits per-cell trace-length stats and a difficulty proxy. Useful before training so we know what "good" eval looks like.
4. **Per-capacity surface controls** (length / TF-IDF / random-label). Mirror of the paper appendix stub. Defends against "model just learns surface."
5. **Add R-cells.** Blocked on paper showing R-* perturbations have signal — currently they don't. Hold until that changes.

## How to resume

```bash
cd /Users/bytedance/Downloads/probing
git fetch origin atomic-repair-data-v0
git checkout atomic-repair-data-v0
claude
# first prompt: "read atomic_repair/HANDOFF.md and pick up item N"
```

The chat agent's auto-memory will load `project_atomic_repair_v0.md`, which contains the design contract, validator invariants, and the reasoning behind each open question. Don't relitigate the locked decisions (5 cells, no R-*, 6 families, 80/15/5 surface mix, `(problem, tentative)` pair-uniqueness) without explicit user OK.

## Don't

- Force-push v0. New work goes on a new branch.
- Add API calls or GPU dependencies to the local pipeline.
- Loosen any validator check without updating the matching contract in the README.
- Touch the paper repo from this branch — the paper lives in a separate clone.

---

For full schema, sample rows, and the why-LLM-Physics rationale, see `README_atomic_repair.md`.
