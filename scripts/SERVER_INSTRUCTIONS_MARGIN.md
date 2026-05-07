# Wrong-Answer Scoring on Server: Qwen3-8B + Qwen3.5-9B only

Goal: produce per-item log-probabilities of the **planted-wrong answer** (in
addition to the existing per-item gold log-probabilities) so the local
margin analysis can compute `gold_drop`, `wrong_gain`, `margin_variant`,
and `forced_choice_correct` for the Wrong-Claim, Wrong-Step, and
Wrong-Bridge cells (KD / RD / CD).

We only need this for the two flagship models:
- **Qwen3-8B** (instruct)
- **Qwen3.5-9B** (instruct)

## Step 0: Pull latest code (LOCAL → push, then on SERVER pull)

The two new scripts live on `feat/probing-pipeline`:

```bash
# LOCAL
cd /Users/bytedance/Downloads/probing
git add scripts/enrich_wrong_implied.py scripts/score_wrong_answer.py \
        scripts/margin_analysis.py scripts/SERVER_INSTRUCTIONS_MARGIN.md
git commit -m "Add wrong-answer scoring + margin analysis pipeline"
git push origin feat/probing-pipeline
```

```bash
# SERVER
cd ~/kb-rb-probing  # or wherever the repo lives
git pull origin feat/probing-pipeline
```

## Step 1: Annotate wrong-implied answers (LOCAL, API only — no GPU)

Adds `metadata.wrong_implied_answer` to `wrongclaim` and `wrong_intermediate`
rows. The `wrong_bridge` rows already have it.

```bash
# LOCAL (or anywhere with API access; ~3 minutes for 100 rows)
python -m scripts.enrich_wrong_implied \
    --input runs/full_25/output/dataset.jsonl \
    --output runs/full_25/output/dataset_with_wrong_implied.jsonl
```

Then push the enriched dataset:

```bash
git add runs/full_25/output/dataset_with_wrong_implied.jsonl
git commit -m "Enrich dataset with wrong_implied_answer for KD/RD"
git push origin feat/probing-pipeline
```

## Step 2: Score the wrong answer per item (SERVER, GPU)

For each of the two models, the server already has:
- `model_outputs.jsonl`: per-item gold log-prob, generation, prompt, etc.

We need to load the model, score the planted-wrong answer's log-prob on the
same prompt, and append `wrong_logprob_*` columns. This is fast (≈150
rows × 2 cot_states ≈ a few minutes per model).

```bash
# SERVER (assumes /opt/tiger/ouro2/<model> layout)
cd ~/kb-rb-probing
git pull origin feat/probing-pipeline   # picks up dataset + scripts

# Qwen3-8B (instruct)
python -m scripts.score_wrong_answer \
    --model_name /opt/tiger/ouro2/Qwen3-8B \
    --dataset runs/full_25/output/dataset_with_wrong_implied.jsonl \
    --model_outputs runs/Qwen3-8B/model_outputs.jsonl \
    --output runs/Qwen3-8B/model_outputs_with_wrong.jsonl \
    --device cuda

# Qwen3.5-9B (instruct)
python -m scripts.score_wrong_answer \
    --model_name /opt/tiger/ouro2/Qwen3.5-9B \
    --dataset runs/full_25/output/dataset_with_wrong_implied.jsonl \
    --model_outputs runs/Qwen3.5-9B/model_outputs.jsonl \
    --output runs/Qwen3.5-9B/model_outputs_with_wrong.jsonl \
    --device cuda
```

If `runs/Qwen3.5-9B/model_outputs.jsonl` doesn't exist on the server (the
9B model may not have been part of the last full extraction), run the
full extraction first, then this script:

```bash
# Optional pre-step if model_outputs.jsonl is missing for Qwen3.5-9B
./probing_mvp/run_probing.sh /opt/tiger/ouro2/Qwen3.5-9B
```

After both finish, push the new files:

```bash
git add runs/Qwen3-8B/model_outputs_with_wrong.jsonl \
        runs/Qwen3.5-9B/model_outputs_with_wrong.jsonl
git commit -m "Add wrong-answer log-prob scoring for Qwen3-8B and Qwen3.5-9B"
git push origin feat/probing-pipeline
```

(JSONLs are big-ish — ~10MB each. If `.gitignore` blocks `model_outputs*`,
override with `git add -f`.)

## Step 3: Margin analysis (LOCAL, no GPU)

```bash
# LOCAL
git pull origin feat/probing-pipeline   # picks up the two JSONLs

python -m scripts.margin_analysis \
    --runs runs/Qwen3-8B runs/Qwen3.5-9B \
    --output_dir reports/margin \
    --figure_dir .claude/worktrees/agent-af40cbcbaeed15410/paper/figures
```

Outputs:
- `reports/margin/family_vs_capacity.csv`
- `reports/margin/wrong_claim_margin.csv`
- `reports/margin/all_capacity_rows.csv`
- `paper/figures/fig_family_vs_capacity.png`
- `paper/figures/fig_wrong_claim_margin.png`

## Step 4: Plug numbers into paper (LOCAL)

Once `wrong_claim_margin.csv` lands, the `\todo{...}` markers in
`paper/sections/01_intro.tex` and `paper/sections/06_results.tex` get
filled in with real per-model `gold_drop`, `wrong_gain`, `margin_variant`,
and `forced_choice_correct` values, and the new figures appear inline.

Then push to `xingweiqu/atomic_capacity` main; pull on Overleaf.

## Notes

- The patch only re-uses `_gold_logprob` from `probing_mvp/extract_hidden_states.py`; nothing
  else in the extraction pipeline changes.
- Memory footprint is identical to the original extraction (one model on
  device); compute is much smaller (only 150 wrong-answer scoring forward
  passes vs. 600 hidden-state extractions).
- If you'd rather run on more models later, just append more
  `--runs runs/<model>` entries; the script is a no-op on rows that don't
  have a `wrong_implied_answer`.
