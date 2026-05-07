# Wrong-Answer Scoring on Server: Qwen3-8B + Qwen3.5-9B only

For each Wrong-Claim / Wrong-Step / Wrong-Bridge item, score the
log-probability of the **planted-wrong answer** on the same prompt.
Combined with the existing per-item gold log-prob, this gives the four
margin quantities (`gold_drop`, `wrong_gain`, `margin_variant`,
`forced_choice_correct`) for §6.3.

Only two models are needed:
- **Qwen3-8B** (instruct)
- **Qwen3.5-9B** (instruct)

## Step 0: Sync code

```bash
cd ~/kb-rb-probing
git pull origin feat/probing-pipeline
```

Confirm `scripts/score_wrong_answer.py` is present.

## Step 1: Verify `model_outputs.jsonl` exists for both models

```bash
ls -la runs/Qwen3-8B/model_outputs.jsonl runs/Qwen3.5-9B/model_outputs.jsonl
```

If one is missing, run the full extraction first:

```bash
python -m probing_mvp.extract_hidden_states \
    --model_name /opt/tiger/Flame/Qwen3.5-9B \
    --dataset runs/full_25/output/dataset.jsonl \
    --output_dir runs/Qwen3.5-9B \
    --device cuda
```

(adjust `/opt/tiger/Flame/...` to whatever path your Qwen3.5-9B weights
live at; `ls /opt/tiger/` if unsure)

## Step 2: Score wrong-answer log-probs

The script reads the wrong-answer string straight from the existing
dataset metadata (`injected_premise` / `wrong_claim` /
`wrong_bridge_implied_answer`) by simple regex-style parsing. No API
call, no enrichment step. Rows where parsing fails or the parsed string
equals the gold answer are skipped (logged at startup).

```bash
# Qwen3-8B
python -m scripts.score_wrong_answer \
    --model_name /opt/tiger/Flame/Qwen3-8B \
    --dataset runs/full_25/output/dataset.jsonl \
    --model_outputs runs/Qwen3-8B/model_outputs.jsonl \
    --output runs/Qwen3-8B/model_outputs_with_wrong.jsonl \
    --device cuda

# Qwen3.5-9B
python -m scripts.score_wrong_answer \
    --model_name /opt/tiger/Flame/Qwen3.5-9B \
    --dataset runs/full_25/output/dataset.jsonl \
    --model_outputs runs/Qwen3.5-9B/model_outputs.jsonl \
    --output runs/Qwen3.5-9B/model_outputs_with_wrong.jsonl \
    --device cuda
```

Each model: ~150 wrong-answer scoring forward passes × 2 cot_states.
A few minutes each.

Expected log lines:
```
rows scoring wrong-answer: 252 / 600 (skipped: 44 unparseable, 4 match gold)
0/252 scored
50/252 scored
...
wrote runs/Qwen3-8B/model_outputs_with_wrong.jsonl
```

(Numbers are approximate; expect ~80% extraction success on natural-mode
wrongclaim/wrong_intermediate, plus all 50 wrong_bridge items per
cot_state.)

## Step 3: Sanity check + push

```bash
python -c "
import json
for m in ['Qwen3-8B', 'Qwen3.5-9B']:
    p = f'runs/{m}/model_outputs_with_wrong.jsonl'
    rows = [json.loads(l) for l in open(p)]
    n = sum(1 for r in rows if 'wrong_logprob_mean' in r)
    print(f'{m}: {len(rows)} rows, {n} have wrong_logprob_mean')
"

# Force-add in case .gitignore blocks model_outputs*.jsonl
git add -f runs/Qwen3-8B/model_outputs_with_wrong.jsonl \
           runs/Qwen3.5-9B/model_outputs_with_wrong.jsonl
git commit -m "Add wrong-answer log-prob scoring for Qwen3-8B and Qwen3.5-9B"
git push origin feat/probing-pipeline
```

After push, ping back. Local side runs `scripts/margin_analysis.py`,
fills in §6.3 numbers, and pushes the paper.
