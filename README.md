# Atomic Capacities, Not Task Labels

Code for the paper *Atomic Capacities, Not Task Labels: Diagnosing Language
Model Failures*.

We argue that the actionable diagnostic unit for LLM failure is the
**atomic capacity** — small enough that a paired assay isolates it,
specific enough that its failure names a distinct mode, repair-relevant
enough that the failure points to a different intervention than its
neighbours. We instantiate nine atomic capacities on a 3 × 3 grid
(Knowledge / Reasoning / Bridge × Provide / Block / Distract), one
paired capacity assay per cell, and a behaviour-derived metric
(gold-answer Δlog p / token).

Across 17 Qwen3 / Qwen3.5 models, capacity profiles scale heterogeneously
and family-level improvement is not capacity-level improvement. Wrong-
Claim Robustness on the Qwen3 instruct series shows that gold log-prob
drop and gold-vs-wrong margin grow together: confidence sensitivity
and decision robustness rise in lockstep.

## Repository layout

```
probing/
├── dataset_synthesis_mvp/      # 9-variant capacity-assay dataset generator
│   ├── generate.py             # main entry point
│   ├── structures.py base_items.py variants.py symbolic.py
│   ├── validation/             # leakage, type-match, answer-preservation
│   └── repair/                 # automated fix passes
├── probing_mvp/
│   ├── extract_hidden_states.py  # per-item logits + per-layer hidden states
│   ├── derive_labels.py          # capacity + task-family labels
│   ├── linear_probe.py           # logistic regression read-out
│   ├── lora_probe.py             # LoRA-r16 read-out
│   ├── geometry_analysis.py      # silhouette + clustering
│   ├── make_summary.py           # per-model summary.json
│   └── make_plots.py             # capacity matrix + scaling figures
├── scripts/
│   ├── score_wrong_answer.py     # plant wrong-answer log-prob (KD/RD/CD)
│   ├── margin_analysis.py        # gold_drop, margin_variant, forced-choice
│   ├── bootstrap_ci.py           # paired-bootstrap CIs by family_id
│   └── run_prompt_artifact_controls.py
├── runs/<model>/                 # per-model outputs (summary.json etc.)
├── reports/                      # CSVs of per-cell magnitudes + CIs
└── figures/                      # rendered plots
```

## Setup

```bash
pip install torch transformers numpy matplotlib scikit-learn
# For dataset generation only:
export ANTHROPIC_BASE_URL=...    # your provider base URL
export ANTHROPIC_AUTH_TOKEN=...  # your API token
```

## Pipeline

### 1. Generate the capacity assay dataset (one time)

```bash
python -m dataset_synthesis_mvp.generate \
    --kb 25 --rb 25 --hybrid 25 \
    --output_dir runs/full_25/output
```

Produces a 650-item dataset: 75 backbones × (1 original + 8 variants) ×
(natural + symbolic). Each variant is the paired assay for one of the
nine atomic capacities (KP / KB / KD / RP / RB / RD / CP / CB / CD).

### 2. Extract per-item logits + hidden states (per model, GPU)

```bash
python -m probing_mvp.extract_hidden_states \
    --model_name /path/to/Qwen3-8B \
    --dataset runs/full_25/output/dataset.jsonl \
    --output_dir runs/Qwen3-8B \
    --device cuda
```

Records per (item, CoT) the gold-answer per-token log-prob, the
last-token and mean-pool hidden states for every transformer layer, and
the top-5 competitors at the gold position.

### 3. Score the planted-wrong answer (per model, GPU)

```bash
python -m scripts.score_wrong_answer \
    --model_name /path/to/Qwen3-8B \
    --dataset runs/full_25/output/dataset.jsonl \
    --model_outputs runs/Qwen3-8B/model_outputs.jsonl \
    --output runs/Qwen3-8B/model_outputs_with_wrong.jsonl \
    --device cuda
```

Adds a wrong-answer log-prob column for the wrongclaim,
wrong_intermediate, and wrong_bridge variants. The wrong answer is
parsed from the dataset metadata; no API needed.

### 4. Probes + summary (per model, GPU optional)

```bash
python -m probing_mvp.linear_probe runs/Qwen3-8B
python -m probing_mvp.lora_probe   runs/Qwen3-8B
python -m probing_mvp.make_summary runs/Qwen3-8B
```

### 5. Cross-model analysis (local, no GPU)

```bash
python -m scripts.bootstrap_ci \
    --runs runs/Qwen3-* runs/Qwen3.5-* \
    --output_matrix reports/behavior_matrix_with_ci.csv \
    --output_trends reports/headline_trend_tests.csv

python -m scripts.margin_analysis \
    --runs runs/Qwen3-* runs/Qwen3.5-* \
    --output_dir reports/margin \
    --figure_dir figures

python -m probing_mvp.make_plots runs/Qwen3-*/summary.json \
    --output-dir figures
```

## Key results

- **Wrong-Claim Robustness (KD)** on Qwen3 instruct: `|mean Δlog p|` grows
  from 0.38 (1.7B-Base) to 2.34 (8B), and the matched 8B / 8B-Base pair
  isolates the jump as instruction tuning rather than scale.
- **Margin decomposition on KD**: gold_drop −0.28 → −2.52 across Qwen3
  instruct 0.6B → 8B, while margin_variant grows +0.10 → +2.66.
  Sensitivity rises with scale and decision robustness rises with it.
- **Bridge-Fact Integration (CP)** decays with scale (1.41 → 0.03):
  bridge-RAG is high-value on small models, redundant on 8B.
- **Wrong-Bridge Robustness (CD)** is the opposite of KD: gold_drop
  grows but margin_variant stays negative throughout. The model is
  genuinely overridden by the planted wrong bridge.

## Author

Xingwei Qu (`xingweiqu`) / SnowDist.

## License

MIT.
