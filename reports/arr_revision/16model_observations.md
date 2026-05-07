# Sixteen-model intervention-signature observations

Generated 2026-05-07 from `runs/Qwen3-{0.6B, 0.6B-Base, 1.7B, 1.7B-Base, 4B-Base, 8B, 8B-Base}/summary.json` and `runs/Qwen3.5-{0.8B, 2B, 2B-Base, 4B, 4B-Base, 9B, 9B-Base, 35B-A3B, 35B-A3B-Base}/summary.json`. Cell metric is `|mean Δlogprob/token|` from `delta_value_stats`, `cot_state=no_cot`. CD is empty — Wrong-Bridge variant is not yet generated.

## Per-capability scaling reads (figures/_cross_model/fig_capacity_scaling.png)

- **KP (Hint Gain)** — flat across scale, ~0.0 to 0.7. No reliable scaling story.
- **KB (Paraphrase Drop)** — small magnitudes (≤1.0). Qwen3-1.7B-Base shows a 1.0 spike, otherwise ~0.0 to 0.5. Weak signal.
- **KD (Wrong-Claim Drop)** — dominated by a single Qwen3-8B-instruct point at 2.35. All other models, including 35B, are below 1.0. The previous draft cited "0.38 → 2.35 with scale" as the headline KD trend; with the broader sweep this is now better described as a Qwen3-8B-instruct outlier and the trend should be weakened in the paper.
- **RP (Scaffold Gain)** — clearest base-vs-instruct separation. Instruct line is generally above base across 0.6–9B, with the 8B-instruct point at 1.16. Qwen3.5-35B-A3B-Base spikes to 1.6, reversing direction; worth flagging.
- **RB (Rule-Removal Drop)** — modest, ~0.05 to 0.85, no monotonic trend.
- **RD (Wrong-Step Drop)** — instruct shows two Qwen3-1.7B / 4B spikes (~1.5); 8B-instruct drops to 0.0. Highly model-specific.
- **CP (Bridge-Fact Gain)** — strongest scaling story in the sweep: 2.0 (Qwen3-0.6B-Base) → 0.03 (Qwen3-8B) → 0.07 (Qwen3-8B-Base) → 0.76 (35B-A3B-Base). Bridge-fact help shrinks at mid scales then partially returns at the largest scale.
- **CB (Retrieval-Block Drop)** — bumpy. Instruct line trends downward with scale until 8B (0.18), then jumps back up at 35B-A3B (0.69).
- **CD (Wrong-Bridge Drop)** — not implemented.

## Cross-model matrix grid (figures/_cross_model/fig_capacity_matrix_grid.png)

Per-row colour structure is consistent across models: KB row spans mid-red to mid-green, RB row sits mid-yellow to mid-green, Hybrid row spans red to green. The within-row colour rank (which capability dominates) varies model-by-model; this is the visual case for the heterogeneity claim in the paper's discussion.

The Qwen3.5-35B-A3B (mixture-of-experts) entry is visibly different from dense models of similar scale: its Hybrid row has the strongest CB signal in the sweep (0.71) and a non-trivial CP signal (0.76) where dense 8B/9B models have collapsed to ~0.

## Implications for the paper draft

1. **§6.2 trend prose needs softening.** The "KD grows from 0.38 to 2.35 with scale" claim relies almost entirely on a single Qwen3-8B-instruct point. With the full 16-model sweep, the safer language is "we observe a Qwen3-8B-instruct outlier in Wrong-Claim Drop magnitude; the scaling trend is otherwise weak across the family."
2. **The CP-decay claim survives.** Bridge-Fact Gain drops monotonically from 0.6B to 8B in both base and instruct lines (with one 1.7B-base anomaly). This is the cleanest signature trend in the sweep.
3. **The RP-instruct-jump claim survives.** Instruct lines are consistently higher than base lines across most sizes. The 8B-instruct point at 1.16 is no longer a singular spike; 4B-instruct is also at 1.05.
4. **35B-A3B is worth a sentence.** The MoE entry breaks several scaling trends that dense models follow. We should note this without overinterpreting (single-data-point caveat).
5. **CD column remains empty.** The paper's claim of a 9-cell matrix still requires the Wrong-Bridge variant to land server-side (PART 1 of EXPERIMENT_PLAN_ARR_REVISION.md).

## Recommended next steps

- Phase 2: implement Wrong-Bridge variant generator on the server (highest priority — fills the CD column).
- Phase 3: bootstrap CIs over `model_outputs.jsonl` (lets us flag per-cell variability that scaling lines hide).
- Phase 4: rewrite §6.2 prose using the trend list above. Replace "0.38 → 2.35" with something like "Wrong-Claim Drop is heterogeneous across models with a Qwen3-8B-instruct outlier."
- Defer: per-model annotated heatmap (per-cell CI) until Phase 3 lands.
