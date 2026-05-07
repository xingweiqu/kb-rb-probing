# Prompt-artefact and split controls

All baselines use `GroupKFold` over `family_id` so backbones never
appear across train/test splits. Chance level on the three-way
task-family label is 1/3 ≈ 0.333.

| Control | Mean balanced acc | Std | Note |
|---|---|---|---|
| length_features | 0.864 | 0.069 | 5 numeric features, GroupKFold by family_id |
| tfidf_prompt | 0.914 | 0.082 | (1,2)-gram, max_features=5000, fit per-fold |
| random_label | 0.229 | 0.034 | labels shuffled, length features, expect ~0.33 |
| random_feature | 0.247 | 0.172 | N(0,I) features dim=64, expect ~0.33 |
| hidden_state_probe | 0.987 |  | Qwen3-8B, layer 12 |
| hidden_state_probe | 0.987 |  | Qwen3-8B-Base, layer 24 |
| hidden_state_probe | 0.987 |  | Qwen3-4B-Base, layer 13 |
| hidden_state_probe | 0.973 |  | Qwen3-1.7B-Base, layer 10 |

Dataset: `runs/full_25/output/dataset.jsonl` (75 natural originals).
Folds: 5 GroupKFold by `family_id`.
