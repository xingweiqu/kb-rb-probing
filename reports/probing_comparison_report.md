# First-pass Probing & Dataset Comparison

- 模型：`/opt/tiger/coding-agent-synth-data/Qwen3-8B`
- seeds：`[0, 1, 2, 3, 4]`
- extreme-bin top quantile：`0.2`

## 核心结论（first pass）

### probe-ready 上当前最稳定的 atomic probes

- premise_sensitive (best_macro_f1≈0.899)
- wrong_claim_susceptible (best_macro_f1≈0.815)
- scaffold_sensitive (best_macro_f1≈1.000)
- paraphrase_fragile (best_macro_f1≈0.981)

### probe-ready 上当前不稳定/不可 probe 的标签

- removal_dependent (best_macro_f1≈-1.000)
- substitution_fragile (best_macro_f1≈-1.000)

## 数据对比概览

- 见 `dataset_comparison_summary.json`（family/block/sub_family/mode/variant 覆盖）
- probe-ready diagnostics：`family_diagnostics_probe_ready.json`
- gpt_test diagnostics：`family_diagnostics_gpt_test.json`

## 主要输出路径

- probing 结果目录：`outputs/first_pass_probing`
- hidden state cache：`cache/first_pass/hidden_states/qwen3_8b`
- score cache：`cache/first_pass/scores/qwen3_8b`
- metadata cache：`cache/first_pass/metadata/qwen3_8b`

