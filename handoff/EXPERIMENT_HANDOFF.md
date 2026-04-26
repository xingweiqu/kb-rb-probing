# Experiment Handoff (Minimal)

本文件只做“做了什么实验/产物在哪里”的整合说明，不做研究结论。

## 1) Repo / 版本

- Repo: `seed/latent_mas` (branch: `dev`)
- 当前 commit: `e27a17569c98e9252603e44a9f75c1d2e422b0f0`
- 模型路径: `/opt/tiger/coding-agent-synth-data/Qwen3-8B`

## 2) 数据与主要流程（first-pass probing + dataset comparison）

### 输入数据（路径）

- probe-ready paired: `data/probe_ready_180_paired.jsonl`
- probe-ready MCQ: `data/probe_ready_180_paired_mcq.jsonl`
- gpt_test: `data/gpt_test.jsonl`

### 评分与标签

- 对每个 `(family, variant)` 计算选择题式 logprob 打分（候选答案集），并落盘到 score cache。
- 从行为分数构造 per-family signals（例如 `wrongclaim_drop = orig - wrongclaim_bare`、`removal_drop = orig - premise_removal`）。
- 使用 extreme-bin（top/bottom quantile，中间丢弃）把 signals 转成二分类 atomic labels，并输出：
  - `atomic_labels_probe_ready.json`
  - `atomic_labels_gpt_test.json`

### Hidden states（用于 probing 的特征）

- 对每个 prompt 抽取 last-nonpad token 的 all-layer hidden states。
- 两个位置（position）用于缓存：
  - `final_input`: 直接对 prompt 做 tokenizer
  - `pre_answer`: prompt 末尾追加 `\nAnswer:` 后做 tokenizer

### Probing 任务

- 线性 probing：对每层做 `StandardScaler + LogisticRegression`，seed 复用，输出 per-layer metrics 与最佳层。
- feature modes：
  - `absolute::h_original`
  - `delta::<variant>-original`（同 family 内 `h_variant - h_original`）
- split/transfer（family-level）：
  - `random_family_split`（sanity）
  - `held_out_sub_family::<sf>`
  - `block::train_*__test_*`
  - `realization::train_natural__test_symbolic` / `realization::train_symbolic__test_natural`
  - `cross_dataset_transfer_v2`（probe_ready <-> gpt_test，family_id 加前缀避免碰撞，label 用 train dataset 阈值对齐后再在 test 上赋值）
  - `mcq_transfer_v2`（non-MCQ strict_symbolic <-> MCQ strict_symbolic；不应混入主结论）

## 3) 本次 focused validation（仅 two labels）

目标：只做聚焦整理与 sanity check，不重跑模型、不重提 hidden states、不下研究结论。

### 输出文件（已生成）

- hidden input sanity：`reports/hidden_input_sanity_check.md`
  - 给出 5 条“实际送入 tokenizer 的 input_text 样例”（original / wrongclaim_bare / premise_removal），并检查 gold 是否出现在 input_text。
- strict rows（只保留严格满足计数阈值的既有结果）：`reports/focused_two_label_strict_rows.csv`
  - label in {`wrong_claim_susceptible`, `removal_dependent`}
  - 非 MCQ、非 random split
  - train/test pos/neg 均 >= 10
  - 优先 delta feature（CSV 内 `is_delta_feature` 标记）
- two-label compact summary：`reports/focused_two_label_summary.md`
  - 列出两标签的 valid delta transfer rows
  - 标注是否存在 natural<->symbolic 双向结果
  - 标注 cross_dataset_transfer_v2 是否存在 delta（如无则写明 no delta cross-dataset result）
  - 给出 `removal_drop` / `wrongclaim_drop` 的 signal value 分布与阈值/同值信息（用于解释极端 pos/neg 计数现象）

## 4) 最小化交付物（建议人工从这里开始看）

- `handoff/probing_minimal_review_bundle.zip`
  - `REVIEW_PACKET.md`
  - `key_results_for_human_review.csv`
- （新增）`handoff/focused_two_label_bundle.zip`
  - `hidden_input_sanity_check.md`
  - `focused_two_label_strict_rows.csv`
  - `focused_two_label_summary.md`

