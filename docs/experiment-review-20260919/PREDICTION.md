# 对 benchmark loss 的预测：与实际答题准确率分开

[← 返回总览](README.md)

所有误差均使用完整三个 seed 的实际均值；一个 cell = 模型 × repair 训练轴 × 评测组 × 目标剂量。每次比较严格使用预测都有效的相同 cells。更小更好。目标是已知 DEV 模型的 held-out dose，不是新模型盲测。

## 1. 冻结饱和响应模型 vs 最近校准点

| 指标 | 模型 | cells | 饱和 MAE | 最近点 MAE | 误差差值 | 均值较小者 |
|---|---|---|---|---|---|---|
| mean_per_item_nll | 全部 cells 加权 | 300 | 0.12927 | 0.11703 | +0.01225 | 最近校准点 |
| mean_per_item_nll | Qwen3-0.6B | 20 | 0.01184 | 0.01284 | -0.00100 | 饱和模型 |
| mean_per_item_nll | Qwen3-1.7B | 20 | 0.16326 | 0.18692 | -0.02366 | 饱和模型 |
| mean_per_item_nll | Qwen3-4B | 20 | 0.01711 | 0.01847 | -0.00136 | 饱和模型 |
| mean_per_item_nll | Qwen3-8B | 60 | 0.12710 | 0.13346 | -0.00636 | 饱和模型 |
| mean_per_item_nll | Qwen2.5-7B-Instruct | 20 | 0.05464 | 0.05816 | -0.00351 | 饱和模型 |
| mean_per_item_nll | Qwen3.5-2B | 20 | 0.14723 | 0.12871 | +0.01852 | 最近校准点 |
| mean_per_item_nll | Qwen3.5-9B | 20 | 0.06620 | 0.04801 | +0.01819 | 最近校准点 |
| mean_per_item_nll | Llama-3.1-8B-Instruct | 60 | 0.23862 | 0.17613 | +0.06249 | 最近校准点 |
| mean_per_item_nll | Mistral-7B-Instruct-v0.3 | 20 | 0.27643 | 0.20069 | +0.07574 | 最近校准点 |
| mean_per_item_nll | Olmo-3-7B-Instruct | 20 | 0.02004 | 0.03001 | -0.00997 | 饱和模型 |
| mean_per_item_nll | gemma-4-12B-it | 20 | 0.08519 | 0.14284 | -0.05766 | 饱和模型 |
| bits_per_byte | 全部 cells 加权 | 300 | 0.05545 | 0.04963 | +0.00582 | 最近校准点 |
| bits_per_byte | Qwen3-0.6B | 20 | 0.00398 | 0.00406 | -0.00008 | 饱和模型 |
| bits_per_byte | Qwen3-1.7B | 20 | 0.07252 | 0.07849 | -0.00597 | 饱和模型 |
| bits_per_byte | Qwen3-4B | 20 | 0.00718 | 0.00763 | -0.00045 | 饱和模型 |
| bits_per_byte | Qwen3-8B | 60 | 0.05878 | 0.06078 | -0.00200 | 饱和模型 |
| bits_per_byte | Qwen2.5-7B-Instruct | 20 | 0.01984 | 0.01877 | +0.00107 | 最近校准点 |
| bits_per_byte | Qwen3.5-2B | 20 | 0.06350 | 0.06078 | +0.00272 | 最近校准点 |
| bits_per_byte | Qwen3.5-9B | 20 | 0.03044 | 0.02017 | +0.01027 | 最近校准点 |
| bits_per_byte | Llama-3.1-8B-Instruct | 60 | 0.09199 | 0.06668 | +0.02531 | 最近校准点 |
| bits_per_byte | Mistral-7B-Instruct-v0.3 | 20 | 0.14167 | 0.10215 | +0.03951 | 最近校准点 |
| bits_per_byte | Olmo-3-7B-Instruct | 20 | 0.00630 | 0.00931 | -0.00301 | 饱和模型 |
| bits_per_byte | gemma-4-12B-it | 20 | 0.03395 | 0.06066 | -0.02671 | 饱和模型 |


### 按目标 repair 剂量拆分

| 指标 | repair token 比例 | cells | 饱和 MAE | 最近点 MAE | 误差差值 |
|---|---|---|---|---|---|
| mean_per_item_nll | 20% | 220 | 0.09078 | 0.08474 | +0.00604 |
| mean_per_item_nll | 40% | 40 | 0.17073 | 0.13970 | +0.03102 |
| mean_per_item_nll | 80% | 40 | 0.29953 | 0.27193 | +0.02760 |
| bits_per_byte | 20% | 220 | 0.03986 | 0.03643 | +0.00343 |
| bits_per_byte | 40% | 40 | 0.06888 | 0.05519 | +0.01368 |
| bits_per_byte | 80% | 40 | 0.12776 | 0.11666 | +0.01110 |

20% 覆盖 11 个模型；40% / 80% 只覆盖两个 anchor。因此上方总体是 cell 加权，不是 11 模型等权。


## 2. 其他预测基线：共同有效 cell 交集

| 指标 | 预测器 | 共同 cells | MAE ↓ |
|---|---|---|---|
| mean_per_item_nll | fixed_tau_saturating | 297 | 0.12353 |
| mean_per_item_nll | nearest_q10 | 297 | 0.11409 |
| mean_per_item_nll | log_linear | 297 | 0.20196 |
| mean_per_item_nll | raw_linear | 297 | 0.20549 |
| bits_per_byte | fixed_tau_saturating | 297 | 0.05369 |
| bits_per_byte | nearest_q10 | 297 | 0.04894 |
| bits_per_byte | log_linear | 297 | 0.08788 |
| bits_per_byte | raw_linear | 297 | 0.08830 |

raw-linear 每项指标各有 3 个 cell 没有有效预测；本表统一使用 297 个共同有效 cells，不能把它和上方 300-cell MAE 直接相减。


## 3. 额外九模型：校准预算消融

同一目标 cell 内比较；Clean baseline 始终需要。一个 repair pilot 档位 = 四个 repair 各跑该档 × 三个 seeds，不是一个训练 job。复用已有校准数据，属于预先冻结的离线消融。

| 指标 | 校准方案 | cells | 相对绝对误差均值 ↓ |
|---|---|---|---|
| mean_per_item_nll | transfer_no_repair_pilot | 180 | 9.947% |
| mean_per_item_nll | calibrate_q025_only | 180 | 8.396% |
| mean_per_item_nll | calibrate_q10_only | 180 | 3.172% |
| mean_per_item_nll | calibrate_q025_q10 | 180 | 3.741% |
| bits_per_byte | transfer_no_repair_pilot | 180 | 9.752% |
| bits_per_byte | calibrate_q025_only | 180 | 8.394% |
| bits_per_byte | calibrate_q10_only | 180 | 3.085% |
| bits_per_byte | calibrate_q025_q10 | 180 | 3.651% |


### 校准成本（每个新模型）

| 校准方案 | Clean baseline 训练 | repair pilot 训练 | 本次新增训练 |
|---|---|---|---|
| 无需 repair pilot | 3 seeds | 0 | 0（离线复用） |
| 仅低剂量档 | 3 seeds | 4 repair × 3 seeds = 12 | 0（离线复用） |
| 仅较高剂量档 | 3 seeds | 4 repair × 3 seeds = 12 | 0（离线复用） |
| 两档一起 | 3 seeds | 4 repair × 2 档 × 3 seeds = 24 | 0（离线复用） |

不含用于迁移拟合的 anchor 成本，也不含目标剂量的检验训练；一次训练身份不代表跨模型 GPU 小时相等。


方法追溯：nearest_q10 指直接沿用最近校准剂量观测；固定饱和曲线用同一套已冻结校准数据，τ=0.10。具体剂量与预测器标识保留在原始数据中。q 是监督 token 中 repair 的比例，不是数据量 N。

[逐 cell 的四预测器绝对误差 CSV](data/forecast_errors_by_cell.csv)：保留全部 600 个指标 cell，空白代表该预测器无有效预测。
