# 新一轮独立 seed 确认：启动记录

启动快照：2026-09-19T22:39:15.944Z（UTC）。此页是启动记录，不是实时刷新页面。

5 台节点均已启动训练，40 张 GPU 确认有计算负载。共 18 个固定训练任务，当前 5 个并发，其余自动接续；训练后自动完成 Math 与 grounded Knowledge 读数。

## 本轮优先目标

| 模型 | 主验证目标 | 三方比较 | 新 seeds |
|---|---|---|---|
| OLMo-3-7B-Instruct | Math 五组 reference NLL 均值 | Clean-SFT / matched-uniform / Ours | 81 / 82 / 83 |
| Qwen2.5-7B-Instruct | Math Clean BMK 宽松答案准确率 | Clean-SFT / matched-uniform / Ours | 81 / 82 / 83 |

每个任务 N=3,968、31 optimizer updates，学习率及其他训练设置保持不变；三组匹配行数和更新数，实际输入/监督 token 总数并不完全相同，会在训练回执中记录。配方与数据集沿用已观察 DEV 的候选，新 seeds 用于独立训练复验，不是新的盲测 benchmark。本轮 2 模型优先批次不是完整 11 模型面板。

“清晰胜出”的预设筛查条件：

- OLMo NLL：相对两个对照分别至少降低 0.01 nats/token，且三个 seed 的配对方向均有利。
- Qwen2.5 BMK：相对两个对照分别至少提升 0.5 个百分点，三个 seed 配对方向均有利，平均严格答案提取准确率不能低于任一对照。
- 同时报全部 Math / Knowledge 分项、误拒答与失败结果。通过这个筛查不等于已经证明统计显著，也不代表所有 domain 都赢。

## 启动核验

| 节点 | 当前训练 | PID |
|---|---|---:|
| node1 | `winner-v1-olmo3-7b-n3968-selected_recipe-s81` | 601658 |
| node2 | `winner-v1-olmo3-7b-n3968-uniform_matched_q-s81` | 625990 |
| node3 | `winner-v1-olmo3-7b-n3968-clean-s81` | 4094884 |
| node4 | `winner-v1-qwen25-7b-n3968-selected_recipe-s81` | 2603178 |
| node5 | `winner-v1-qwen25-7b-n3968-uniform_matched_q-s81` | 3457431 |

18/18 dry-runs 通过；新身份和旧队列完全分离。旧预算上限不约束此次新授权，但每一批仍使用固定任务清单，避免无边界重复训练。技术失败最多重试两次；不因分数不理想换 seed。

后台监控每 60 秒汇总一次，实时文件位于：

- `/mnt/hdfs/xwqu/winner_confirmation_20260919_v1/LIVE_EXPERIMENT_STATUS.json`
- `/mnt/hdfs/xwqu/winner_confirmation_20260919_v1/PROGRESS_REPORT.md`

监控只汇总，不参与选择、重新训练或提前宣布胜出。已有训练损失日志不作为最终 benchmark 结果。

[启动回执与完整任务清单](LAUNCH_STATUS.json) · [开跑前冻结的协议](PROTOCOL_FREEZE.json) · [上一批完整对照表](../experiment-review-20260919/README.md)
