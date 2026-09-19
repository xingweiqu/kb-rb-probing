# 指标与来源审计

[← 返回总览](README.md)

数值为训练 seed 均值 ± 样本标准差；方括号为可用 seed 数。Δ 只使用 Ours 与 Uniform 都存在的相同 seed，不能直接减两列的非配对均值。SD 不是置信区间；不作显著性声明。所有结果均为已知 DEV，非盲测或 sealed final。


## 三批实验不能混用

| 实验 | 选择方法 / 条件 | 能回答的问题 |
|---|---|---|
| 九模型 N=3968 | 旧分段响应 + balanced residual；配方选于18步，验证31步并扩展源池 | 旧配方迁移到新训练条件后的实测对照 |
| 固定原条件 | 18步/2304draws；同一 Math-trained checkpoint 另读 Knowledge | 原训练条件下的配方表现及跨域读数 |
| 19日纯剂量预测 | 固定饱和曲线；冻结后测新剂量；不重选上述配方 | 未用于校准的剂量预测误差 |
| 两个anchor数据量 | 有限候选响应模型选择；N1920/3968/9984，seed72/73 | 已选配方跨数据量验证；此处与uniform重合 |

## BMK 定义

- Math Clean/EVD/PARA：acc_loose（宽松提取）；Clean 严格 acc_exact 另表。FMT：format.main（格式与内容共同正确）；ANS：insufficient_stop（应停止时停止），不能称通用答案准确率。
- Knowledge Clean/EVD：acc_strict；FMT：format.main；ANS：insufficient_stop。Knowledge 无 PARA，不做填补。ANS 两项控制指标保留在快照，不混进四/五组均值。
- Math 主生成面板分组 n：Clean 529、FMT 529、ANS 249、EVD 529、PARA 50；完整4700项面板还包含控制条件。逐 seed 原始 summary 的 n 随快照保留。
- NLL：每项 canonical full-response token-average NLL 再按 item 平均，单位 nats/token；BPB：bits/UTF-8 response byte。不是通用语言建模 loss。
- 宏 NLL 为组间等权描述性汇总；不是每项 pooled loss，也不是原选择器目标。
- 所有数值均保留不利方向；不按结果删模型。完整三 seed 的 pair 才计入摘要胜出计数；不是成功概率或统计显著性。

## 来源与复现

- `data/readouts.json.gz.b64`：gzip + Base64 编码的 JSON 快照，只含汇总指标、配方、hash 和来源路径，不含训练文本、原始问题或权重。导出逐条核对 spec 与 checkpoint 绑定。
- `data/historical.json`：历史数据量面板的 source-backed 配对数据。
- `data/forecast_errors.csv.gz.b64` 与 `data/calibration_budget_errors.csv.gz.b64`：压缩保留已冻结预测与实测，不做重拟合。`make_report.read_input()` 自动解码；`data/measurements_long.csv` 可直接查看。
- `export_readouts.py`：远端只读导出；`make_report.py`：离线构建，stdout 输出 path/content JSON。
- 复算：在本目录执行 `python3 make_report.py`；输出是所有生成文档的内容映射。`python3 make_report.py --file VOLUME_3968.md` 输出单个文档。`python3 -m unittest -v test_report.py` 执行核验。
- `.gz.b64` 为 UTF-8 文本的 gzip 压缩再 Base64，不是加密；压缩前后的 SHA-256 见 `data/input_manifest.json`。Knowledge 的生成与 NLL 均检查同一 checkpoint，生成面板的 scorer/fixture hash 也保留。

## 导出审计警告

[]
