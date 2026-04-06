# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 在服务器上用 GPU 运行
python probe.py --base_model /opt/tiger/Flame/Qwen3-8B-Base --instruct_model /opt/tiger/Flame/Qwen3-8B --device cuda

# 安装依赖
pip install torch transformers matplotlib numpy
```

## 架构

单文件实验：`probe.py` 对 Qwen3-8B 运行三种机制可解释性方法，区分知识检索与推理计算。

**模型分工**：base 模型（`Qwen3-8B-Base`）仅用于方法一；instruct 模型（`Qwen3-8B`）用于方法二和三。两个模型顺序加载，用完即删以节省显存。

**三种方法：**

1. **Paraphrase Invariance**（`method1_paraphrase`）：对改写对在每层提取最后一个 token 的 hidden state，计算 cosine distance。知识 → 浅层即稳定；推理 → 不稳定，峰值在深层。

2. **Information-theoretic**（`method2_information`）：对比有无 hint 注入时答案的 Δlogprob。知识 → delta 为负（模型本已确信）；推理 → delta 为正（看到答案后置信度大幅提升）。

3. **MLP vs Attention Ablation**（`method3_ablation`）：逐层用 forward hook 将 MLP 或 Attention 输出置零，测量 logprob 下降量。知识 → 对 MLP ablation 更敏感（尤其第 0 层）；Attention ablation 信号弱（可能因 Qwen3 使用 GQA）。

**输出文件**：`fig1_paraphrase.png`、`fig2_information.png`、`fig3_ablation.png`、`probe_results.json`（中间结果，可直接粘贴给 Claude 分析）。

## 关键实现细节

- Hidden states：`output_hidden_states=True` 返回 `[n_layers+1, B, T, H]`（含 embedding 层）。方法一使用全部层；方法三用 `len(model.model.layers)`（仅 transformer block，不含 embedding）。
- Ablation hook 将输出张量置零；attention 输出为 tuple 时只置零第一个元素。
- `apply_template` 用 `enable_thinking=False` 将文本包装为 Qwen3 chat template（仅 instruct 模型）。
- 结果用 `.tolist()` 将 numpy 数组转为 JSON 保存。
