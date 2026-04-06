# Probing Knowledge vs Reasoning Boundaries in LLMs

## 研究问题

LLM 在回答问题时，究竟是在**检索记忆中的知识**（knowledge retrieval）还是在**执行推理计算**（reasoning computation）？这两种行为在模型内部是否有可区分的表示？

---

## 实验设计

### 模型

- **Method 1**：Qwen3-8B-Base（base model，无 instruction tuning）
- **Method 2 & 3**：Qwen3-8B（instruct model，有 chat template）

### 数据构造原则

**核心挑战**：knowledge 和 reasoning 不能用表面特征区分，必须控制变量。

**Knowledge 样本**：高度 memorized 的世界事实，模型在预训练中几乎必然见过原文。
```
"The capital of France is Paris."
"Shakespeare wrote Hamlet."
"Oxygen has atomic number 8."
```

**Reasoning 样本**：使用大数/罕见数字的计算题，模型不可能背过具体答案，必须推导。
```
"73847 multiplied by 29 equals 2141563."
"If 47x equals 3901, then x equals 83."
"A factory produces 2347 units per day. In 13 days it produces 30511 units."
```

**关键设计**：reasoning 样本刻意使用 5 位数乘法、多步文字题、代数方程，确保答案不在训练数据中以 verbatim 形式出现。

---

## 三种实验方法

### Method 1：Paraphrase Invariance（base model）

**假设**：如果模型对某类信息有稳定的内部表示（knowledge），换一种说法后 hidden state 应该变化很小。如果需要推理（reasoning），表面形式的变化会导致更大的激活差异。

**做法**：构造 20 对 paraphrase（原句 + 改写句），在每一层提取最后一个 token 的 hidden state，计算 cosine distance。

```
原句：  "The capital of France is Paris."
改写：  "Paris is the capital city of France."

原句：  "73847 multiplied by 29 equals 2141563."
改写：  "29 times 73847 is 2141563."
```

**指标**：每层的平均 paraphrase distance（1 − cosine similarity）。

---

### Method 2：Information-theoretic（instruct model）

**假设**：如果模型已经"知道"答案（knowledge），在 prompt 中提前给出答案对 logprob 的提升应该很小。如果模型需要推理（reasoning），提前给出答案会显著提升答案的 logprob。

**做法**：
- **无 hint**：`"Q: What is 73847 times 29? A:"` → 测量答案 token 的 log-prob
- **有 hint**：在 prompt 中注入答案 `"(Note: the answer is 2141563)"` → 再测量答案 log-prob
- **Delta** = logprob(hint) − logprob(no hint)

**指标**：knowledge 和 reasoning 样本的 delta 分布差异。

---

### Method 3：MLP vs Attention Ablation（instruct model）

**假设**（来自 Physics of LLMs 框架）：
- **Knowledge** 存储在 MLP 权重中（key-value memory），ablate MLP 会导致知识丢失
- **Reasoning** 依赖 Attention 机制在上下文中组合信息，ablate Attention 会破坏推理

**做法**：逐层将 MLP 或 Attention 的输出置零（forward hook），测量答案 token 的 logprob 下降量。

```
impact(layer l) = logprob_baseline − logprob_ablated_at_l
```

**指标**：knowledge vs reasoning 样本在每层的 logprob drop 曲线。

---

## 实验结果

### Method 1：Paraphrase Invariance

| 类型 | 平均 paraphrase distance |
|------|--------------------------|
| Knowledge | **0.080** |
| Reasoning | **0.133** |

**结论**：reasoning 样本的 paraphrase distance 比 knowledge 高 66%，说明推理类内容的表示对表面形式更敏感。

**Per-layer 分析**：
- 两类样本在 layer 0（embedding 层）distance 均为 0（完全相同的 token 序列）
- Knowledge 在 layer ~10 达到峰值（~0.11），之后**趋于平稳并在最后几层收敛**（layer 36 仅 0.007）
- Reasoning 在 layer ~27-28 才达到峰值（~0.23），**峰值更高、收敛更慢**

这个模式与假设一致：knowledge 在浅层就形成稳定表示，reasoning 需要更深的层来"计算"，中间层的表示更不稳定。

---

### Method 2：Information-theoretic

| 类型 | mean delta | std |
|------|-----------|-----|
| Knowledge | **−0.197** | 0.786 |
| Reasoning | **+0.484** | 0.818 |

**结论**：方向完全符合假设。
- Knowledge delta 为**负**：给出答案 hint 反而略微降低了 logprob，说明模型本来就对这些事实有高置信度，hint 引入了轻微的 context shift
- Reasoning delta 为**正**：给出答案 hint 显著提升了 logprob（+0.48），说明模型在没有 hint 时对这些计算结果不确定，看到答案后置信度大幅提升

**注意**：std 较大（~0.8），说明样本间差异显著，部分 reasoning 题目模型也有一定把握（可能是较简单的计算）。

---

### Method 3：MLP vs Attention Ablation

| 类型 | MLP drop | Attn drop |
|------|----------|-----------|
| Knowledge | **0.304** | 0.012 |
| Reasoning | **0.121** | −0.039 |

**结论**：
- **Knowledge 对 MLP 更敏感**（0.304 vs 0.121）：ablate MLP 后 knowledge 答案的 logprob 下降更多，支持"知识存储在 MLP 权重"的假设
- **Attention ablation 对两类影响都很小**（甚至为负），说明 Attention 的贡献在这个实验设置下不显著

**Per-layer 分析**：
- Layer 0 的 MLP impact 最大（knowledge=5.87, reasoning=0.56），说明**第一层 MLP 是 knowledge 存储的关键位置**
- 中间层（layer 5-30）的 MLP impact 接近 0，说明中间层 MLP 对最终答案的直接贡献较小
- 最后几层（layer 31-35）MLP impact 略有回升

---

## 综合结论

三种方法从不同角度都观察到了 knowledge 和 reasoning 的可区分信号：

| 方法 | 信号强度 | 方向 |
|------|---------|------|
| Paraphrase Invariance | ★★★ | reasoning 表示更不稳定 ✓ |
| Information-theoretic | ★★★ | reasoning 更依赖 hint ✓ |
| MLP Ablation | ★★ | knowledge 更依赖 MLP ✓ |
| Attention Ablation | ★ | 信号弱，不显著 |

**最强信号**来自 Method 1（paraphrase distance 差异 66%）和 Method 2（delta 方向完全相反）。

---

## 局限性与下一步

1. **样本量**：每类 20-30 个样本，统计功效有限，std 较大
2. **Attention ablation 失效**：可能因为 Qwen3 使用 GQA（grouped query attention），ablate 整个 attention 影响太全局，掩盖了层间差异；可以改为只 ablate 特定 attention head
3. **Knowledge 定义的模糊性**：部分"knowledge"样本（如速度、数字）模型可能也需要一定推理，边界不清晰
4. **下一步**：用 causal tracing（ROME 风格）精确定位存储特定事实的 MLP 层，与 ablation 结果对比验证
