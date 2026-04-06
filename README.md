# V-Probing: Knowledge vs Reasoning Boundary

## 目标

用线性探针（linear probe）在 LLM 各层的 hidden states 上，区分模型是在做**知识检索**还是**推理计算**，并观察这个边界随训练步数如何变化。

## 核心假设

| 类型 | 特征 | 预期激活模式 |
|------|------|-------------|
| Knowledge | 直接事实提取（"巴黎是法国首都"） | 信号在**早期层**就稳定 |
| Reasoning | 多步推理（数学、逻辑） | 信号在**深层**才出现 |

随着训练增加，两类信号的分离度应该增强，边界层（boundary layer）应该更清晰。

## 方法

### 1. 数据集
- **Knowledge**：factual QA（TriviaQA / NaturalQuestions 风格）
- **Reasoning**：math/logic（GSM8K / LogiQA 风格）

### 2. 激活提取
对每个样本，在每层 `l` 提取最后一个 token 的 hidden state `h_l`：
```
sample → [h_0, h_1, ..., h_L]   shape: [n_layers, hidden_dim]
```

### 3. 线性探针（V-Probing）
在每层独立训练一个 logistic regression：
```
f_l : h_l → {0=knowledge, 1=reasoning}
```
用 4-fold cross-validation 评估准确率。

### 4. 跨 Checkpoint 分析
对不同训练步数的 checkpoint 重复上述过程，观察：
- 每层探针准确率曲线的形状变化
- "边界层"（argmax 准确率的层）随训练的漂移

## 运行

```bash
# 安装依赖
pip install numpy scikit-learn matplotlib transformers torch

# 模拟模式（无需 GPU，验证流程）
python probe.py --simulate

# 真实模型（需要 GPU + HuggingFace checkpoints）
python probe.py \
  --model_name gpt2 \
  --checkpoints ./ckpt_step100 ./ckpt_step500 ./ckpt_step1000 \
  --device cuda
```

## 输出

`probe_results.png` 包含两张图：

1. **左图**：每层探针准确率曲线（不同颜色=不同 checkpoint）
   - 曲线越陡、峰值越高 → 边界越清晰

2. **右图**：边界层（argmax 准确率）随训练步数的变化
   - 如果边界层随训练向早期层移动 → 知识被"压缩"到浅层
   - 如果边界层向深层移动 → 推理能力在深层增强

## 扩展方向

- **更大数据集**：替换 `KNOWLEDGE_SAMPLES` / `REASONING_SAMPLES` 为真实数据集
- **PCA 可视化**：在 boundary layer 对激活做 PCA，看两类样本的分离
- **非线性探针**：用 MLP probe 替换 logistic regression，测试是否有更强的非线性结构
- **Token-level**：不只看最后一个 token，看整个生成过程中激活的演变
