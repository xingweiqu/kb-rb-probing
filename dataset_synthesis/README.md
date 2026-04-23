# Dataset Synthesis — Pilot Item-Family Pipeline

第一阶段只做 pilot dataset synthesis，不做 probing、训练、评测。

## 核心概念

**Item Family**: 数据基本单位不是单题，而是 family。每个 family 对应同一个 underlying problem，通过 variants 构造 response profile。

**三层结构**:
- Normal variants: 19 种 variant，对齐 12 个 atomic capabilities
- Symbolic variants: 用 Unicode 符号（∆, ◇, ⊕...）替换实体，与 normal 同构
- MCQ variants: 4 选 1，symbolic 优先

## 12 个 Atomic Capabilities

| 大类 | 原子能力 | 对应 Variants |
|------|---------|--------------|
| Premise-sensitive | evidence-access | original, hint, premise, premise_removal |
| Premise-sensitive | evidence-localization | highlight |
| Premise-sensitive | evidence-integration | full_support_bundle |
| Scaffold-sensitive | decomposition-sensitive | scaffold_1, scaffold_2, scaffold_3 |
| Scaffold-sensitive | order-sensitive | scaffold_shuffled |
| Scaffold-sensitive | intermediate-state-sensitive | cot_full, cot_partial |
| Wrong-claim-susceptible | cue-susceptible | wrongclaim_bare |
| Wrong-claim-susceptible | authority-susceptible | wrongclaim_confident, wrongclaim_attributed |
| Wrong-claim-susceptible | conflict-resolution-weak | competing_claims |
| Substitution-fragile | lexical-fragile | paraphrase |
| Substitution-fragile | terminology-fragile | terminology_swap |
| Substitution-fragile | structure-misaligned | substitution |

另有 cot_shuffled（CoT 链打乱顺序），共 20 种 variant。

## 19 种 Variant 定义

- **original**: 标准题面
- **hint**: 方向性帮助，不补齐 support chain
- **premise**: 显式补入关键 support fact/rule
- **premise_removal**: 删掉关键 support
- **highlight**: 高亮关键证据位置，不加新事实
- **full_support_bundle**: 提供完整 support chain
- **scaffold_1/2/3**: 给 1/2/3 步中间拆解
- **scaffold_shuffled**: scaffold 内容打乱顺序
- **cot_full**: 完整推理链 + 问 final answer
- **cot_partial**: 部分推理链，模型补完
- **cot_shuffled**: 推理链打乱顺序
- **wrongclaim_bare**: 插入裸错误 claim
- **wrongclaim_confident**: 错误 claim + 高置信包装
- **wrongclaim_attributed**: 错误 claim + 权威归因
- **competing_claims**: 同时给对错 claim
- **paraphrase**: 只改表达不改语义
- **terminology_swap**: 通用词换领域术语
- **substitution**: 替换实体，保持结构

## Pipeline

5 阶段，每阶段有 checkpoint，支持断点续跑：

```
Stage 1: structures    → checkpoints/01_structures.json
Stage 2: base_items    → checkpoints/02_base_items.json
Stage 3: variants      → checkpoints/03_variants.json
Stage 4: symbolic      → checkpoints/04_symbolic.json  (纯程序化，无 API)
Stage 5: mcq           → checkpoints/05_mcq.json
Export:  → output/dataset.json, output/dataset.jsonl, output/stats.json
```

## 使用

```bash
# 需要先实现 api_client.py 中的 call_api 方法
python -m dataset_synthesis.pipeline --output_dir ./synthesis_output

# 自定义数量
python -m dataset_synthesis.pipeline --kb 25 --rb 25 --hybrid 25 --output_dir ./synthesis_output
```

## API 接口

唯一需要你实现的是 `dataset_synthesis/api_client.py` 中的 `APIClient.call_api` 方法。
接口签名：`call_api(self, system: str, user: str) -> str`

## 目录结构

```
dataset_synthesis/
  __init__.py
  schema.py              # Family, Variant, MCQItem dataclasses
  structures.py          # Stage 1-2: structure + base item generation
  builders/
    kb.py                # KB prompts
    rb.py                # RB prompts
    hybrid.py            # Hybrid prompts
  variants.py            # Stage 3: 19 variant generators
  symbolic.py            # Stage 4: Unicode 符号替换
  mcq.py                 # Stage 5: MCQ distractor generation
  pipeline.py            # 5-stage orchestrator
  api_client.py          # API wrapper (需要你实现 call_api)
  export.py              # JSON/JSONL 导出
  stats.py               # 统计报告
  configs/
    defaults.py          # 默认参数
```
