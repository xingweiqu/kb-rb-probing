# 下载 Pythia 和 OLMo 模型

## Pythia

Pythia 有完整的训练 checkpoint（143 个），直接从 HuggingFace 下载。

### 方式 1：代码里直接用 revision 加载（无需手动下载）

```python
from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-1b",
    revision="step64000",  # 指定训练步数
    cache_dir="./pythia_cache",
)
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-1b",
    revision="step64000",
)
```

可用 revision 格式：`step1`, `step512`, `step1000`, `step2000`, `step4000`, `step8000`, `step16000`, `step32000`, `step64000`, `step128000`, `step143000`

### 方式 2：提前下载到本地

```bash
pip install huggingface_hub

# 下载单个 checkpoint
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='EleutherAI/pythia-1b',
    revision='step64000',
    local_dir='./pythia-1b-step64000',
)
"

# 批量下载多个 checkpoint
for step in 1 512 1000 4000 16000 64000 143000; do
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='EleutherAI/pythia-1b',
    revision='step${step}',
    local_dir='./pythia-1b-step${step}',
)
"
done
```

### 可用模型规模

| 模型 | 参数量 | 推荐 |
|------|--------|------|
| `EleutherAI/pythia-160m` | 160M | 快速验证 |
| `EleutherAI/pythia-1b` | 1B | 推荐 |
| `EleutherAI/pythia-2.8b` | 2.8B | 更强信号 |
| `EleutherAI/pythia-6.9b` | 6.9B | 需要大显存 |

---

## OLMo

OLMo 也有公开 checkpoint，但格式和 Pythia 不同。

### 方式 1：HuggingFace 加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# OLMo-1B，有多个训练阶段 checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-1B",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "allenai/OLMo-1B",
    trust_remote_code=True,
)
```

### OLMo 中间 checkpoint

OLMo 的中间 checkpoint 在 HuggingFace 上以独立 repo 形式存放：

```bash
# 列出可用 checkpoint（需要 huggingface_hub）
python -c "
from huggingface_hub import list_repo_refs
refs = list_repo_refs('allenai/OLMo-1B')
for branch in refs.branches:
    print(branch.name)
"

# 下载特定 step
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='allenai/OLMo-1B',
    revision='step10000-tokens42B',  # 替换为实际 revision 名
    local_dir='./olmo-1b-step10000',
    trust_remote_code=True,
)
"
```

### 注意事项

- OLMo 需要 `trust_remote_code=True`
- OLMo 的 hidden states 访问方式与 Pythia/Qwen3 相同，`probe.py` 可直接复用
- OLMo checkpoint 命名格式：`stepXXXXX-tokensXXXB`，不如 Pythia 规整

---

## 推荐实验顺序

1. 先用 `EleutherAI/pythia-160m` 快速验证流程（小模型，下载快）
2. 确认信号方向正确后，换 `pythia-1b` 跑完整实验
3. OLMo 作为跨模型验证（证明结论不依赖特定架构）

## 国内服务器加速

```bash
# 设置镜像（如果 HuggingFace 访问慢）
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常运行 probe.py，会自动用镜像下载
python probe.py --pythia_model EleutherAI/pythia-1b --pythia_steps 1 512 1000 ...
```
