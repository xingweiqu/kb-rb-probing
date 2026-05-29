# Atomic-repair data v0 — LLM-Physics-style SFT dataset

A small, oracle-verifiable SFT dataset for the **atomic-repair** task.
Each item is a `(problem, tentative_answer) → (diagnosis, repair_skill, repair_trace, final_answer)` example.
The dataset is produced **entirely locally**: no API calls, no GPU, no third-party generation. The pipeline is seedable and reproducible.

This directory is meant to be shipped as a git branch (`atomic-repair-data-v0`) and consumed on a GPU server via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

---

## 1. Why LLM-Physics style?

Conventional "fine-tune on chat data" pipelines have two failure modes:

- **Pure symbolic input** (`book_17 rel_3 author_8`) is easy to verify but gives a model nothing to ground its reasoning in. Models that only see formal notation tend not to transfer to natural-language probes.
- **Per-item LLM generation** (sending every prompt through a frontier model) is expensive, non-deterministic, and gives no oracle: we cannot programmatically check that a `repair_trace` actually uses the bridge fact.

The LLM-Physics / iGSM line (Allen-Zhu et al.) splits the difference: build a **symbolic oracle world**, then **deterministically realize** it into natural language. That gives every example:

1. a programmatic oracle so the validator can refuse anything ungrounded;
2. natural-language *problems* the model can actually generalize from;
3. an explicit **retry/correction trace** for the repair cells, modeled on Part 2.2 of the iGSM line: wrong state → diagnose → repair → final answer.

That is what this pipeline does.

## 2. Why no pure-symbolic-only input?

We do retain a small *compact-symbolic* surface form (5% of items) so the model is forced to encode structure rather than surface tokens. But the main surface form (80%) is natural English, and the rest (15%) is "Facts: ... / Question: ..." fact-table style. We never train on 100% symbolic notation, and we never train on 100% fluent-LLM prose.

## 3. Why only H-Aug / H-Abl / H-Cor / K-Cor / Clean (no R-* yet)?

The body of the paper shows two things relevant to this dataset:

- **H-Cor flips most** under a wrong-bridge attack (mean flip rate ≈ 0.66 across 13 backbones with H-Cor data, only cell with negative mean margin),
- the **R-* cells move very little** under perturbations (R-Aug diagnostic rate 0.02, R-Cor 0.03).

So R-* perturbations don't currently produce a clean repair-vs-no-repair signal on these item pools. Until we have harder R items, training a repair model on R-* would teach over-repair without a clear win. We start with the cells where the signal is strongest:

| Cell | Failure mode | Repair skill |
|---|---|---|
| **H-Aug** | bridge fact missing | retrieve the bridge fact, then answer |
| **H-Abl** | bridge entity masked | recover the bridge entity from oracle facts, then answer |
| **H-Cor** | wrong bridge planted | verify the bridge against the gold facts, reject the false bridge |
| **K-Cor** | wrong 1-hop claim | contradiction check vs. the known fact |
| **Clean** | nothing is wrong | **do not over-repair**; keep the answer |

`Clean` is critical — without it, the model learns to "always repair," which collapses performance on benign inputs.

## 4. Pipeline at a glance

```
generate_repair_data.py   →   repair_raw_*.jsonl
validate_repair_data.py   →   data_sanity_report.json  (exits 1 on FAIL)
convert_to_llamafactory.py →  repair_lf_*.json + dataset_info.json
push_repair_data_branch.sh → push to atomic-repair-data-v0
```

### Generate

```bash
bash scripts/run_00_generate_data.sh
```

Produces `data/repair_raw_train.jsonl` (2,100 rows) and `data/repair_raw_eval.jsonl` (550 rows). Each row carries `oracle_facts` (English) + `symbolic_facts` (triples) so anyone downstream can re-verify it.

### Validate

```bash
bash scripts/run_01_validate_data.sh
```

Runs the hard checks in `validate_repair_data.py`. **Exit 1 if any check fails.** A `data/data_sanity_report.json` is always written.

Hard checks include:

- expected counts per (split, cell);
- `id` and `problem` uniqueness;
- `final_answer == gold_answer` for every row;
- `Clean tentative_answer == gold_answer`; `non-Clean tentative_answer != gold_answer`;
- `Clean should_repair == false`; `non-Clean should_repair == true`;
- `H-Cor`/`K-Cor` must carry a `planted_wrong_answer` distinct from gold, and `tentative_answer == planted_wrong_answer`;
- `diagnosis` and `repair_skill` strings match the cell;
- `oracle_facts` / `symbolic_facts` non-empty;
- `repair_trace` mentions a token from at least one oracle fact (so it can't be ungrounded);
- LLaMA-Factory output JSON round-trips;
- duplicate problem rate == 0;
- **held-out template check**: eval templates are disjoint from train templates;
- **held-out entity check**: eval head entities are disjoint from train head entities.

### Convert to LLaMA-Factory

```bash
bash scripts/run_02_convert_to_llamafactory.sh
```

Produces `data/repair_lf_train.json`, `data/repair_lf_eval.json`, and `data/dataset_info.json` (alpaca format).

The instruction is fixed across rows:

> You are an atomic repair agent. Given a problem and a tentative answer, diagnose whether there is an atomic-capacity failure. If there is a failure, choose the correct repair skill and repair the answer. If there is no failure, do not over-repair. Return only valid JSON.

The `input` field is `"Problem:\n{problem}\n\nTentative answer:\n{tentative_answer}"` and `output` is a JSON string with `diagnosis / repair_skill / repair_trace / final_answer`.

### Push the branch

```bash
bash scripts/push_repair_data_branch.sh atomic-repair-data-v0
```

## 5. Server usage (LLaMA-Factory)

On a server with a Qwen3-8B-Base checkpoint:

```bash
# 1. Clone LLaMA-Factory.
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"

# 2. Pull this branch alongside it.
cd ..
git clone -b atomic-repair-data-v0 <repo-url> probing
cd probing/atomic_repair

# 3. Symlink the dataset into LLaMA-Factory's data dir (or point dataset_dir at it).
#    The configs in this repo use `dataset_dir: ./data` (relative to atomic_repair/).
#    Either run from atomic_repair/, or copy data/ into LLaMA-Factory/data/.
ln -sf "$(pwd)/data/dataset_info.json" "$HOME/LLaMA-Factory/data/atomic_repair_dataset_info.json"  # optional pattern
# Recommended: just point the config at our data dir directly.

# 4. Edit configs/qwen3_8b_repair_lora_sft.yaml:
#    - set model_name_or_path to the actual Qwen3-8B-Base path
#    - confirm the `template` field matches your LLaMA-Factory version
#      (Qwen3 typically uses `qwen`; some forks use `qwen3` or `qwen2`).

# 5. Train.
LF_DIR=$HOME/LLaMA-Factory bash scripts/run_03_train_llamafactory.sh

# 6. Predict on the held-out eval set.
LF_DIR=$HOME/LLaMA-Factory bash scripts/run_04_predict_llamafactory.sh
```

## 6. Reading the sanity report

`data/data_sanity_report.json` is the single source of truth for whether the dataset is healthy:

- `status`: `"PASS"` if everything is clean, `"FAIL"` otherwise.
- `failures`: list of human-readable strings naming each failed check.
- `counts`: totals + per-cell + per-family + per-surface for both splits.
- `held_out_templates` / `held_out_entities`: the disjointness check made explicit, including the overlap count (should be 0).
- `duplicate_problems`: should be 0 for both splits.
- `samples`: 5 example rows per cell per split (for human spot-check).

If `status != "PASS"`, **do not push**. Regenerate or fix the generator.

## 7. Schema (raw JSONL row)

```json
{
  "id": "H-Cor_train_000001",
  "cell": "H-Cor",
  "surface_type": "naturalized",
  "relation_family": "book_author_nationality",
  "graph_pattern": "two_hop_bridge",
  "task_type": "hybrid_bridge_verification",
  "problem": "It is widely said that Silver River was written by Anton Hale. What is the nationality of the author who wrote Silver River?",
  "tentative_answer": "Norlandian",
  "gold_answer": "Lydorian",
  "planted_wrong_answer": "Norlandian",
  "diagnosis": "wrong_bridge_contamination",
  "repair_skill": "bridge_source_verification",
  "repair_trace": "The tentative answer follows the planted bridge 'Anton Hale'. Verify the bridge claim against the known facts. The planted bridge is false: Silver River was written by Maria Voss (not Anton Hale). Use the correct bridge fact: Maria Voss is Lydorian. Therefore the final answer is Lydorian, not Norlandian.",
  "final_answer": "Lydorian",
  "oracle_facts": [
    "Silver River was written by Maria Voss.",
    "Maria Voss is Lydorian.",
    "Anton Hale is Norlandian."
  ],
  "symbolic_facts": [
    ["Silver River", "written_by", "Maria Voss"],
    ["Maria Voss", "nationality", "Lydorian"],
    ["Anton Hale", "nationality", "Norlandian"]
  ],
  "should_repair": true,
  "split": "train",
  "template_id": "book_T1",
  "entity_split": "train"
}
```

## 8. Caveats

- **Synthetic entities**: country / author / city / scientist names are entirely fictitious, drawn from a fixed inventory in `generate_repair_data.py`. This is deliberate — it prevents pretraining-knowledge leakage and lets the validator be fully oracle-verifiable.
- **Six relation families** is a starting set; more can be added by appending to `FAMILIES` and re-running the generator.
- **Eval generalization is template + entity, not graph-structure**: every cell here is the same 2-hop or 1-hop pattern. Held-out graph patterns are a future axis.
- **Single-LoRA setup**: the configs target Qwen3-8B-Base; we have not tuned hyperparameters for any other backbone.

That's the contract.
