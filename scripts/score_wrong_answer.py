"""Score the planted-wrong answer's log-probability for wrongclaim,
wrong_intermediate, and wrong_bridge variants — needed for margin analysis.

Reads:
    runs/<model>/model_outputs.jsonl  (existing per-item gold logps)
    runs/full_25/output/dataset.jsonl  (has wrong-claim metadata)

Writes:
    runs/<model>/model_outputs_with_wrong.jsonl
        same rows + columns:
            wrong_answer: str (planted wrong A')
            wrong_logprob_sum, wrong_logprob_mean, n_wrong_tokens
            wrong_per_token_logprob: list[float]

Usage on server:
    python -m scripts.score_wrong_answer \\
        --model_name /opt/tiger/Flame/Qwen3-8B \\
        --dataset runs/full_25/output/dataset.jsonl \\
        --model_outputs runs/Qwen3-8B/model_outputs.jsonl \\
        --output runs/Qwen3-8B/model_outputs_with_wrong.jsonl \\
        --device cuda

Then pull `runs/<model>/model_outputs_with_wrong.jsonl` back locally for
analysis.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-export _gold_logprob from probing_mvp for code reuse
from probing_mvp.extract_hidden_states import _gold_logprob, COT_PREFIXES

logger = logging.getLogger(__name__)


def _wrong_answer_for(item: dict) -> str | None:
    """Extract the planted-wrong answer string for variants where it exists."""
    md = item.get("metadata", {}) or {}
    v = item["variant"]
    if v == "wrong_bridge":
        return md.get("wrong_bridge_implied_answer")
    if v == "wrongclaim":
        # Parse "Mount Everest is located in India" -> "India".
        # Heuristic: the part of `wrong_claim` after the gold-related verb.
        # Safer: store wrong_implied_answer at generation time. We fall back
        # to last-noun-ish heuristic when not stored.
        for k in ("wrong_implied_answer", "wrong_answer", "planted_wrong_answer"):
            if md.get(k):
                return md[k]
        # Heuristic last-resort: take the last word of `wrong_claim` that is
        # capitalized or numeric. Fragile; prefer pre-storing.
        wc = md.get("wrong_claim", "")
        if wc:
            tokens = [t.strip(",.") for t in wc.split() if t.strip(",.")]
            for tok in reversed(tokens):
                if tok and (tok[0].isupper() or tok[0].isdigit()):
                    return tok
        return None
    if v == "wrong_intermediate":
        # metadata has e.g. {"wrong_claim": "the number is 9", "correct_answer": "6"}
        wc = md.get("wrong_claim", "")
        for tok in reversed(wc.replace(".", "").split()):
            if tok and (tok[0].isdigit() or tok[0].isupper()):
                return tok
        return None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model_outputs", required=True,
                   help="existing model_outputs.jsonl with gold_logprob")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Load dataset and key by (family_id, variant, mode) to grab metadata.
    ds_index: dict[tuple, dict] = {}
    with open(args.dataset) as f:
        for line in f:
            item = json.loads(line)
            key = (item["family_id"], item["variant"], item.get("mode", "natural"))
            ds_index[key] = item

    # Load existing outputs.
    outs: list[dict] = []
    with open(args.model_outputs) as f:
        for line in f:
            outs.append(json.loads(line))

    # Build a list of rows that need wrong-answer scoring.
    work: list[tuple[int, dict, str]] = []
    for i, row in enumerate(outs):
        v = row["variant"]
        if v not in ("wrongclaim", "wrong_intermediate", "wrong_bridge"):
            continue
        key = (row["family_id"], v, row.get("mode", "natural"))
        item = ds_index.get(key)
        if item is None:
            logger.warning("missing dataset row for %s; skipping", key)
            continue
        wrong = _wrong_answer_for(item)
        if not wrong:
            logger.warning("no wrong answer for %s; skipping", key)
            continue
        work.append((i, row, wrong))

    logger.info("rows needing wrong-answer scoring: %d / %d", len(work), len(outs))

    # Load the model.
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    with torch.no_grad():
        for n, (i, row, wrong) in enumerate(work):
            prompt = row["prompt"]
            lp = _gold_logprob(model, tokenizer, prompt, wrong,
                               args.device, args.max_length)
            outs[i]["wrong_answer"] = wrong
            outs[i]["wrong_logprob_sum"] = lp["sum"]
            outs[i]["wrong_logprob_mean"] = lp["mean"]
            outs[i]["n_wrong_tokens"] = lp["n_tokens"]
            outs[i]["wrong_per_token_logprob"] = lp["per_token_logprob"]
            outs[i]["wrong_first_token_rank"] = lp["gold_first_token_rank"]
            if n % 50 == 0:
                logger.info("%d/%d scored", n, len(work))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in outs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
