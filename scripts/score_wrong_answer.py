"""Score the planted-wrong answer's log-probability for wrongclaim,
wrong_intermediate, and wrong_bridge variants — needed for margin analysis.

Reads:
    runs/<model>/model_outputs.jsonl   (existing per-item gold logps)
    runs/full_25/output/dataset.jsonl  (has wrong-claim metadata)

Writes:
    runs/<model>/model_outputs_with_wrong.jsonl
        same rows + (where applicable) columns:
            wrong_answer: str (planted wrong A')
            wrong_logprob_sum, wrong_logprob_mean, n_wrong_tokens
            wrong_per_token_logprob: list[float]
            wrong_first_token_rank: int

Wrong-answer extraction:
    - wrong_bridge: read metadata.wrong_bridge_implied_answer directly.
    - wrongclaim / wrong_intermediate: parse the trailing
      capitalized-or-numeric phrase from `injected_premise` (preferred)
      or `wrong_claim`. Rows where parsing fails or the parsed string
      equals the gold answer are skipped (no wrong_logprob_* written).

No API dependency.

Usage on server:
    python -m scripts.score_wrong_answer \\
        --model_name /opt/tiger/Flame/Qwen3-8B \\
        --dataset runs/full_25/output/dataset.jsonl \\
        --model_outputs runs/Qwen3-8B/model_outputs.jsonl \\
        --output runs/Qwen3-8B/model_outputs_with_wrong.jsonl \\
        --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probing_mvp.extract_hidden_states import _gold_logprob

logger = logging.getLogger(__name__)

_KEEP_LOWER = {"the", "a", "an", "of", "in"}


def _extract_wrong_answer(item: dict) -> str | None:
    """Pure-rule extraction of the planted-wrong answer.

    For wrong_bridge, read metadata.wrong_bridge_implied_answer.
    For wrongclaim / wrong_intermediate, walk back from the end of
    `injected_premise` (preferred) or `wrong_claim`, collecting trailing
    Capitalized / numeric / function-word tokens. Returns None when no
    candidate can be extracted (the row is then skipped during scoring).
    """
    md = item.get("metadata", {}) or {}
    if item["variant"] == "wrong_bridge":
        return md.get("wrong_bridge_implied_answer")

    text = md.get("injected_premise") or md.get("wrong_claim", "")
    if not text:
        return None

    text = text.strip().rstrip(".!?,;:")
    tokens = text.split()
    keep: list[str] = []
    for tok in reversed(tokens):
        bare = tok.strip(",.;:\"'")
        if not bare:
            continue
        if bare[0].isupper() or bare[0].isdigit() or bare.lower() in _KEEP_LOWER:
            keep.append(bare)
        else:
            break
    keep.reverse()
    while keep and keep[0].lower() in _KEEP_LOWER:
        keep.pop(0)
    return " ".join(keep) if keep else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", required=True,
                   help="dataset.jsonl with metadata.wrong_*; no enrichment needed")
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

    ds_index: dict[tuple, dict] = {}
    with open(args.dataset) as f:
        for line in f:
            item = json.loads(line)
            key = (item["family_id"], item["variant"], item.get("mode", "natural"))
            ds_index[key] = item

    outs: list[dict] = []
    with open(args.model_outputs) as f:
        for line in f:
            outs.append(json.loads(line))

    work: list[tuple[int, dict, str]] = []
    skipped_no_extract = 0
    skipped_matches_gold = 0
    for i, row in enumerate(outs):
        v = row["variant"]
        if v not in ("wrongclaim", "wrong_intermediate", "wrong_bridge"):
            continue
        key = (row["family_id"], v, row.get("mode", "natural"))
        item = ds_index.get(key)
        if item is None:
            logger.warning("missing dataset row for %s; skipping", key)
            continue
        wrong = _extract_wrong_answer(item)
        if not wrong:
            skipped_no_extract += 1
            continue
        if wrong.strip().lower() == row["gold_answer"].strip().lower():
            skipped_matches_gold += 1
            continue
        work.append((i, row, wrong))

    logger.info(
        "rows scoring wrong-answer: %d / %d (skipped: %d unparseable, %d match gold)",
        len(work), len(outs), skipped_no_extract, skipped_matches_gold,
    )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    with torch.no_grad():
        for n, (i, row, wrong) in enumerate(work):
            lp = _gold_logprob(model, tokenizer, row["prompt"], wrong,
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
