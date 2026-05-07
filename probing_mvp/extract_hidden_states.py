"""Extract hidden states from a causal LM under multiple CoT and pooling settings.

For every dataset item we save:
  - last-token hidden state for every layer
  - mean-pooled hidden state for every layer
under each requested cot_state. The model also produces a short greedy
continuation; we record it together with a soft-correctness flag.

Outputs (under ``--output_dir``):
    hidden_<cot_state>_<pool>.npy        (N_items, n_layers, hidden_dim)
    item_index.jsonl                     one row per item with
                                            family_id, variant, mode, cot_state,
                                            row index in the .npy files
    model_outputs.jsonl                  one row per (item, cot_state) with
                                            generated text + correctness flag

This module never trains anything; it is the GPU-bound side of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


COT_PREFIXES = {
    "no_cot": "",
    "with_cot": "Let's think step by step.\n",
}


def _build_prompt(question: str, cot_state: str, tokenizer, use_chat_template: bool) -> str:
    prefix = COT_PREFIXES[cot_state]
    user_text = f"{prefix}{question}".strip()
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def _soft_correct(generated: str, gold: str) -> bool:
    g = (gold or "").strip().lower()
    p = (generated or "").strip().lower()
    if not g:
        return False
    return g in p or p.startswith(g)


def _gold_logprob(
    model,
    tokenizer,
    prompt: str,
    gold: str,
    device: str,
    max_length: int,
    top_k: int = 5,
) -> dict:
    """Score `gold` conditioned on `prompt`. Returns rich diagnostics:

        sum, mean: total / per-token log-probability of gold
        n_tokens: number of gold tokens scored
        per_token_logprob: list[float], length n_tokens
        gold_token_strs: list[str] (decoded gold tokens, for debugging)
        gold_first_token_rank: int — vocab rank of the first gold token at
                                its prediction position (0 = top-1)
        top_k_tokens: list[list[str]] — for each gold position, the top-k
                                competitors (decoded)
        top_k_logprobs: list[list[float]] — same shape, the logprobs of those

    If the prompt+gold overruns max_length, we left-truncate the prompt so
    the gold tokens are always scored.
    """
    nan_record = {
        "sum": float("nan"), "mean": float("nan"), "n_tokens": 0,
        "per_token_logprob": [], "gold_token_strs": [],
        "gold_first_token_rank": -1,
        "top_k_tokens": [], "top_k_logprobs": [],
    }
    if not gold:
        return nan_record
    sep = "" if (prompt.endswith((" ", "\n", "\t")) or not prompt) else " "
    full = prompt + sep + gold

    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_prompt = prompt_ids.shape[0]
    n_full = full_ids.shape[0]
    n_gold = max(n_full - n_prompt, 0)
    if n_gold == 0:
        return nan_record

    if n_full > max_length:
        keep = max_length
        full_ids = full_ids[-keep:]
        n_prompt = max(n_prompt - (n_full - keep), 0)
        n_full = full_ids.shape[0]
        n_gold = n_full - n_prompt
        if n_gold <= 0:
            return nan_record

    input_ids = full_ids.unsqueeze(0).to(device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0]                                  # [T, V]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # gold positions in input_ids are [n_prompt, n_full); their logits live at [n_prompt-1, n_full-1)
    target_idx = full_ids[n_prompt:].to(device)             # [n_gold]
    gather_at = log_probs[n_prompt - 1: n_full - 1]         # [n_gold, V]
    token_lp = gather_at.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    per_token = token_lp.cpu().tolist()
    s = float(sum(per_token))

    # gold rank at first gold position
    first_pos_lp = gather_at[0]                              # [V]
    sorted_desc = first_pos_lp.argsort(descending=True)
    rank_pos = (sorted_desc == target_idx[0]).nonzero(as_tuple=True)[0]
    gold_rank = int(rank_pos.item()) if rank_pos.numel() else -1

    # top-k competitors per gold position
    top_vals, top_idx = gather_at.topk(top_k, dim=-1)        # [n_gold, top_k]
    top_k_tokens = [[tokenizer.decode([int(t)]) for t in row] for row in top_idx.cpu().tolist()]
    top_k_logprobs = top_vals.cpu().tolist()
    gold_token_strs = [tokenizer.decode([int(t)]) for t in target_idx.cpu().tolist()]

    return {
        "sum": s,
        "mean": s / n_gold,
        "n_tokens": int(n_gold),
        "per_token_logprob": per_token,
        "gold_token_strs": gold_token_strs,
        "gold_first_token_rank": gold_rank,
        "top_k_tokens": top_k_tokens,
        "top_k_logprobs": top_k_logprobs,
    }


def extract_hidden_states(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    cot_states: Iterable[str] = ("no_cot", "with_cot"),
    pools: Iterable[str] = ("last", "mean"),
    device: str = "cuda",
    dtype: str = "float16",
    max_new_tokens: int = 32,
    use_chat_template: bool = True,
    max_length: int = 1024,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cot_states = tuple(cot_states)
    pools = tuple(pools)
    for c in cot_states:
        if c not in COT_PREFIXES:
            raise ValueError(f"unknown cot_state {c!r}; choose from {list(COT_PREFIXES)}")
    for p in pools:
        if p not in ("last", "mean"):
            raise ValueError(f"unknown pool {p!r}; choose from ('last', 'mean')")

    logger.info("loading model %s on %s (%s)", model_name, device, dtype)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    items = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
    logger.info("loaded %d items from %s", len(items), dataset_path)

    n_layers = model.config.num_hidden_layers + 1  # + embedding layer
    hidden_dim = model.config.hidden_size

    buffers = {
        (cot, pool): np.zeros((len(items), n_layers, hidden_dim), dtype=np.float16)
        for cot in cot_states
        for pool in pools
    }
    index_rows: list[dict] = []
    output_rows: list[dict] = []

    with torch.no_grad():
        for i, item in enumerate(items):
            if i % 25 == 0:
                logger.info("processing %d/%d", i, len(items))
            for cot in cot_states:
                prompt = _build_prompt(item["question"], cot, tokenizer, use_chat_template)
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                out_obj = model(**enc, output_hidden_states=True, use_cache=False)
                # hidden_states is a tuple of [B=1, T, H]; stack to [n_layers, T, H]
                stacked = torch.stack([h[0] for h in out_obj.hidden_states], dim=0)  # [L, T, H]
                attention_mask = enc.attention_mask[0]  # [T]

                if "last" in pools:
                    last_pos = int(attention_mask.sum().item()) - 1
                    h_last = stacked[:, last_pos, :].float().cpu().numpy()
                    buffers[(cot, "last")][i] = h_last
                if "mean" in pools:
                    mask = attention_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, T, 1]
                    summed = (stacked * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1.0)
                    h_mean = (summed / counts).float().cpu().numpy()
                    buffers[(cot, "mean")][i] = h_mean

                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                cont = tokenizer.decode(
                    gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True
                ).strip()

                gold = item.get("gold_answer", "")
                lp = _gold_logprob(model, tokenizer, prompt, gold, device, max_length)

                output_rows.append(
                    {
                        "row": i,
                        "family_id": item["family_id"],
                        "variant": item["variant"],
                        "mode": item.get("mode", "natural"),
                        "cot_state": cot,
                        "prompt": prompt,
                        "generation": cont,
                        "gold_answer": gold,
                        "correct": _soft_correct(cont, gold),
                        "gold_logprob_sum": lp["sum"],
                        "gold_logprob_mean": lp["mean"],
                        "n_gold_tokens": lp["n_tokens"],
                        "gold_per_token_logprob": lp["per_token_logprob"],
                        "gold_token_strs": lp["gold_token_strs"],
                        "gold_first_token_rank": lp["gold_first_token_rank"],
                        "top_k_tokens": lp["top_k_tokens"],
                        "top_k_logprobs": lp["top_k_logprobs"],
                    }
                )

            index_rows.append(
                {
                    "row": i,
                    "family_id": item["family_id"],
                    "variant": item["variant"],
                    "mode": item.get("mode", "natural"),
                }
            )

    for (cot, pool), arr in buffers.items():
        path = out / f"hidden_{cot}_{pool}.npy"
        np.save(path, arr)
        logger.info("saved %s shape=%s", path, arr.shape)

    with open(out / "item_index.jsonl", "w", encoding="utf-8") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "model_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in output_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d index rows and %d generation rows", len(index_rows), len(output_rows))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, help="HF model id or local path")
    p.add_argument("--dataset", required=True, help="dataset.jsonl produced by dataset_synthesis_mvp")
    p.add_argument("--output_dir", required=True, help="where to write hidden_*.npy and *.jsonl")
    p.add_argument("--cot_states", nargs="+", default=["no_cot", "with_cot"],
                   choices=list(COT_PREFIXES))
    p.add_argument("--pools", nargs="+", default=["last", "mean"], choices=["last", "mean"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--no_chat_template", action="store_true",
                   help="feed the bare prompt instead of using the tokenizer chat template")
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()

    extract_hidden_states(
        model_name=args.model_name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        cot_states=args.cot_states,
        pools=args.pools,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=not args.no_chat_template,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
