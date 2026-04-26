#!/usr/bin/env python3
"""完整 first-pass probing + dataset comparison（线性 probe）。

目标：
  - 在同一套 probing 框架下，对 probe-ready（non-MCQ / MCQ）与 gpt_test 做一致的：
      * diagnostics
      * family-level 行为标签（从 score 差值导出，extreme-bin quantile）
      * layerwise linear probing（absolute + delta；final_input + pre_answer）
      * 各类 split + transfer + cross-dataset transfer

运行：
  python run_first_pass_probing.py \
    --model /opt/tiger/coding-agent-synth-data/Qwen3-8B

说明：
  - 本脚本会缓存 hidden states 与 variant-level scores，避免重复跑模型。
  - hard gate：如果某个组合（标签/数据/切分）不可 probe，会输出 unusable reason，不会 silent skip。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probing.hidden_states import HiddenStateStore
from probing.metrics import per_layer_metrics
from probing.probes import run_probe_with_seeds


REQUIRED_VARIANTS = [
    "original",
    "hint",
    "premise",
    "premise_removal",
    "highlight",
    "wrongclaim_bare",
    "competing_claims",
    "paraphrase",
    "scaffold_1",
    "scaffold_2",
]


PRIMARY_LABELS = [
    "premise_sensitive",
    "removal_dependent",
    "wrong_claim_susceptible",
    "scaffold_sensitive",
    "paraphrase_fragile",
    "substitution_fragile",
]

EXTRA_LABELS = [
    "access_sensitive",
    "localization_sensitive",
    "integration_sensitive",
    "decomposition_sensitive",
    "order_sensitive",
    "intermediate_state_sensitive",
    "cue_susceptible",
    "authority_susceptible",
    "conflict_resolution_weak",
    "lexical_fragile",
    "terminology_fragile",
    "structure_misaligned",
]


STRICT_TOKENS = {"E1", "E2", "R1", "C1", "V1"}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_json(obj: Any, path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    _ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
        return
    # stable field order
    fieldnames: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _token_contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    if re.fullmatch(r"[A-Za-z0-9_\-]+", needle):
        return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None
    return needle in haystack


def _gold_aliases(gold: str) -> list[str]:
    g = (gold or "").strip()
    if not g:
        return []
    out: list[str] = []

    def add(x: str) -> None:
        x = (x or "").strip()
        if not x:
            return
        if x not in out:
            out.append(x)

    add(g)
    if re.match(r"(?i)^the\s+", g):
        add(re.sub(r"(?i)^the\s+", "", g).strip())
    if "(" in g and ")" in g:
        add(re.sub(r"\s*\([^)]*\)\s*", " ", g).strip())
    if "," in g:
        add(g.split(",", 1)[0].strip())
    ng = _norm(g)
    if ng in {"yes", "true"}:
        add("yes"); add("true")
    if ng in {"no", "false"}:
        add("no"); add("false")
    # numeric normalization
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", g):
        try:
            add(str(int(float(g))) if "." not in g else str(float(g)))
        except Exception:
            pass
    return [x for x in out if len(_norm(x)) >= 2]


@dataclass
class Family:
    dataset: str
    family_id: str
    task_family: str
    sub_family: str
    mode: str
    source_family_id: str
    paired_family_id: str
    gold_answer: str
    answer_type: str
    atomic_answer_candidates: list[str] | None
    variants: dict[str, str]
    # optional MCQ
    mcq_question: str | None = None
    mcq_options: list[str] | None = None
    mcq_correct_index: int | None = None


def load_probe_ready_paired(path: Path, dataset_name: str) -> list[Family]:
    rows = _read_jsonl(path)
    out: list[Family] = []
    for r in rows:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        tf = str(r.get("task_family", ""))
        sf = str(r.get("sub_family", ""))
        fid = str(r.get("family_id", ""))
        mode = str(meta.get("mode", ""))
        src = str(meta.get("source_family_id", ""))
        paired = str(meta.get("paired_family_id", ""))
        gold = str(r.get("gold_answer", "") or "")
        at = str(meta.get("answer_type", ""))
        aac = meta.get("atomic_target_candidates") if isinstance(meta.get("atomic_target_candidates"), list) else None
        if isinstance(aac, list):
            aac = [str(x) for x in aac if isinstance(x, str) and x.strip()]
        nv = r.get("normal_variants")
        variants: dict[str, str] = {}
        if isinstance(nv, dict):
            for k in REQUIRED_VARIANTS:
                v = nv.get(k)
                q = v.get("question") if isinstance(v, dict) else None
                if isinstance(q, str) and q.strip():
                    variants[k] = q.strip()
        out.append(
            Family(
                dataset=dataset_name,
                family_id=fid,
                task_family=tf,
                sub_family=sf,
                mode=mode,
                source_family_id=src,
                paired_family_id=paired,
                gold_answer=gold,
                answer_type=at,
                atomic_answer_candidates=aac,
                variants=variants,
            )
        )
    return out


def load_probe_ready_symbolic_mcq(path: Path, dataset_name: str) -> list[Family]:
    rows = _read_jsonl(path)
    out: list[Family] = []
    for r in rows:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        tf = str(r.get("task_family", ""))
        sf = str(r.get("sub_family", ""))
        fid = str(r.get("family_id", ""))
        mode = str(meta.get("mode", ""))
        src = str(meta.get("source_family_id", ""))
        paired = str(meta.get("paired_family_id", ""))
        gold = str(r.get("gold_answer", "") or "")
        at = str(meta.get("answer_type", ""))
        aac = meta.get("atomic_target_candidates") if isinstance(meta.get("atomic_target_candidates"), list) else None
        if isinstance(aac, list):
            aac = [str(x) for x in aac if isinstance(x, str) and x.strip()]

        nv = r.get("normal_variants")
        variants: dict[str, str] = {}
        if isinstance(nv, dict) and "original" in nv:
            q = (nv.get("original") or {}).get("question")
            if isinstance(q, str) and q.strip():
                variants["original"] = q.strip()

        mcq = r.get("mcq_variants") if isinstance(r.get("mcq_variants"), dict) else {}
        item = mcq.get("symbolic_original") if isinstance(mcq, dict) else None
        mcq_q = None
        mcq_opts = None
        mcq_ci = None
        if isinstance(item, dict):
            q = item.get("question")
            opts = item.get("options")
            ci = item.get("correct_index")
            if isinstance(q, str) and isinstance(opts, list) and all(isinstance(x, str) for x in opts) and isinstance(ci, int):
                mcq_q = q
                mcq_opts = list(opts)
                mcq_ci = ci

        out.append(
            Family(
                dataset=dataset_name,
                family_id=fid,
                task_family=tf,
                sub_family=sf,
                mode=mode,
                source_family_id=src,
                paired_family_id=paired,
                gold_answer=gold,
                answer_type=at,
                atomic_answer_candidates=aac,
                variants=variants,
                mcq_question=mcq_q,
                mcq_options=mcq_opts,
                mcq_correct_index=mcq_ci,
            )
        )
    return out


def load_gpt_test(path: Path, dataset_name: str) -> list[Family]:
    rows = _read_jsonl(path)
    out: list[Family] = []
    for r in rows:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        tf = str(meta.get("task_family", ""))
        mode = str(meta.get("mode", ""))
        src = str(meta.get("source_family_id", ""))
        paired = str(meta.get("paired_family_id", ""))
        gold = str(meta.get("gold_answer", "") or "")
        at = str(meta.get("answer_type", ""))
        # gpt_test 的 atomic_target_candidates 是 label candidates，不作为答案候选
        aac = meta.get("atomic_target_candidates") if isinstance(meta.get("atomic_target_candidates"), list) else None
        if isinstance(aac, list):
            aac2 = [str(x) for x in aac if isinstance(x, str) and x.strip()]
            aac = None if _looks_like_label_candidates(aac2) else aac2
        fid = str(r.get("family_id", ""))
        # sub_family: prefer reasoning/relation type
        sf = str(meta.get("relation_type") or meta.get("reasoning_type") or "")
        vv = r.get("variants") if isinstance(r.get("variants"), dict) else {}
        variants: dict[str, str] = {}
        for k in REQUIRED_VARIANTS:
            v = vv.get(k)
            q = v.get("question") if isinstance(v, dict) else None
            if isinstance(q, str) and q.strip():
                variants[k] = q.strip()
        out.append(
            Family(
                dataset=dataset_name,
                family_id=fid,
                task_family=tf,
                sub_family=sf,
                mode=mode,
                source_family_id=src,
                paired_family_id=paired,
                gold_answer=gold,
                answer_type=at,
                atomic_answer_candidates=aac,
                variants=variants,
            )
        )
    return out


def _looks_like_label_candidates(cands: list[str]) -> bool:
    # gpt_test 的 atomic_target_candidates 是 label 名（带下划线），不是答案候选。
    if not cands:
        return False
    underscore = sum(1 for x in cands if isinstance(x, str) and "_" in x)
    return underscore >= max(1, int(0.6 * len(cands)))


def _extract_distractors_from_text(text: str) -> list[str]:
    """从 wrongclaim/competing prompts 中尽量抽出错误答案字符串（非常保守）。"""
    t = text or ""
    out: list[str] = []
    pats = [
        r"answer is ([^\.\n]+)",
        r"gives ([^\.\n]+)",
        r"one says ([^,\.\n]+)",
        r"another says ([^,\.\n]+)",
    ]
    for p in pats:
        for m in re.finditer(p, t, flags=re.I):
            s = m.group(1).strip().strip("\"'")
            # 去掉尾部说明
            s = re.sub(r"\s*(ignore|rely|answer|what).*$", "", s, flags=re.I).strip()
            if s and len(_norm(s)) >= 1:
                out.append(s)
    # 去重
    dedup: list[str] = []
    seen = set()
    for x in out:
        k = _norm(x)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(x)
    return dedup[:6]


def build_answer_candidates(
    fam: Family,
    global_gold_pool: dict[str, list[str]],
    prefer_k: int = 4,
) -> list[str]:
    gold = (fam.gold_answer or "").strip()
    if not gold:
        return []

    # 1) 优先用数据自带 candidates（probe-ready）
    if fam.atomic_answer_candidates:
        c = [x for x in fam.atomic_answer_candidates if isinstance(x, str) and x.strip()]
        # strict_symbolic：必须都是 token
        if fam.mode == "strict_symbolic" and gold in STRICT_TOKENS:
            # 保守：直接固定 token 集
            base = ["E2", "C1", "V1", "E1"]
            return base if gold == "E2" else [gold] + [x for x in base if x != gold]
        if gold not in c:
            c = [gold] + [x for x in c if _norm(x) != _norm(gold)]
        # 截断到 prefer_k
        return c[:prefer_k]

    # 2) strict_symbolic token 回退
    if fam.mode == "strict_symbolic" and gold in STRICT_TOKENS:
        base = ["E2", "C1", "V1", "E1"]
        return base if gold == "E2" else [gold] + [x for x in base if x != gold]

    # probe-ready: family.metadata.atomic_target_candidates 存在于数据文件里，但本脚本 Family 不带 meta。
    # 因此：优先从 prompt 抽 distractors，再用全局 pool 补足。
    distractors: list[str] = []
    if "wrongclaim_bare" in fam.variants:
        distractors.extend(_extract_distractors_from_text(fam.variants["wrongclaim_bare"]))
    if "competing_claims" in fam.variants:
        distractors.extend(_extract_distractors_from_text(fam.variants["competing_claims"]))

    # filter gold/aliases
    aliases = _gold_aliases(gold)
    d2: list[str] = []
    seen = set([_norm(gold)])
    for d in distractors:
        if _token_contains_any(d, aliases):
            continue
        k = _norm(d)
        if not k or k in seen:
            continue
        seen.add(k)
        d2.append(d)

    # pool fill
    pool = global_gold_pool.get(fam.answer_type or "", [])
    for x in pool:
        if len(d2) >= (prefer_k - 1):
            break
        k = _norm(x)
        if not k or k in seen:
            continue
        if _token_contains_any(x, aliases):
            continue
        seen.add(k)
        d2.append(x)

    cands = [gold] + d2
    return cands[:prefer_k]


def _token_contains_any(text: str, needles: Iterable[str]) -> bool:
    for n in needles:
        if n and _token_contains(text, n):
            return True
    return False


def _batch_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool,
) -> Any:
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )


def _pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


def compute_choice_logprobs(
    model,
    tokenizer,
    prompt: str,
    candidates: list[str],
    batch_size: int = 32,
) -> list[float]:
    """返回每个 candidate 的总 logprob（越大越好）。"""
    if not candidates:
        return []
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cand_ids_list = [tokenizer(c, add_special_tokens=False).input_ids for c in candidates]
    # build full sequences
    seqs: list[list[int]] = [prompt_ids + cids for cids in cand_ids_list]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out_scores: list[float] = [float("-inf")] * len(candidates)

    device = next(model.parameters()).device
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start : start + batch_size]
        ids, mask = _pad_batch(batch, pad_id)
        ids = ids.to(device)
        mask = mask.to(device)
        out = _batch_forward(model, ids, mask, output_hidden_states=False)
        logits = out.logits  # [B, T, V]
        logp = torch.log_softmax(logits, dim=-1)
        for bi in range(ids.shape[0]):
            idx = start + bi
            p_len = len(prompt_ids)
            cids = cand_ids_list[idx]
            # sum logprobs for candidate tokens
            s = 0.0
            ok = True
            for j, tok in enumerate(cids):
                pos = p_len + j
                if pos == 0:
                    ok = False
                    break
                if pos >= ids.shape[1]:
                    ok = False
                    break
                s += float(logp[bi, pos - 1, tok].item())
            out_scores[idx] = s if ok else float("-inf")
    return out_scores


def extract_last_token_hidden(model, tokenizer, prompts: list[str], batch_size: int = 16) -> list[np.ndarray]:
    """对每个 prompt 抽取 last-nonpad token 的所有层 hidden states: [L, H]."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch_texts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = _batch_forward(model, input_ids, attn, output_hidden_states=True)
        hs = out.hidden_states  # tuple: emb + layers
        # drop embedding
        layers = hs[1:]
        # last non-pad index per row
        last_idx = attn.sum(dim=1) - 1
        for bi in range(input_ids.shape[0]):
            li = int(last_idx[bi].item())
            vecs = [layers[l][bi, li, :].detach().to(torch.float16).cpu() for l in range(len(layers))]
            stacked = torch.stack(vecs, dim=0).numpy()
            results.append(stacked)
    return results


def build_mcq_prompt(q: str, options: list[str]) -> tuple[str, str]:
    """返回 (final_input_prompt, pre_answer_prompt)。"""
    letters = ["A", "B", "C", "D"]
    lines = [q.rstrip()]
    for i, opt in enumerate(options[:4]):
        lines.append(f"{letters[i]}. {opt}")
    final_prompt = "\n".join(lines)
    pre_prompt = final_prompt + "\nAnswer:"
    return final_prompt, pre_prompt


def build_pre_answer_prompt(prompt: str) -> str:
    p = prompt.rstrip()
    if re.search(r"\bAnswer:\s*$", p):
        return p
    return p + "\nAnswer:"


@dataclass
class VariantResult:
    score: float
    pred: str
    gold: str
    candidates: list[str]
    logprobs: list[float]


def score_dataset(
    families: list[Family],
    model,
    tokenizer,
    cache_dir: Path,
    use_mcq: bool,
    batch_size: int = 32,
) -> dict[str, dict[str, VariantResult]]:
    """返回: family_id -> variant -> VariantResult。"""
    _ensure_dir(cache_dir)
    cache_path = cache_dir / ("scores_mcq_v2.json" if use_mcq else "scores_v2.json")
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        out: dict[str, dict[str, VariantResult]] = {}
        for fid, vd in raw.items():
            out[fid] = {}
            for vt, rr in vd.items():
                out[fid][vt] = VariantResult(
                    score=float(rr["score"]),
                    pred=str(rr["pred"]),
                    gold=str(rr["gold"]),
                    candidates=list(rr["candidates"]),
                    logprobs=list(rr["logprobs"]),
                )
        return out

    # global pool for distractor completion
    pool: dict[str, list[str]] = defaultdict(list)
    for f in families:
        if f.gold_answer:
            pool[f.answer_type or ""].append(f.gold_answer)

    # build all scoring jobs
    jobs: list[tuple[str, str, str, list[str], str]] = []
    # (family_id, variant, prompt, candidates, gold)
    for f in families:
        if use_mcq:
            if f.mcq_question and f.mcq_options and isinstance(f.mcq_correct_index, int):
                final_p, pre_p = build_mcq_prompt(f.mcq_question, f.mcq_options)
                # score uses pre_p as prompt
                gold = f.mcq_options[f.mcq_correct_index]
                cands = list(f.mcq_options)
                jobs.append((f.family_id, "mcq_original", pre_p, cands, gold))
        else:
            for vt, prompt in f.variants.items():
                cands = build_answer_candidates(f, pool, prefer_k=4)
                if not cands:
                    continue
                gold = f.gold_answer
                if gold not in cands:
                    cands = [gold] + [x for x in cands if x != gold]
                    cands = cands[:4]
                jobs.append((f.family_id, vt, prompt, cands, gold))

    # run scoring (batch by candidates within job; still batched at model level via compute_choice_logprobs)
    out: dict[str, dict[str, VariantResult]] = defaultdict(dict)
    for fid, vt, prompt, cands, gold in jobs:
        lps = compute_choice_logprobs(model, tokenizer, build_pre_answer_prompt(prompt), cands, batch_size=batch_size)
        if not lps or all(math.isinf(x) for x in lps):
            pred = ""
            score = 0.0
        else:
            pi = int(np.argmax(np.array(lps)))
            pred = cands[pi]
            score = 1.0 if _norm(pred) == _norm(gold) else 0.0
        out[fid][vt] = VariantResult(score=score, pred=pred, gold=gold, candidates=cands, logprobs=lps)

    # persist
    serial: dict[str, Any] = {}
    for fid, vd in out.items():
        serial[fid] = {}
        for vt, r in vd.items():
            serial[fid][vt] = {
                "score": r.score,
                "pred": r.pred,
                "gold": r.gold,
                "candidates": r.candidates,
                "logprobs": r.logprobs,
            }
    _write_json(serial, cache_path)
    return out


def cache_hidden_states(
    families: list[Family],
    model,
    tokenizer,
    cache_root: Path,
    dataset_tag: str,
    position_types: list[str],
    use_mcq: bool,
    batch_size: int = 16,
) -> dict[tuple[str, str], str]:
    """缓存 hidden states。

    返回：(family_id, variant) -> relative_path（.pt 文件，内含 final_input/pre_answer）。
    """
    root = cache_root / dataset_tag
    _ensure_dir(root)

    mapping: dict[tuple[str, str], str] = {}
    prompts_final: list[str] = []
    prompts_pre: list[str] = []
    keys: list[tuple[str, str, Path]] = []

    def add_job(fid: str, vt: str, final_p: str, pre_p: str, out_path: Path) -> None:
        prompts_final.append(final_p)
        prompts_pre.append(pre_p)
        keys.append((fid, vt, out_path))

    for f in families:
        if use_mcq:
            if not (f.mcq_question and f.mcq_options):
                continue
            # 仅对 strict_symbolic MCQ 原题做缓存
            vt = "mcq_original"
            rel = Path(f"{f.family_id}__{vt}.pt")
            out_path = root / rel
            mapping[(f.family_id, vt)] = str(Path(dataset_tag) / rel)
            if out_path.exists():
                continue
            final_p, pre_p = build_mcq_prompt(f.mcq_question, f.mcq_options)
            add_job(f.family_id, vt, final_p, pre_p, out_path)
        else:
            for vt, prompt in f.variants.items():
                rel = Path(f"{f.family_id}__{vt}.pt")
                out_path = root / rel
                mapping[(f.family_id, vt)] = str(Path(dataset_tag) / rel)
                if out_path.exists():
                    continue
                final_p = prompt
                pre_p = build_pre_answer_prompt(prompt)
                add_job(f.family_id, vt, final_p, pre_p, out_path)

    if not keys:
        return mapping

    # batch extract
    hs_final = extract_last_token_hidden(model, tokenizer, prompts_final, batch_size=batch_size)
    hs_pre = extract_last_token_hidden(model, tokenizer, prompts_pre, batch_size=batch_size)

    for (fid, vt, out_path), h1, h2 in zip(keys, hs_final, hs_pre):
        payload = {
            "final_input": torch.from_numpy(h1),
            "pre_answer": torch.from_numpy(h2),
        }
        _ensure_dir(out_path.parent)
        torch.save(payload, out_path)
    return mapping


def build_metadata_rows(
    families: list[Family],
    scores: dict[str, dict[str, VariantResult]],
    hs_paths: dict[tuple[str, str], str],
    model_name: str,
    dataset_tag: str,
    position_key: str,
    use_mcq: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in families:
        if use_mcq:
            vt = "mcq_original"
            if f.family_id not in scores or vt not in scores[f.family_id]:
                continue
            rr = scores[f.family_id][vt]
            p = hs_paths.get((f.family_id, vt))
            if not p:
                continue
            uid = f"{dataset_tag}::{f.family_id}::{vt}::{position_key}"
            rows.append(
                {
                    "uid": uid,
                    "family_id": f.family_id,
                    "task_family": f.task_family,
                    "sub_family": f.sub_family,
                    "variant": "original",
                    "score": float(rr.score),
                    "model_name": model_name,
                    "split": "",
                    "hidden_state_path": p,
                    "hidden_state_key": position_key,
                    "layer_index": None,
                    "position_type": position_key,
                    "dataset": dataset_tag,
                    "mode": f.mode,
                    "source_family_id": f.source_family_id,
                    "paired_family_id": f.paired_family_id,
                }
            )
        else:
            for vt in REQUIRED_VARIANTS:
                if vt not in f.variants:
                    continue
                rr = scores.get(f.family_id, {}).get(vt)
                if rr is None:
                    continue
                p = hs_paths.get((f.family_id, vt))
                if not p:
                    continue
                uid = f"{dataset_tag}::{f.family_id}::{vt}::{position_key}"
                rows.append(
                    {
                        "uid": uid,
                        "family_id": f.family_id,
                        "task_family": f.task_family,
                        "sub_family": f.sub_family,
                        "variant": vt,
                        "score": float(rr.score),
                        "model_name": model_name,
                        "split": "",
                        "hidden_state_path": p,
                        "hidden_state_key": position_key,
                        "layer_index": None,
                        "position_type": position_key,
                        "dataset": dataset_tag,
                        "mode": f.mode,
                        "source_family_id": f.source_family_id,
                        "paired_family_id": f.paired_family_id,
                    }
                )
    return rows


def family_diagnostics(families: list[Family], scores: dict[str, dict[str, VariantResult]]) -> dict[str, Any]:
    per_family: dict[str, Any] = {}
    for f in families:
        vd = scores.get(f.family_id, {})
        sv = {k: float(v.score) for k, v in vd.items()}
        vals = list(sv.values())
        if not vals:
            kind = "unscored"
        else:
            all_same = all(abs(x - vals[0]) < 1e-9 for x in vals)
            if all_same and vals[0] >= 0.999:
                kind = "ceiling_family"
            elif all_same and vals[0] <= 1e-9:
                kind = "floor_family"
            elif all_same:
                kind = "flat_family"
            else:
                kind = "informative_family"
        per_family[f.family_id] = {
            "task_family": f.task_family,
            "sub_family": f.sub_family,
            "mode": f.mode,
            "source_family_id": f.source_family_id,
            "paired_family_id": f.paired_family_id,
            "scores": sv,
            "diagnostic": kind,
        }
    return {
        "n_families": len(families),
        "diagnostic_counts": dict(Counter(v["diagnostic"] for v in per_family.values())),
        "families": per_family,
    }


def compute_signals_for_family(score_map: dict[str, float]) -> dict[str, float | None]:
    def get(v: str) -> float | None:
        return score_map.get(v)

    orig = get("original")
    if orig is None:
        return {}

    premise = get("premise")
    prem_rem = get("premise_removal")
    wrong = get("wrongclaim_bare")
    comp = get("competing_claims")
    para = get("paraphrase")
    hi = get("highlight")
    sc1 = get("scaffold_1")
    sc2 = get("scaffold_2")
    sub = get("substitution")

    scaffold_best = None
    if sc1 is not None or sc2 is not None:
        scaffold_best = max([x for x in [sc1, sc2] if x is not None])

    out: dict[str, float | None] = {
        "premise_gain": (premise - orig) if premise is not None else None,
        "removal_drop": (orig - prem_rem) if prem_rem is not None else None,
        "wrongclaim_drop": (orig - wrong) if wrong is not None else None,
        "scaffold_gain": (scaffold_best - orig) if scaffold_best is not None else None,
        "paraphrase_drop": (orig - para) if para is not None else None,
        "conflict_drop": (orig - comp) if comp is not None else None,
        "highlight_delta": (hi - orig) if hi is not None else None,
        "substitution_drop": (orig - sub) if sub is not None else None,
        "orig": orig,
    }
    return out


def extreme_bin_labels(
    signals: dict[str, float],
    top_q: float,
) -> tuple[float, float, dict[str, int | None]]:
    vals = np.array(list(signals.values()), dtype=np.float32)
    pos_thr = float(np.quantile(vals, 1 - top_q))
    neg_thr = float(np.quantile(vals, top_q))
    labels: dict[str, int | None] = {}
    for fid, v in signals.items():
        if v >= pos_thr:
            labels[fid] = 1
        elif v <= neg_thr:
            labels[fid] = 0
        else:
            labels[fid] = None
    return pos_thr, neg_thr, labels


def build_atomic_labels(
    families: list[Family],
    scores: dict[str, dict[str, VariantResult]],
    top_q: float,
    dataset_tag: str,
) -> dict[str, Any]:
    # per-family signal table
    fam_signals: dict[str, dict[str, float | None]] = {}
    for f in families:
        sm = {k: float(v.score) for k, v in scores.get(f.family_id, {}).items()}
        fam_signals[f.family_id] = compute_signals_for_family(sm)

    # label -> signal_name mapping
    label_to_signal = {
        "premise_sensitive": "premise_gain",
        "removal_dependent": "removal_drop",
        "wrong_claim_susceptible": "wrongclaim_drop",
        "scaffold_sensitive": "scaffold_gain",
        "paraphrase_fragile": "paraphrase_drop",
        "substitution_fragile": "substitution_drop",
        # extra
        "access_sensitive": "removal_drop",
        "localization_sensitive": "highlight_delta",
        "integration_sensitive": "full_support_bundle_gain",  # likely missing
        "decomposition_sensitive": "scaffold_gain",
        "order_sensitive": "order_delta",  # missing
        "intermediate_state_sensitive": "cot_delta",  # missing
        "cue_susceptible": "wrongclaim_drop",
        "authority_susceptible": "authority_drop",  # missing
        "conflict_resolution_weak": "conflict_drop",
        "lexical_fragile": "paraphrase_drop",
        "terminology_fragile": "terminology_drop",  # missing
        "structure_misaligned": "substitution_drop",
    }

    labels_out: dict[str, Any] = {}
    unusable: dict[str, str] = {}
    for lab, sig_name in label_to_signal.items():
        # collect available signals
        sigs: dict[str, float] = {}
        for fid, sd in fam_signals.items():
            v = sd.get(sig_name)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                sigs[fid] = float(v)
        if len(sigs) < 8:
            unusable[lab] = f"signal '{sig_name}' 不可用或样本过少 (n={len(sigs)})"
            continue
        pos_thr, neg_thr, y = extreme_bin_labels(sigs, top_q)
        pos = sum(1 for v in y.values() if v == 1)
        neg = sum(1 for v in y.values() if v == 0)
        mid = sum(1 for v in y.values() if v is None)
        if pos == 0 or neg == 0:
            unusable[lab] = f"extreme-bin 退化 (pos={pos}, neg={neg}, mid={mid})"
            continue
        labels_out[lab] = {
            "signal": sig_name,
            "thresholds": {"pos_ge": pos_thr, "neg_le": neg_thr, "top_quantile": top_q},
            "counts": {"pos": pos, "neg": neg, "dropped_mid": mid, "total_with_signal": len(sigs)},
            "labels": {fid: (None if yv is None else int(yv)) for fid, yv in y.items()},
        }

    return {
        "dataset": dataset_tag,
        "top_quantile": top_q,
        "labels": labels_out,
        "unusable_reasons": unusable,
        "signals": fam_signals,
    }


def stratified_random_split(family_ids: list[str], strata: dict[str, str], test_size: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.RandomState(seed)
    by = defaultdict(list)
    for fid in family_ids:
        by[strata.get(fid, "")].append(fid)
    train: list[str] = []
    test: list[str] = []
    for _, ids in by.items():
        ids2 = list(ids)
        rng.shuffle(ids2)
        n_test = max(1, int(round(len(ids2) * test_size)))
        test.extend(ids2[:n_test])
        train.extend(ids2[n_test:])
    return sorted(train), sorted(test)


def held_out_subfamily_splits(families: dict[str, dict[str, Any]]) -> list[tuple[str, list[str], list[str]]]:
    # returns list of (split_name, train_ids, test_ids)
    by_sf = defaultdict(list)
    for fid, info in families.items():
        by_sf[str(info.get("sub_family", ""))].append(fid)
    splits: list[tuple[str, list[str], list[str]]] = []
    all_ids = set(families.keys())
    for sf, test_ids in sorted(by_sf.items(), key=lambda kv: kv[0]):
        test = sorted(test_ids)
        train = sorted(list(all_ids - set(test_ids)))
        if len(test) >= 4 and len(train) >= 8:
            splits.append((f"held_out_sub_family::{sf}", train, test))
    return splits


def block_level_splits(families: dict[str, dict[str, Any]]) -> list[tuple[str, list[str], list[str]]]:
    by_tf = defaultdict(list)
    for fid, info in families.items():
        by_tf[str(info.get("task_family", ""))].append(fid)

    def ids(tf_list: list[str]) -> list[str]:
        out: list[str] = []
        for tf in tf_list:
            out.extend(by_tf.get(tf, []))
        return sorted(out)

    setups = [
        ("block::train_KB_RB__test_Hybrid", ["KB", "RB"], ["Hybrid"]),
        ("block::train_KB__test_RB", ["KB"], ["RB"]),
        ("block::train_KB__test_Hybrid", ["KB"], ["Hybrid"]),
        ("block::train_RB__test_Hybrid", ["RB"], ["Hybrid"]),
    ]
    out: list[tuple[str, list[str], list[str]]] = []
    for name, tr, te in setups:
        tr_ids = ids(tr)
        te_ids = ids(te)
        if len(tr_ids) >= 8 and len(te_ids) >= 4:
            out.append((name, tr_ids, te_ids))
    return out


def realization_splits(families: dict[str, dict[str, Any]]) -> list[tuple[str, list[str], list[str]]]:
    nat = sorted([fid for fid, info in families.items() if str(info.get("mode")) == "naturalized"])
    sym = sorted([fid for fid, info in families.items() if str(info.get("mode")) in {"strict_symbolic", "symbolic"}])
    out: list[tuple[str, list[str], list[str]]] = []
    if len(nat) >= 8 and len(sym) >= 8:
        out.append(("realization::train_natural__test_symbolic", nat, sym))
        out.append(("realization::train_symbolic__test_natural", sym, nat))
    return out


def _resolve_layers(store: HiddenStateStore, sample_path: str, key: str) -> list[int]:
    # load one tensor
    arr = store._load_file(sample_path, key)  # type: ignore[attr-defined]
    if arr.ndim == 1:
        return [0]
    return list(range(arr.shape[0]))


def run_layerwise_probe(
    *,
    dataset_tag: str,
    out_root: Path,
    store: HiddenStateStore,
    rows: list[dict[str, Any]],
    labels: dict[str, int],
    train_ids: list[str],
    test_ids: list[str],
    variant: str,
    delta_variant: str | None,
    seeds: list[int],
    split_name: str,
    position_key: str,
    feature_mode: str,
    label_name: str,
    task_is_multiclass: bool,
) -> dict[str, Any]:
    # Build per-family row lookup for the needed variant
    by_fid_variant: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by_fid_variant[(str(r["family_id"]), str(r["variant"]))] = r

    # Resolve layers
    sample_row = None
    for r in rows:
        if str(r.get("variant")) == "original":
            sample_row = r
            break
    if sample_row is None:
        return {"status": "unusable", "reason": "missing original rows"}
    layers = _resolve_layers(store, str(sample_row["hidden_state_path"]), position_key)

    # Prepare numpy features per layer
    def vec(fid: str, vt: str, layer: int) -> np.ndarray | None:
        rr = by_fid_variant.get((fid, vt))
        if rr is None:
            return None
        class _Row:
            hidden_state_path = rr["hidden_state_path"]
            hidden_state_key = rr.get("hidden_state_key")
            uid = rr.get("uid")
            layer_index = None
            position_type = rr.get("position_type")

        try:
            return store.get(_Row, layer)
        except Exception:
            return None

    def delta(fid: str, vt: str, layer: int) -> np.ndarray | None:
        v1 = vec(fid, vt, layer)
        v0 = vec(fid, "original", layer)
        if v1 is None or v0 is None:
            return None
        return v1 - v0

    # Collect fids with labels and needed features
    def build_xy(ids: list[str], layer: int, use_delta: bool, dv: str | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        X: list[np.ndarray] = []
        y: list[int] = []
        kept: list[str] = []
        for fid in ids:
            if fid not in labels:
                continue
            if use_delta:
                if not dv:
                    continue
                v = delta(fid, dv, layer)
            else:
                v = vec(fid, variant, layer)
            if v is None:
                continue
            X.append(v.astype(np.float32))
            y.append(int(labels[fid]))
            kept.append(fid)
        if not X:
            raise ValueError("no features")
        return np.stack(X, 0), np.array(y, dtype=np.int64), kept

    # Fit scaler on train
    from sklearn.preprocessing import StandardScaler

    layer_results: dict[int, list[dict[str, Any]]] = defaultdict(list)
    best_pick = None

    out_dir = out_root / dataset_tag / position_key / feature_mode / label_name / split_name

    # Resume/skip: if this exact run already produced a summary, reuse it.
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            # Only skip successful runs. "unusable" may become usable after bugfixes.
            if isinstance(prev, dict) and prev.get("status") == "ok":
                return prev
        except Exception:
            pass

    _ensure_dir(out_dir)

    per_layer_rows_json: list[dict[str, Any]] = []
    pred_rows_best: list[dict[str, Any]] = []
    best_layer = None
    best_score = -1.0

    for layer in layers:
        try:
            X_tr, y_tr, kept_tr = build_xy(train_ids, layer, use_delta=(delta_variant is not None), dv=delta_variant)
            X_te, y_te, kept_te = build_xy(test_ids, layer, use_delta=(delta_variant is not None), dv=delta_variant)
        except Exception:
            continue

        # Degenerate splits: cannot fit a classifier with a single training class.
        # Also skip single-class test splits (macro-F1 becomes uninformative).
        if len(np.unique(y_tr)) < 2:
            continue
        if len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        seed_results = run_probe_with_seeds(
            X_tr,
            y_tr,
            X_te,
            y_te,
            seeds=seeds,
            probe_type="logistic",
            class_weight="balanced",
        )
        for sr in seed_results:
            sr = dict(sr)
            sr["layer"] = layer
            layer_results[layer].append(sr)

        # aggregate for best layer selection
        pl = per_layer_metrics({layer: seed_results})
        if not pl:
            continue
        row = pl[0]
        row["layer"] = layer
        per_layer_rows_json.append(row)
        score = float(row.get("macro_f1_mean", -1.0))
        if score > best_score:
            best_score = score
            best_layer = layer
            best_pick = seed_results
            # cache best preds
            pred_rows_best = []
            for sr in seed_results:
                for fid, yt, yp in zip(kept_te, sr["y_true"], sr["y_pred"]):
                    pred_rows_best.append({"family_id": fid, "y_true": int(yt), "y_pred": int(yp), "seed": int(sr["seed"]), "layer": layer})

    if best_layer is None or best_pick is None:
        summary = {
            "status": "unusable",
            "reason": "no valid layers/features (可能缺 hidden states 或标签退化)",
        }
        _write_json(summary, out_dir / "summary.json")
        return summary

    # write outputs
    per_layer_rows = per_layer_metrics(layer_results)
    _write_csv(per_layer_rows, out_dir / "per_layer_metrics.csv")
    _write_json(per_layer_rows_json, out_dir / "per_layer_metrics.json")
    best_layer_metrics = {
        "best_layer": int(best_layer),
        "best_macro_f1_mean": float(best_score),
        "n_layers": len(layers),
    }
    _write_json(best_layer_metrics, out_dir / "best_layer_metrics.json")
    _write_csv(pred_rows_best, out_dir / "predictions.csv")

    summary = {
        "status": "ok",
        "dataset": dataset_tag,
        "label": label_name,
        "split": split_name,
        "position": position_key,
        "feature_mode": feature_mode,
        "variant": variant,
        "delta_variant": delta_variant,
        "best_layer": int(best_layer),
        "best_macro_f1_mean": float(best_score),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "seeds": list(seeds),
    }
    _write_json(summary, out_dir / "summary.json")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型路径，例如 /opt/tiger/coding-agent-synth-data/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--top_quantile", type=float, default=0.2)
    ap.add_argument("--cache_dir", type=Path, default=Path("cache/first_pass"))
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/first_pass_probing"))
    ap.add_argument(
        "--datasets",
        default="probe_ready,gpt_test",
        help="Comma-separated dataset tags to probe: probe_ready,gpt_test (probe_ready_mcq is used for transfer).",
    )
    ap.add_argument(
        "--positions",
        default="final_input,pre_answer",
        help="Comma-separated hidden-state positions to probe: final_input,pre_answer",
    )
    ap.add_argument(
        "--no_transfers",
        action="store_true",
        help="Skip MCQ transfer and cross-dataset transfer (useful for faster partial runs).",
    )
    ap.add_argument(
        "--no_report",
        action="store_true",
        help="Skip writing reports/probing_comparison_report.md (useful for faster partial runs).",
    )
    ap.add_argument(
        "--label_set",
        default="all",
        choices=["all", "primary"],
        help="Which atomic labels to probe: 'all' (PRIMARY+EXTRA) or 'primary' only.",
    )
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    # Load datasets
    probe_ready = load_probe_ready_paired(Path("data/probe_ready_180_paired.jsonl"), "probe_ready")
    probe_ready_mcq_sym = load_probe_ready_symbolic_mcq(Path("data/probe_ready_90_symbolic_mcq.jsonl"), "probe_ready_mcq")
    gpt_test = load_gpt_test(Path("data/gpt_test.jsonl"), "gpt_test")

    # Load model
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(args.device)
    model.eval()

    # ========== scoring + hidden state caching ==========
    cache_root = args.cache_dir
    hs_root = cache_root / "hidden_states" / "qwen3_8b"
    score_root = cache_root / "scores" / "qwen3_8b"
    meta_root = cache_root / "metadata" / "qwen3_8b"

    # Non-MCQ: probe_ready + gpt_test
    scores_probe_ready = score_dataset(probe_ready, model, tokenizer, score_root / "probe_ready", use_mcq=False)
    scores_gpt = score_dataset(gpt_test, model, tokenizer, score_root / "gpt_test", use_mcq=False)
    # MCQ: only strict_symbolic MCQ families
    scores_probe_ready_mcq = score_dataset(probe_ready_mcq_sym, model, tokenizer, score_root / "probe_ready_mcq", use_mcq=True)

    # cache hidden states
    hs_paths_probe_ready = cache_hidden_states(
        probe_ready,
        model,
        tokenizer,
        hs_root,
        dataset_tag="probe_ready",
        position_types=["final_input", "pre_answer"],
        use_mcq=False,
    )
    hs_paths_gpt = cache_hidden_states(
        gpt_test,
        model,
        tokenizer,
        hs_root,
        dataset_tag="gpt_test",
        position_types=["final_input", "pre_answer"],
        use_mcq=False,
    )
    hs_paths_probe_ready_mcq = cache_hidden_states(
        probe_ready_mcq_sym,
        model,
        tokenizer,
        hs_root,
        dataset_tag="probe_ready_mcq",
        position_types=["final_input", "pre_answer"],
        use_mcq=True,
    )

    # Build metadata rows (two position types separately)
    for pos in ("final_input", "pre_answer"):
        rows_pr = build_metadata_rows(
            probe_ready,
            scores_probe_ready,
            hs_paths_probe_ready,
            model_name=args.model,
            dataset_tag="probe_ready",
            position_key=pos,
            use_mcq=False,
        )
        _write_jsonl(rows_pr, meta_root / f"probe_ready.{pos}.jsonl")

        rows_gt = build_metadata_rows(
            gpt_test,
            scores_gpt,
            hs_paths_gpt,
            model_name=args.model,
            dataset_tag="gpt_test",
            position_key=pos,
            use_mcq=False,
        )
        _write_jsonl(rows_gt, meta_root / f"gpt_test.{pos}.jsonl")

        rows_mcq = build_metadata_rows(
            probe_ready_mcq_sym,
            scores_probe_ready_mcq,
            hs_paths_probe_ready_mcq,
            model_name=args.model,
            dataset_tag="probe_ready_mcq",
            position_key=pos,
            use_mcq=True,
        )
        _write_jsonl(rows_mcq, meta_root / f"probe_ready_mcq.{pos}.jsonl")

    # ========== diagnostics ==========
    diag_pr = family_diagnostics(probe_ready, scores_probe_ready)
    diag_gt = family_diagnostics(gpt_test, scores_gpt)
    _write_json(diag_pr, Path("family_diagnostics_probe_ready.json"))
    _write_json(diag_gt, Path("family_diagnostics_gpt_test.json"))

    # dataset comparison summary (distributions)
    def dist(fams: list[Family]) -> dict[str, Any]:
        tf = Counter(f.task_family for f in fams)
        sf = Counter(f.sub_family for f in fams)
        mode = Counter(f.mode for f in fams)
        return {
            "n_families": len(fams),
            "task_family": dict(tf),
            "sub_family": dict(sf),
            "mode": dict(mode),
            "variant_coverage": dict(
                Counter(v for f in fams for v in f.variants.keys())
            ),
        }

    comp_summary = {
        "model": args.model,
        "probe_ready": dist(probe_ready),
        "gpt_test": dist(gpt_test),
        "probe_ready_mcq": {
            "n_families": len(probe_ready_mcq_sym),
            "task_family": dict(Counter(f.task_family for f in probe_ready_mcq_sym)),
            "sub_family": dict(Counter(f.sub_family for f in probe_ready_mcq_sym)),
            "mode": dict(Counter(f.mode for f in probe_ready_mcq_sym)),
            "has_mcq": sum(1 for f in probe_ready_mcq_sym if f.mcq_options),
        },
    }
    _write_json(comp_summary, Path("dataset_comparison_summary.json"))

    # ========== labels ==========
    labels_pr = build_atomic_labels(probe_ready, scores_probe_ready, top_q=args.top_quantile, dataset_tag="probe_ready")
    labels_gt = build_atomic_labels(gpt_test, scores_gpt, top_q=args.top_quantile, dataset_tag="gpt_test")
    _write_json(labels_pr, Path("atomic_labels_probe_ready.json"))
    _write_json(labels_gt, Path("atomic_labels_gpt_test.json"))

    # ========== probing ==========
    out_root = args.out_dir
    _ensure_dir(out_root)

    def build_family_info(fams: list[Family]) -> dict[str, dict[str, Any]]:
        return {
            f.family_id: {
                "task_family": f.task_family,
                "sub_family": f.sub_family,
                "mode": f.mode,
                "source_family_id": f.source_family_id,
            }
            for f in fams
        }

    info_pr = build_family_info(probe_ready)
    info_gt = build_family_info(gpt_test)

    # store
    store = HiddenStateStore(str(hs_root))

    def run_all_for_dataset(dataset_tag: str, meta_path: Path, fam_info: dict[str, dict[str, Any]], labels_blob: dict[str, Any]) -> list[dict[str, Any]]:
        rows = _read_jsonl(meta_path)
        # baseline task label probe: KB/RB/Hybrid
        strata = {fid: fam_info[fid]["task_family"] for fid in fam_info}
        all_ids = sorted(fam_info.keys())
        train_ids, test_ids = stratified_random_split(all_ids, strata, args.test_size, seed=1337)

        # encode task_family labels
        tf_map = {"KB": 0, "RB": 1, "Hybrid": 2}
        y_task = {fid: tf_map.get(str(fam_info[fid]["task_family"]), -1) for fid in all_ids}
        y_task = {fid: v for fid, v in y_task.items() if v >= 0}

        results: list[dict[str, Any]] = []
        results.append(
            run_layerwise_probe(
                dataset_tag=dataset_tag,
                out_root=out_root,
                store=store,
                rows=rows,
                labels=y_task,
                train_ids=train_ids,
                test_ids=test_ids,
                variant="original",
                delta_variant=None,
                seeds=seeds,
                split_name="random_family_split",
                position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                feature_mode="absolute::h_original",
                label_name="task_family_baseline",
                task_is_multiclass=True,
            )
        )

        # baseline realization transfer
        for split_name, tr, te in realization_splits(fam_info):
            results.append(
                run_layerwise_probe(
                    dataset_tag=dataset_tag,
                    out_root=out_root,
                    store=store,
                    rows=rows,
                    labels=y_task,
                    train_ids=tr,
                    test_ids=te,
                    variant="original",
                    delta_variant=None,
                    seeds=seeds,
                    split_name=split_name,
                    position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                    feature_mode="absolute::h_original",
                    label_name="task_family_baseline",
                    task_is_multiclass=True,
                )
            )

        # label probing
        label_defs = labels_blob.get("labels", {})
        all_labels = PRIMARY_LABELS if args.label_set == "primary" else (PRIMARY_LABELS + EXTRA_LABELS)
        for lab in all_labels:
            if lab not in label_defs:
                # record unusable
                out_dir = out_root / dataset_tag / (rows[0]["hidden_state_key"] if rows else "") / "absolute::h_original" / lab / "random_family_split"
                _ensure_dir(out_dir)
                _write_json({"status": "unusable", "reason": labels_blob.get("unusable_reasons", {}).get(lab, "label 不可用")}, out_dir / "summary.json")
                continue
            y_raw = label_defs[lab]["labels"]
            # filter None
            y = {fid: int(v) for fid, v in y_raw.items() if v is not None and fid in fam_info}
            if len(y) < 8:
                continue
            # use only labeled ids
            ids = sorted(y.keys())
            strata2 = {fid: fam_info[fid]["task_family"] for fid in ids}
            tr, te = stratified_random_split(ids, strata2, args.test_size, seed=1337)

            # Absolute (h_original)
            results.append(
                run_layerwise_probe(
                    dataset_tag=dataset_tag,
                    out_root=out_root,
                    store=store,
                    rows=rows,
                    labels=y,
                    train_ids=tr,
                    test_ids=te,
                    variant="original",
                    delta_variant=None,
                    seeds=seeds,
                    split_name="random_family_split",
                    position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                    feature_mode="absolute::h_original",
                    label_name=lab,
                    task_is_multiclass=False,
                )
            )

            # Delta variants
            delta_map = {
                "premise_sensitive": "premise",
                "removal_dependent": "premise_removal",
                "wrong_claim_susceptible": "wrongclaim_bare",
                "scaffold_sensitive": "scaffold_2",
                "paraphrase_fragile": "paraphrase",
                "substitution_fragile": "substitution",
                "conflict_resolution_weak": "competing_claims",
                "localization_sensitive": "highlight",
                "cue_susceptible": "wrongclaim_bare",
                "lexical_fragile": "paraphrase",
                "decomposition_sensitive": "scaffold_2",
                "structure_misaligned": "substitution",
                "access_sensitive": "premise_removal",
            }
            dv = delta_map.get(lab)
            if dv and any(r.get("variant") == dv for r in rows):
                results.append(
                    run_layerwise_probe(
                        dataset_tag=dataset_tag,
                        out_root=out_root,
                        store=store,
                        rows=rows,
                        labels=y,
                        train_ids=tr,
                        test_ids=te,
                        variant="original",
                        delta_variant=dv,
                        seeds=seeds,
                        split_name="random_family_split",
                        position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                        feature_mode=f"delta::{dv}-original",
                        label_name=lab,
                        task_is_multiclass=False,
                    )
                )

            # ---- additional splits (family-level only) ----
            # held-out sub_family：只对 PRIMARY_LABELS 跑 absolute（避免组合爆炸）
            if lab in PRIMARY_LABELS:
                for split_name2, tr2, te2 in held_out_subfamily_splits({fid: fam_info[fid] for fid in ids}):
                    results.append(
                        run_layerwise_probe(
                            dataset_tag=dataset_tag,
                            out_root=out_root,
                            store=store,
                            rows=rows,
                            labels=y,
                            train_ids=tr2,
                            test_ids=te2,
                            variant="original",
                            delta_variant=None,
                            seeds=seeds,
                            split_name=split_name2,
                            position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                            feature_mode="absolute::h_original",
                            label_name=lab,
                            task_is_multiclass=False,
                        )
                    )

            # block-level splits：只跑 absolute
            for split_name2, tr2, te2 in block_level_splits({fid: fam_info[fid] for fid in ids}):
                results.append(
                    run_layerwise_probe(
                        dataset_tag=dataset_tag,
                        out_root=out_root,
                        store=store,
                        rows=rows,
                        labels=y,
                        train_ids=tr2,
                        test_ids=te2,
                        variant="original",
                        delta_variant=None,
                        seeds=seeds,
                        split_name=split_name2,
                        position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                        feature_mode="absolute::h_original",
                        label_name=lab,
                        task_is_multiclass=False,
                    )
                )

            # realization transfer：跑 absolute + delta（delta 满足“必须实际跑一版”要求）
            for split_name2, tr2, te2 in realization_splits({fid: fam_info[fid] for fid in ids}):
                results.append(
                    run_layerwise_probe(
                        dataset_tag=dataset_tag,
                        out_root=out_root,
                        store=store,
                        rows=rows,
                        labels=y,
                        train_ids=tr2,
                        test_ids=te2,
                        variant="original",
                        delta_variant=None,
                        seeds=seeds,
                        split_name=split_name2,
                        position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                        feature_mode="absolute::h_original",
                        label_name=lab,
                        task_is_multiclass=False,
                    )
                )
                if dv and any(r.get("variant") == dv for r in rows):
                    # Delta transfer is expensive; keep it for PRIMARY_LABELS only.
                    if lab in PRIMARY_LABELS:
                        results.append(
                            run_layerwise_probe(
                                dataset_tag=dataset_tag,
                                out_root=out_root,
                                store=store,
                                rows=rows,
                                labels=y,
                                train_ids=tr2,
                                test_ids=te2,
                                variant="original",
                                delta_variant=dv,
                                seeds=seeds,
                                split_name=split_name2,
                                position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                                feature_mode=f"delta::{dv}-original",
                                label_name=lab,
                                task_is_multiclass=False,
                            )
                        )

        # held-out sub_family
        for split_name, tr, te in held_out_subfamily_splits(fam_info):
            # baseline only (task probe)
            results.append(
                run_layerwise_probe(
                    dataset_tag=dataset_tag,
                    out_root=out_root,
                    store=store,
                    rows=rows,
                    labels=y_task,
                    train_ids=tr,
                    test_ids=te,
                    variant="original",
                    delta_variant=None,
                    seeds=seeds,
                    split_name=split_name,
                    position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                    feature_mode="absolute::h_original",
                    label_name="task_family_baseline",
                    task_is_multiclass=True,
                )
            )

        # block-level
        for split_name, tr, te in block_level_splits(fam_info):
            results.append(
                run_layerwise_probe(
                    dataset_tag=dataset_tag,
                    out_root=out_root,
                    store=store,
                    rows=rows,
                    labels=y_task,
                    train_ids=tr,
                    test_ids=te,
                    variant="original",
                    delta_variant=None,
                    seeds=seeds,
                    split_name=split_name,
                    position_key=str(rows[0]["hidden_state_key"]) if rows else "",
                    feature_mode="absolute::h_original",
                    label_name="task_family_baseline",
                    task_is_multiclass=True,
                )
            )
        return results

    want_datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    want_positions = [x.strip() for x in str(args.positions).split(",") if x.strip()]
    probe_labels = PRIMARY_LABELS if args.label_set == "primary" else (PRIMARY_LABELS + EXTRA_LABELS)

    # Only run the probing loops requested (useful for resuming partial runs).
    results_all: list[dict[str, Any]] = []
    for pos in want_positions:
        if "probe_ready" in want_datasets:
            results_all.extend(
                run_all_for_dataset(
                    "probe_ready",
                    meta_root / f"probe_ready.{pos}.jsonl",
                    info_pr,
                    labels_pr,
                )
            )
        if "gpt_test" in want_datasets:
            results_all.extend(
                run_all_for_dataset(
                    "gpt_test",
                    meta_root / f"gpt_test.{pos}.jsonl",
                    info_gt,
                    labels_gt,
                )
            )

    # MCQ transfer：train non-MCQ(strict_symbolic) -> test MCQ(strict_symbolic)，反向同理。
    # 注意：MCQ metadata 行只有 original；因此这里只跑 absolute::h_original。
    # 仅当本次 run 包含 probe_ready 时才运行（便于分段续跑）。
    if (not args.no_transfers) and ("probe_ready" in want_datasets):
        def _prefix_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for r in rows:
                rr = dict(r)
                rr["family_id"] = f"{prefix}::{rr.get('family_id')}"
                out.append(rr)
            return out

        for pos in ("final_input", "pre_answer"):
            rows_non = _read_jsonl(meta_root / f"probe_ready.{pos}.jsonl")
            rows_mcq = _read_jsonl(meta_root / f"probe_ready_mcq.{pos}.jsonl")

            # Prefix family_ids to avoid collisions between non-MCQ and MCQ rows.
            rows_non_p = _prefix_rows(rows_non, "nonmcq")
            rows_mcq_p = _prefix_rows(rows_mcq, "mcq")
            rows_both = rows_non_p + rows_mcq_p

            # build fid sets
            sym_non = sorted({r["family_id"] for r in rows_non if r.get("mode") == "strict_symbolic" and r.get("variant") == "original"})
            sym_mcq = sorted({r["family_id"] for r in rows_mcq if r.get("mode") == "strict_symbolic" and r.get("variant") == "original"})
            # labels: use probe_ready labels (derived from non-MCQ behavior)
            lab_defs = labels_pr.get("labels", {})
            for lab in probe_labels:
                if lab not in lab_defs:
                    continue
                y_raw = lab_defs[lab]["labels"]
                y = {fid: int(v) for fid, v in y_raw.items() if v is not None}
                # intersect
                tr_base = [fid for fid in sym_non if fid in y]
                te_base = [fid for fid in sym_mcq if fid in y]
                if len(tr_base) < 8 or len(te_base) < 8:
                    continue

                tr_ids = [f"nonmcq::{fid}" for fid in tr_base]
                te_ids = [f"mcq::{fid}" for fid in te_base]
                labels_pref = {f"nonmcq::{fid}": y[fid] for fid in tr_base}
                labels_pref.update({f"mcq::{fid}": y[fid] for fid in te_base})

                results_all.append(
                    run_layerwise_probe(
                    dataset_tag="mcq_transfer_v2",
                        out_root=out_root,
                        store=store,
                        rows=rows_both,
                        labels=labels_pref,
                        train_ids=tr_ids,
                        test_ids=te_ids,
                        variant="original",
                        delta_variant=None,
                        seeds=seeds,
                        split_name=f"mcq_transfer::train_nonmcq__test_mcq::{pos}",
                        position_key=pos,
                        feature_mode="absolute::h_original",
                        label_name=lab,
                        task_is_multiclass=False,
                    )
                )
                # reverse: train MCQ -> test non-MCQ
                results_all.append(
                    run_layerwise_probe(
                    dataset_tag="mcq_transfer_v2",
                        out_root=out_root,
                        store=store,
                        rows=rows_both,
                        labels=labels_pref,
                        train_ids=te_ids,
                        test_ids=tr_ids,
                        variant="original",
                        delta_variant=None,
                        seeds=seeds,
                        split_name=f"mcq_transfer::train_mcq__test_nonmcq::{pos}",
                        position_key=pos,
                        feature_mode="absolute::h_original",
                        label_name=lab,
                        task_is_multiclass=False,
                    )
                )

    # cross-dataset transfer (probe_ready <-> gpt_test)
    # 用 train dataset 的阈值对齐 label（避免 quantile 语义漂移）。
    def transfer_labels(train_blob: dict[str, Any], test_blob: dict[str, Any], label: str) -> tuple[dict[str, int], dict[str, int], str] | None:
        if label not in train_blob.get("labels", {}):
            return None
        sig_name = train_blob["labels"][label]["signal"]
        thr = train_blob["labels"][label]["thresholds"]
        pos_thr = float(thr["pos_ge"])
        neg_thr = float(thr["neg_le"])
        def assign(blob: dict[str, Any]) -> dict[str, int]:
            out = {}
            for fid, sd in blob.get("signals", {}).items():
                v = sd.get(sig_name)
                if not isinstance(v, (int, float)):
                    continue
                fv = float(v)
                if fv >= pos_thr:
                    out[fid] = 1
                elif fv <= neg_thr:
                    out[fid] = 0
            return out
        y_tr = assign(train_blob)
        y_te = assign(test_blob)
        if len(set(y_tr.values())) < 2 or len(set(y_te.values())) < 2:
            return None
        return y_tr, y_te, sig_name

    if (not args.no_transfers) and ("probe_ready" in want_datasets) and ("gpt_test" in want_datasets):
        for pos in ("final_input", "pre_answer"):
            rows_pr = _read_jsonl(meta_root / f"probe_ready.{pos}.jsonl")
            rows_gt = _read_jsonl(meta_root / f"gpt_test.{pos}.jsonl")

            rows_pr_p = _prefix_rows(rows_pr, "probe_ready")
            rows_gt_p = _prefix_rows(rows_gt, "gpt_test")
            rows_both = rows_pr_p + rows_gt_p

            # only use original for features in transfer
            pr_ids = sorted({r["family_id"] for r in rows_pr if r.get("variant") == "original"})
            gt_ids = sorted({r["family_id"] for r in rows_gt if r.get("variant") == "original"})
            for lab in probe_labels:
                t1 = transfer_labels(labels_pr, labels_gt, lab)
                if t1 is not None:
                    y_tr, y_te, sig = t1
                    tr_base = [fid for fid in pr_ids if fid in y_tr]
                    te_base = [fid for fid in gt_ids if fid in y_te]
                    tr = [f"probe_ready::{fid}" for fid in tr_base]
                    te = [f"gpt_test::{fid}" for fid in te_base]
                    labels_pref = {f"probe_ready::{fid}": y_tr[fid] for fid in tr_base}
                    labels_pref.update({f"gpt_test::{fid}": y_te[fid] for fid in te_base})
                    if len(tr) >= 8 and len(te) >= 8:
                        results_all.append(
                            run_layerwise_probe(
                                dataset_tag="cross_dataset_transfer_v2",
                                out_root=out_root,
                                store=store,
                                rows=rows_both,
                                labels=labels_pref,
                                train_ids=tr,
                                test_ids=te,
                                variant="original",
                                delta_variant=None,
                                seeds=seeds,
                                split_name=f"train_probe_ready__test_gpt_test::{lab}::{pos}",
                                position_key=pos,
                                feature_mode="absolute::h_original",
                                label_name=lab,
                                task_is_multiclass=False,
                            )
                        )
                t2 = transfer_labels(labels_gt, labels_pr, lab)
                if t2 is not None:
                    y_tr, y_te, sig = t2
                    tr_base = [fid for fid in gt_ids if fid in y_tr]
                    te_base = [fid for fid in pr_ids if fid in y_te]
                    tr = [f"gpt_test::{fid}" for fid in tr_base]
                    te = [f"probe_ready::{fid}" for fid in te_base]
                    labels_pref = {f"gpt_test::{fid}": y_tr[fid] for fid in tr_base}
                    labels_pref.update({f"probe_ready::{fid}": y_te[fid] for fid in te_base})
                    if len(tr) >= 8 and len(te) >= 8:
                        results_all.append(
                            run_layerwise_probe(
                                dataset_tag="cross_dataset_transfer_v2",
                                out_root=out_root,
                                store=store,
                                rows=rows_both,
                                labels=labels_pref,
                                train_ids=tr,
                                test_ids=te,
                                variant="original",
                                delta_variant=None,
                                seeds=seeds,
                                split_name=f"train_gpt_test__test_probe_ready::{lab}::{pos}",
                                position_key=pos,
                                feature_mode="absolute::h_original",
                                label_name=lab,
                                task_is_multiclass=False,
                            )
                        )

    # Save an aggregated run index
    _write_json({"n_runs": len(results_all), "runs": results_all}, out_root / "run_index.json")

    # ========== final markdown report ==========
    # Only meaningful once probe_ready has been probed (so we can rank atomic labels).
    if (not args.no_report) and ("probe_ready" in want_datasets):
        # 读取 run_index.json 做最简结论汇总
        stable: list[str] = []
        unstable: list[str] = []
        # heuristic: label stable if probe_ready pre_answer delta or absolute achieves macro_f1_mean >= 0.7
        # We read per-run summary.json for probe_ready random split.
        for lab in probe_labels:
            best = -1.0
            for pos in ("final_input", "pre_answer"):
                for feat in (
                    "absolute::h_original",
                    "delta::premise-original",
                    "delta::premise_removal-original",
                    "delta::wrongclaim_bare-original",
                    "delta::scaffold_2-original",
                    "delta::paraphrase-original",
                    "delta::competing_claims-original",
                    "delta::highlight-original",
                ):
                    summ = out_root / "probe_ready" / pos / feat / lab / "random_family_split" / "summary.json"
                    if not summ.exists():
                        continue
                    s = json.loads(summ.read_text(encoding="utf-8"))
                    if s.get("status") != "ok":
                        continue
                    best = max(best, float(s.get("best_macro_f1_mean", -1.0)))
            if best >= 0.7:
                stable.append(f"{lab} (best_macro_f1≈{best:.3f})")
            else:
                unstable.append(f"{lab} (best_macro_f1≈{best:.3f})")

        md_lines = []
        md_lines.append(f"# First-pass Probing & Dataset Comparison\n")
        md_lines.append(f"- 模型：`{args.model}`")
        md_lines.append(f"- seeds：`{seeds}`")
        md_lines.append(f"- extreme-bin top quantile：`{args.top_quantile}`\n")
        md_lines.append("## 核心结论（first pass）\n")
        md_lines.append("### probe-ready 上当前最稳定的 atomic probes\n")
        for x in stable[:12]:
            md_lines.append(f"- {x}")
        if not stable:
            md_lines.append("- （暂无达到阈值的稳定标签；请检查 hidden states / labels 是否退化）")
        md_lines.append("\n### probe-ready 上当前不稳定/不可 probe 的标签\n")
        for x in unstable[:20]:
            md_lines.append(f"- {x}")

        md_lines.append("\n## 数据对比概览\n")
        md_lines.append("- 见 `dataset_comparison_summary.json`（family/block/sub_family/mode/variant 覆盖）")
        md_lines.append("- probe-ready diagnostics：`family_diagnostics_probe_ready.json`")
        md_lines.append("- gpt_test diagnostics：`family_diagnostics_gpt_test.json`\n")

        md_lines.append("## 主要输出路径\n")
        md_lines.append(f"- probing 结果目录：`{out_root}`")
        md_lines.append(f"- hidden state cache：`{hs_root}`")
        md_lines.append(f"- score cache：`{score_root}`")
        md_lines.append(f"- metadata cache：`{meta_root}`\n")

        report_path = Path("reports/probing_comparison_report.md")
        _ensure_dir(report_path.parent)
        report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        print(f"Done. Report: {report_path}")
        print(f"Time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
