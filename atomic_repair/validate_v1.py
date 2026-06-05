"""Validate atomic-repair v1 data. Exits 1 on any failure; always writes a report.

v1 contract (differs from v0):
  - Entities are SHARED across repair train/eval (NOT held out). The validator
    REQUIRES large train/eval entity overlap — a regression to entity-OOD fails.
  - Generalization is held out on FORM: question / corruption / trace form ids,
    and fact-question form ids, must be disjoint between train and eval.
  - Fact coverage: every gold chain triple used by any repair item must appear in
    the injected fact-train set; fact_train and fact_eval cover the same facts.
  - CoT-only and Skill+CoT inputs must be byte-identical per row.
  - Every repair_eval item must use eval-side forms only, and its facts ⊆ fact_train.

Per-record correctness invariants are inherited from v0's contract.

Usage:
  python validate_v1.py --data_dir data_v1 --out data_v1/data_sanity_report_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import forms_v1 as F
import convert_v1 as C

CELL_TO_DIAGNOSIS = {
    "H-Aug": "missing_bridge_fact", "H-Abl": "bridge_entity_missing",
    "H-Cor": "wrong_bridge_contamination", "K-Cor": "wrong_factual_claim",
    "Clean": "no_failure_detected",
}
CELL_TO_SKILL = {
    "H-Aug": "retrieve_bridge_fact", "H-Abl": "recover_bridge_entity",
    "H-Cor": "bridge_source_verification", "K-Cor": "contradiction_check",
    "Clean": "keep_answer",
}

EXPECTED_REPAIR = {
    "train": {"H-Aug": 360, "H-Abl": 360, "H-Cor": 480, "K-Cor": 360, "Clean": 360},
    "eval":  {"H-Aug": 120, "H-Abl": 120, "H-Cor": 180, "K-Cor": 120, "Clean": 120},
}
EXPECTED_FACT_DISTINCT = 345


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def check_repair_record(r: dict, fails: list[str]) -> None:
    where = r.get("id", "?")
    req = ["problem", "tentative_answer", "gold_answer", "final_answer",
           "repair_trace", "diagnosis", "repair_skill", "should_repair",
           "oracle_facts", "symbolic_facts", "cell", "entity_split",
           "question_form_id", "trace_form_id", "form_split"]
    for k in req:
        if k not in r:
            fails.append(f"{where}: missing field {k}")
            return
    cell = r["cell"]
    if r["final_answer"] != r["gold_answer"]:
        fails.append(f"{where}: final_answer != gold_answer")
    if r["diagnosis"] != CELL_TO_DIAGNOSIS[cell]:
        fails.append(f"{where}: diagnosis mismatch for {cell}")
    if r["repair_skill"] != CELL_TO_SKILL[cell]:
        fails.append(f"{where}: repair_skill mismatch for {cell}")
    if r["entity_split"] != "shared":
        fails.append(f"{where}: entity_split should be 'shared'")
    if cell == "Clean":
        if r["tentative_answer"] != r["gold_answer"]:
            fails.append(f"{where}: Clean tentative != gold")
        if r["should_repair"] is not False:
            fails.append(f"{where}: Clean should_repair must be False")
        if r["planted_wrong_answer"] is not None:
            fails.append(f"{where}: Clean planted_wrong_answer must be None")
    else:
        if r["tentative_answer"] == r["gold_answer"]:
            fails.append(f"{where}: non-Clean tentative == gold")
        if r["should_repair"] is not True:
            fails.append(f"{where}: non-Clean should_repair must be True")
    if cell in ("H-Cor", "K-Cor"):
        pw = r.get("planted_wrong_answer")
        if not pw:
            fails.append(f"{where}: {cell} missing planted_wrong_answer")
        elif pw == r["gold_answer"]:
            fails.append(f"{where}: {cell} planted == gold")
        elif r["tentative_answer"] != pw:
            fails.append(f"{where}: {cell} tentative != planted_wrong_answer")
        if not r.get("corruption_form_id"):
            fails.append(f"{where}: {cell} missing corruption_form_id")
    # oracle grounding: trace mentions some oracle token.
    if not r["oracle_facts"]:
        fails.append(f"{where}: empty oracle_facts")
    else:
        toks = {t for s in r["oracle_facts"] for t in s.split() if len(t) >= 4}
        if toks and not any(t in r["repair_trace"] for t in toks):
            fails.append(f"{where}: repair_trace not grounded in oracle_facts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data_v1"))
    ap.add_argument("--out", type=Path, default=Path("data_v1/data_sanity_report_v1.json"))
    args = ap.parse_args()
    d = args.data_dir
    fails: list[str] = []

    fact_train = load(d / "fact_train.jsonl")
    fact_eval = load(d / "fact_eval.jsonl")
    rep_train = load(d / "repair_train.jsonl")
    rep_eval = load(d / "repair_eval.jsonl")

    # 1. per-record contract
    for r in rep_train + rep_eval:
        check_repair_record(r, fails)

    # 2. counts
    for split, rows in (("train", rep_train), ("eval", rep_eval)):
        by = Counter(r["cell"] for r in rows)
        for cell, n in EXPECTED_REPAIR[split].items():
            if by.get(cell, 0) != n:
                fails.append(f"repair {split} {cell}: got {by.get(cell,0)} want {n}")

    # 3. fact coverage (gold chain only)
    fact_triples = {tuple(r["symbolic_fact"]) for r in fact_train}
    repair_triples = set()
    for r in rep_train + rep_eval:
        for t in r.get("gold_symbolic_facts", r["symbolic_facts"]):
            repair_triples.add(tuple(t))
    missing = repair_triples - fact_triples
    if missing:
        fails.append(f"COVERAGE: {len(missing)} repair triples not in fact_train")
    ft_set = {tuple(r["symbolic_fact"]) for r in fact_train}
    fe_set = {tuple(r["symbolic_fact"]) for r in fact_eval}
    if ft_set != fe_set:
        fails.append("fact_train and fact_eval cover different fact sets")
    if len(ft_set) != EXPECTED_FACT_DISTINCT:
        fails.append(f"distinct facts {len(ft_set)} != {EXPECTED_FACT_DISTINCT}")

    # 4. entity check INVERTED: require large overlap (not disjoint).
    def heads(rows):
        return {r["symbolic_facts"][0][0] for r in rows}
    ht, he = heads(rep_train), heads(rep_eval)
    overlap = len(ht & he)
    if he and overlap / len(he) < 0.8:
        fails.append(f"entity overlap too low ({overlap}/{len(he)}) — looks entity-OOD")

    # 5. form disjointness, 3 axes + fact phrasing.
    def idset(rows, k):
        return {r[k] for r in rows if r.get(k)}
    for k in ("question_form_id", "corruption_form_id", "trace_form_id"):
        o = idset(rep_train, k) & idset(rep_eval, k)
        if o:
            fails.append(f"form {k} train∩eval not empty: {sorted(o)}")
    fo = {r["fact_form_id"] for r in fact_train} & {r["fact_form_id"] for r in fact_eval}
    if fo:
        fails.append(f"fact_form_id train∩eval not empty: {sorted(fo)}")

    # 6. eval items use eval-side forms only; facts ⊆ fact_train.
    eval_ids = F.all_form_ids("eval")
    for r in rep_eval:
        if r["question_form_id"] not in eval_ids["question"]:
            fails.append(f"{r['id']}: eval question form not eval-side")
        if r.get("corruption_form_id"):
            cf = r["corruption_form_id"]
            if cf not in eval_ids["wrong_bridge"] and cf not in eval_ids["wrong_claim"]:
                fails.append(f"{r['id']}: eval corruption form not eval-side")
        if r["trace_form_id"] not in eval_ids["trace"]:
            fails.append(f"{r['id']}: eval trace form not eval-side")
        for t in r.get("gold_symbolic_facts", r["symbolic_facts"]):
            if tuple(t) not in fact_triples:
                fails.append(f"{r['id']}: uses fact not in fact_train")
                break

    # 7. identical-input invariant (reconstruct CoT vs Skill+CoT).
    for r in rep_train + rep_eval:
        if C.cot_alpaca(r)["input"] != C.skillcot_alpaca(r)["input"]:
            fails.append(f"{r['id']}: cot_input != skillcot_input")
            break

    report = {
        "status": "PASS" if not fails else "FAIL",
        "failures": fails,
        "counts": {
            "fact_train": len(fact_train), "fact_eval": len(fact_eval),
            "repair_train": len(rep_train), "repair_eval": len(rep_eval),
            "distinct_facts": len(ft_set),
            "repair_gold_triples": len(repair_triples),
        },
        "coverage_ok": not missing,
        "entity_overlap": {"shared_heads": overlap, "eval_heads": len(he)},
        "form_disjoint": {
            "question": sorted(idset(rep_train, "question_form_id") & idset(rep_eval, "question_form_id")),
            "corruption": sorted(idset(rep_train, "corruption_form_id") & idset(rep_eval, "corruption_form_id")),
            "trace": sorted(idset(rep_train, "trace_form_id") & idset(rep_eval, "trace_form_id")),
            "fact": sorted(fo),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"status: {report['status']}  ({len(fails)} failures)  -> {args.out}")
    if fails:
        for f in fails[:30]:
            print("  FAIL:", f)
        raise SystemExit(1)
    print(f"coverage: {len(repair_triples)} gold triples ⊆ {len(fact_triples)} fact-train")
    print(f"entity overlap (shared): {overlap}/{len(he)} eval heads also in train")
    print("form disjoint: question/corruption/trace/fact all train∩eval = ∅")


if __name__ == "__main__":
    main()
