"""Atomic-repair v1 data generator (known-component repair setting).

v1 vs v0
--------
v0 withheld facts from the input and held out ENTITIES between train/eval, so the
model could never recall the (fictitious) bridge entity -> final_answer collapsed
by construction. v1 flips this:

  1. A FACT stage injects every atomic fact (single-hop) of the synthetic world.
     We enumerate ALL edges of ALL families, so coverage of any fact a repair item
     references is guaranteed, not sampled.
  2. The REPAIR stage SHARES entities across train/eval (no entity hold-out). The
     held-out axis is FORM: question wording, corruption wording (wrong-bridge /
     wrong-claim), and trace wording all come from disjoint train/eval banks
     (forms_v1.py). Facts seen, repair forms unseen.

We reuse v0's symbolic world wholesale (entity pools, graphs, oracle-fact
renderers) by importing generate_repair_data, and the form banks from forms_v1.

Outputs (into --out_dir, default data_v1/):
  fact_train.jsonl    fact_eval.jsonl       (same 345 facts, disjoint phrasings)
  repair_train.jsonl  repair_eval.jsonl     (shared entities, disjoint forms)

No API, no GPU. Seedable, deterministic, oracle-verifiable.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import generate_repair_data as v0
import forms_v1 as F

FAMILY_NAMES = F.FAMILY_NAMES

# How many repair items per (cell, split). Entities are shared; only forms split.
REPAIR_TARGETS = {
    "train": {"H-Aug": 360, "H-Abl": 360, "H-Cor": 480, "K-Cor": 360, "Clean": 360},
    "eval":  {"H-Aug": 120, "H-Abl": 120, "H-Cor": 180, "K-Cor": 120, "Clean": 120},
}

# Fact phrasings sampled per (edge, hop) for train; eval uses 1 phrasing per edge.
FACT_TRAIN_PHRASINGS_PER_FACT = 2
SURFACE_WEIGHTS = v0.SURFACE_WEIGHTS  # reuse 80/15/5 naturalized/fact_table/compact


# --------------------------------------------------------------------------- #
# Fact stage: enumerate every atomic fact, both hops, all families.
# --------------------------------------------------------------------------- #
def build_fact_records(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Return (fact_train, fact_eval). Same facts, disjoint phrasings."""
    train, eval_ = [], []
    tid = {"train": 0, "eval": 0}

    for fname in FAMILY_NAMES:
        graph = v0.build_family_graph(fname)
        rel1, rel2 = graph["relations"]

        # Distinct first-hop facts (head -> bridge) and second-hop (bridge -> tail).
        first_facts = []   # (subj=head, rel, obj=bridge)
        second_facts = []  # (subj=bridge, rel, obj=tail)
        seen_first, seen_second = set(), set()
        for e in graph["edges"]:
            if e["head"] not in seen_first:
                seen_first.add(e["head"])
                first_facts.append((e["head"], rel1, e["bridge"]))
            if e["bridge"] not in seen_second:
                seen_second.add(e["bridge"])
                second_facts.append((e["bridge"], rel2, e["tail"]))

        for hop, rel, facts in (("first", rel1, first_facts), ("second", rel2, second_facts)):
            pool = F.FACT_QUESTION_PHRASINGS[rel]
            for (subj, r, obj) in facts:
                # train: a couple of phrasings; eval: one held-out phrasing.
                n_tr = min(FACT_TRAIN_PHRASINGS_PER_FACT, len(pool["train"]))
                tr_choices = rng.sample(pool["train"], n_tr)
                for ph in tr_choices:
                    train.append(_fact_record(
                        fname, hop, ph, subj, obj, [subj, r, obj], "train", tid))
                ev_ph = rng.choice(pool["eval"])
                eval_.append(_fact_record(
                    fname, hop, ev_ph, subj, obj, [subj, r, obj], "eval", tid))

    return train, eval_


def _fact_record(fname, hop, phrasing, subj, obj, triple, split, tid) -> dict:
    rec = {
        "id": f"fact_{split}_{tid[split]:06d}",
        "relation_family": fname,
        "hop": hop,
        "fact_form_id": phrasing["id"],
        "fact_split": split,
        "question": phrasing["text"].format(subj=subj),
        "answer": obj,
        "symbolic_fact": triple,
        "surface_type": "naturalized",
    }
    tid[split] += 1
    return rec


# --------------------------------------------------------------------------- #
# Repair stage: shared entities, split forms.
# --------------------------------------------------------------------------- #
def _pick_distinct(rng: random.Random, pool: list[str], banned: str) -> str:
    pick = rng.choice(pool)
    tries = 0
    while pick == banned and tries < 30:
        pick = rng.choice(pool)
        tries += 1
    return pick


def _render_question(qtpl: dict[str, str], surface: str, head: str) -> str:
    if surface == "fact_table":
        return qtpl["fact_table"].format(head=head)
    if surface == "compact":
        return qtpl["compact"].format(head=head)
    return qtpl["natural"].format(head=head)


def _inject(surface: str, sentence: str, body: str) -> str:
    """Prepend an injected sentence, keeping the surface readable."""
    if surface == "fact_table":
        try:
            facts_block, q_block = body.split("Question:")
            return facts_block.rstrip() + f"\n- {sentence}\n" + "Question:" + q_block
        except ValueError:
            return f"{sentence}\n{body}"
    if surface == "compact":
        return f"hint: {sentence}\n{body}"
    return f"{sentence} {body}"


def build_repair_record(*, cell: str, split: str, rng: random.Random,
                        fname: str, graph: dict[str, Any], edge: dict[str, str],
                        qtpl: dict[str, str], idx: int) -> dict:
    spec = v0.CELL_SPEC[cell]
    head, bridge, tail = edge["head"], edge["bridge"], edge["tail"]
    surface = v0.pick_surface(rng)
    rel1, rel2 = graph["relations"]

    common = dict(
        id=f"{cell}_{split}_{idx:06d}",
        cell=cell,
        surface_type=surface,
        relation_family=fname,
        graph_pattern=graph.get("graph_pattern", "two_hop_bridge"),
        task_type=spec["task_type"],
        diagnosis=spec["diagnosis"],
        repair_skill=spec["repair_skill"],
        should_repair=spec["should_repair"],
        split=split,
        entity_split="shared",            # v1: entities are NOT held out
        question_form_id=qtpl["id"],
        form_split=split,
    )

    # Pick a trace template for this cell+split.
    trace_tpl = rng.choice(F.TRACE_TEMPLATES[cell][split])
    common["trace_form_id"] = trace_tpl["id"]

    gold = tail
    fact1 = v0.first_fact_sentence(fname, head, bridge)
    fact2 = v0.second_fact_sentence(fname, bridge, tail)

    if cell == "Clean":
        problem = _render_question(qtpl, surface, head)
        oracle, sym = v0.build_oracle_facts(fname, edge)
        trace = trace_tpl["text"].format(fact1=fact1, fact2=fact2, gold=gold)
        return dict(**common, problem=problem, tentative_answer=gold,
                    gold_answer=gold, planted_wrong_answer=None,
                    corruption_form_id=None, repair_trace=trace,
                    final_answer=gold, oracle_facts=oracle, symbolic_facts=sym)

    if cell == "H-Aug":
        problem = _render_question(qtpl, surface, head)
        wrong_tail = _pick_distinct(rng, graph["tails"], gold)
        oracle, sym = v0.build_oracle_facts(fname, edge)
        trace = trace_tpl["text"].format(fact1=fact1, fact2=fact2, gold=gold)
        return dict(**common, problem=problem, tentative_answer=wrong_tail,
                    gold_answer=gold, planted_wrong_answer=None,
                    corruption_form_id=None, repair_trace=trace,
                    final_answer=gold, oracle_facts=oracle, symbolic_facts=sym)

    if cell == "H-Abl":
        case_id = f"Case #{abs(hash(head)) % 10000:04d}-{idx % 100:02d}"
        if surface == "naturalized":
            body = qtpl["ablate"]
        elif surface == "fact_table":
            body = qtpl["fact_table"].format(head="[MASK]")
        else:
            body = qtpl["compact"].format(head="[MASK]")
        problem = f"{case_id}: {body}"
        wrong_tail = _pick_distinct(rng, graph["tails"], gold)
        oracle, sym = v0.build_oracle_facts(fname, edge)
        trace = trace_tpl["text"].format(fact1=fact1, fact2=fact2, gold=gold)
        return dict(**common, problem=problem, tentative_answer=wrong_tail,
                    gold_answer=gold, planted_wrong_answer=None,
                    corruption_form_id=None, repair_trace=trace,
                    final_answer=gold, oracle_facts=oracle, symbolic_facts=sym)

    if cell == "H-Cor":
        wrong_bridge = _pick_distinct(rng, graph["bridges"], bridge)
        wb_edge = next((e for e in graph["edges"] if e["bridge"] == wrong_bridge), None)
        wrong_tail = wb_edge["tail"] if wb_edge else _pick_distinct(rng, graph["tails"], tail)
        if wrong_tail == tail:
            wrong_tail = _pick_distinct(rng, graph["tails"], tail)
        wb_phrase = rng.choice(F.WRONG_BRIDGE_PHRASINGS[fname][split])
        inj = wb_phrase["text"].format(head=head, wrong_bridge=wrong_bridge)
        problem = _inject(surface, inj, _render_question(qtpl, surface, head))
        oracle, sym = v0.build_oracle_facts(
            fname, edge, extra=dict(wrong_bridge=wrong_bridge, wrong_tail=wrong_tail))
        trace = trace_tpl["text"].format(
            fact1=fact1, fact2=fact2, gold=gold,
            wrong_tail=wrong_tail, wrong_bridge=wrong_bridge)
        common["corruption_form_id"] = wb_phrase["id"]
        # gold_symbolic_facts = the chain the model must KNOW (head->bridge,
        # bridge->tail); sym additionally carries the corruption triple
        # [wrong_bridge, rel2, wrong_tail], which is a fabricated contrast fact,
        # NOT something to inject or require coverage for.
        gold_sym = [[head, rel1, bridge], [bridge, rel2, tail]]
        return dict(**common, problem=problem, tentative_answer=wrong_tail,
                    gold_answer=gold, planted_wrong_answer=wrong_tail,
                    repair_trace=trace, final_answer=gold,
                    oracle_facts=oracle, symbolic_facts=sym,
                    gold_symbolic_facts=gold_sym)

    if cell == "K-Cor":
        wrong_tail = _pick_distinct(rng, graph["tails"], tail)
        wc_phrase = rng.choice(F.WRONG_CLAIM_PHRASINGS[fname][split])
        inj = wc_phrase["text"].format(bridge=bridge, wrong_tail=wrong_tail)
        # 1-hop question about the bridge->tail edge.
        if surface == "fact_table":
            problem = f"Facts:\n- {inj}\nQuestion:\n- {_kcor_q(fname, bridge)}"
        elif surface == "compact":
            problem = f"claim: {inj}\nQ: {_kcor_q(fname, bridge)}"
        else:
            problem = f"{inj} {_kcor_q(fname, bridge)}"
        oracle = [fact2]
        sym = [[bridge, rel2, tail]]
        trace = trace_tpl["text"].format(
            fact1=fact2, fact2=fact2, gold=gold,
            wrong_tail=wrong_tail, bridge=bridge)
        common["corruption_form_id"] = wc_phrase["id"]
        return dict(**common, problem=problem, tentative_answer=wrong_tail,
                    gold_answer=gold, planted_wrong_answer=wrong_tail,
                    repair_trace=trace, final_answer=gold,
                    oracle_facts=oracle, symbolic_facts=sym)

    raise ValueError(cell)


def _kcor_q(fname: str, bridge: str) -> str:
    """A simple held-out-independent 1-hop question for K-Cor (second hop)."""
    rel2 = F.FAMILY_RELATIONS[fname][1]
    # Use a fixed phrasing per relation that is NOT in the fact-QA banks, so the
    # K-Cor question wording does not collide with the fact stage.
    table = {
        "nationality": f"What nationality is {bridge}?",
        "uses_currency": f"What currency does {bridge} use?",
        "headquartered_in": f"Where is {bridge} headquartered?",
        "birth_country": f"Where was {bridge} born?",
        "field": f"Which scientific field does {bridge} belong to?",
    }
    return table[rel2]


def build_all_repair(rng: random.Random) -> tuple[list[dict], list[dict]]:
    graphs = {fn: v0.build_family_graph(fn) for fn in FAMILY_NAMES}
    train, eval_ = [], []

    for split in ("train", "eval"):
        targets = REPAIR_TARGETS[split]
        out = train if split == "train" else eval_
        cell_idx = {c: 0 for c in targets}
        used_pairs: set[tuple[str, str]] = set()

        # Interleave cells to avoid template/family streaks.
        cells_cycle = []
        planned = {c: 0 for c in targets}
        for _ in range(max(targets.values())):
            for c in ("H-Aug", "H-Abl", "H-Cor", "K-Cor", "Clean"):
                if planned[c] < targets[c]:
                    cells_cycle.append(c)
                    planned[c] += 1

        for cell in cells_cycle:
            fname = FAMILY_NAMES[cell_idx[cell] % len(FAMILY_NAMES)]
            graph = graphs[fname]
            edges = graph["edges"]            # ALL edges usable both sides
            qpool = F.QUESTION_TEMPLATES[fname][split]

            rec = None
            for attempt in range(120):
                edge = edges[(cell_idx[cell] + attempt) % len(edges)]
                qtpl = qpool[(cell_idx[cell] + attempt) % len(qpool)]
                cand = build_repair_record(
                    cell=cell, split=split, rng=rng, fname=fname, graph=graph,
                    edge=edge, qtpl=qtpl, idx=cell_idx[cell])
                # Cells other than H-Cor have no corruption triple, so their gold
                # chain == symbolic_facts. (H-Cor sets gold_symbolic_facts itself.)
                cand.setdefault("gold_symbolic_facts", cand["symbolic_facts"])
                key = (cand["problem"], cand["tentative_answer"])
                if key not in used_pairs:
                    used_pairs.add(key)
                    rec = cand
                    break
            if rec is None:
                cand["problem"] = cand["problem"] + f"  // variant#{cell_idx[cell]}"
                used_pairs.add((cand["problem"], cand["tentative_answer"]))
                rec = cand
            cell_idx[cell] += 1
            out.append(rec)

    return train, eval_


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):5d} records -> {path}")


def generate(out_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    fact_train, fact_eval = build_fact_records(rng)
    repair_train, repair_eval = build_all_repair(rng)

    # Hard coverage guarantee before writing: every triple any repair item uses
    # must appear in the injected fact-train set.
    fact_triples = {tuple(r["symbolic_fact"]) for r in fact_train}
    repair_triples = set()
    for r in repair_train + repair_eval:
        # Coverage is required only for the GOLD chain the model must know, not
        # for fabricated corruption-contrast triples (H-Cor's wrong bridge).
        for t in r.get("gold_symbolic_facts", r["symbolic_facts"]):
            repair_triples.add(tuple(t))
    missing = repair_triples - fact_triples
    if missing:
        raise SystemExit(
            f"COVERAGE FAILURE: {len(missing)} repair triple(s) not in fact_train, "
            f"e.g. {sorted(missing)[:5]}")
    print(f"coverage OK: {len(repair_triples)} repair triples ⊆ "
          f"{len(fact_triples)} fact-train triples")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(fact_train, out_dir / "fact_train.jsonl")
    _write_jsonl(fact_eval, out_dir / "fact_eval.jsonl")
    _write_jsonl(repair_train, out_dir / "repair_train.jsonl")
    _write_jsonl(repair_eval, out_dir / "repair_eval.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("data_v1"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
