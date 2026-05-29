"""
Hard validator for atomic-repair raw JSONL data.

Exits 1 if any hard check fails. Always writes a sanity report.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Expected counts (must match generate_repair_data.TARGET_COUNTS).
EXPECTED = {
    "train": {"H-Aug": 400, "H-Abl": 400, "H-Cor": 500, "K-Cor": 400, "Clean": 400},
    "eval":  {"H-Aug": 100, "H-Abl": 100, "H-Cor": 150, "K-Cor": 100, "Clean": 100},
}

CELL_TO_DIAGNOSIS = {
    "H-Aug": "missing_bridge_fact",
    "H-Abl": "bridge_entity_missing",
    "H-Cor": "wrong_bridge_contamination",
    "K-Cor": "wrong_factual_claim",
    "Clean": "no_failure_detected",
}
CELL_TO_REPAIR_SKILL = {
    "H-Aug": "retrieve_bridge_fact",
    "H-Abl": "recover_bridge_entity",
    "H-Cor": "bridge_source_verification",
    "K-Cor": "contradiction_check",
    "Clean": "keep_answer",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def fail(msg: str, failures: list[str]):
    print(f"FAIL: {msg}", file=sys.stderr)
    failures.append(msg)


def check_one(rec: dict[str, Any], failures: list[str], where: str):
    cell = rec["cell"]
    rid  = rec["id"]
    ctx = f"[{where} {rid} cell={cell}]"

    # required fields
    for k in ("problem", "gold_answer", "final_answer", "repair_trace",
              "diagnosis", "repair_skill", "should_repair",
              "oracle_facts", "symbolic_facts", "split", "template_id",
              "entity_split", "relation_family", "surface_type",
              "graph_pattern", "task_type"):
        if k not in rec:
            fail(f"{ctx} missing field {k}", failures)

    if not rec.get("gold_answer"):
        fail(f"{ctx} empty gold_answer", failures)
    if rec.get("final_answer") != rec.get("gold_answer"):
        fail(f"{ctx} final_answer != gold_answer", failures)

    if cell == "Clean":
        if rec.get("tentative_answer") != rec.get("gold_answer"):
            fail(f"{ctx} Clean tentative_answer != gold_answer", failures)
        if rec.get("should_repair") is not False:
            fail(f"{ctx} Clean should_repair != False", failures)
        if rec.get("planted_wrong_answer") is not None:
            fail(f"{ctx} Clean planted_wrong_answer must be null", failures)
    else:
        if rec.get("tentative_answer") == rec.get("gold_answer"):
            fail(f"{ctx} non-Clean tentative_answer == gold_answer", failures)
        if rec.get("should_repair") is not True:
            fail(f"{ctx} non-Clean should_repair != True", failures)

    if cell in ("H-Cor", "K-Cor"):
        if not rec.get("planted_wrong_answer"):
            fail(f"{ctx} planted_wrong_answer missing", failures)
        if rec.get("planted_wrong_answer") == rec.get("gold_answer"):
            fail(f"{ctx} planted_wrong_answer == gold_answer", failures)
        if rec.get("tentative_answer") != rec.get("planted_wrong_answer"):
            fail(f"{ctx} tentative_answer != planted_wrong_answer", failures)

    if rec.get("diagnosis") != CELL_TO_DIAGNOSIS.get(cell):
        fail(f"{ctx} diagnosis mismatch (got {rec.get('diagnosis')})", failures)
    if rec.get("repair_skill") != CELL_TO_REPAIR_SKILL.get(cell):
        fail(f"{ctx} repair_skill mismatch (got {rec.get('repair_skill')})", failures)

    if not rec.get("oracle_facts"):
        fail(f"{ctx} oracle_facts empty", failures)
    if not rec.get("symbolic_facts"):
        fail(f"{ctx} symbolic_facts empty", failures)

    # repair_trace must mention at least one oracle fact (substring match on
    # the noun span — we check for any token longer than 3 chars from any
    # oracle fact).
    trace = rec.get("repair_trace") or ""
    facts = rec.get("oracle_facts") or []
    mentioned = False
    for fact in facts:
        # check if any 4+-char word from the fact appears in the trace
        for token in fact.replace(".", "").split():
            if len(token) >= 4 and token in trace:
                mentioned = True
                break
        if mentioned:
            break
    if not mentioned:
        fail(f"{ctx} repair_trace mentions no oracle fact token", failures)


def held_out_check(train: list[dict[str, Any]], eval_: list[dict[str, Any]],
                   failures: list[str]):
    train_templates = {(r["relation_family"], r["template_id"]) for r in train}
    eval_templates  = {(r["relation_family"], r["template_id"]) for r in eval_}
    overlap_t = train_templates & eval_templates
    if overlap_t:
        fail(f"held-out template check: {len(overlap_t)} (family, template) overlap; "
             f"sample={sorted(overlap_t)[:3]}", failures)

    def entity_of(r):
        # entity == head; we can pull it from symbolic_facts[0][0]
        if r["symbolic_facts"]:
            return (r["relation_family"], r["symbolic_facts"][0][0])
        return None
    train_entities = {entity_of(r) for r in train if entity_of(r) is not None}
    eval_entities  = {entity_of(r) for r in eval_ if entity_of(r) is not None}
    overlap_e = train_entities & eval_entities
    if overlap_e:
        # K-Cor uses bridge as the "head" of its 1-hop, so K-Cor entities may
        # legitimately overlap with H-* train heads only if bridge of one
        # family happens to equal head of another family — not the case
        # since pools are disjoint. So any overlap is a bug.
        fail(f"held-out entity check: {len(overlap_e)} (family, head) overlap; "
             f"sample={sorted(overlap_e)[:3]}", failures)


def check_split(records: list[dict[str, Any]], split: str, failures: list[str]):
    # expected counts by cell
    by_cell = Counter(r["cell"] for r in records)
    for c, n in EXPECTED[split].items():
        if by_cell.get(c, 0) != n:
            fail(f"{split} cell {c}: expected {n}, got {by_cell.get(c,0)}", failures)
    # id unique
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        fail(f"{split} duplicate ids: {len(dupes)} (sample={dupes[:3]})", failures)
    # (problem, tentative_answer) pair unique — different cells can share a
    # `problem` (H-Aug vs Clean differ only in `tentative_answer`), so we
    # check pair uniqueness rather than problem uniqueness.
    pairs = [(r["problem"], r["tentative_answer"]) for r in records]
    if len(set(pairs)) != len(pairs):
        dupes = [k for k, v in Counter(pairs).items() if v > 1]
        fail(f"{split} duplicate (problem, tentative_answer) pairs: {len(dupes)}", failures)
    for r in records:
        check_one(r, failures, where=split)


def llamafactory_roundtrip(records: list[dict[str, Any]], failures: list[str]):
    """Spot-check: the JSON-serializable output we'd write to LLaMA-Factory
    must be parseable for every record."""
    for r in records[:200]:  # spot check, 200 is enough
        try:
            out = {
                "diagnosis":    r["diagnosis"],
                "repair_skill": r["repair_skill"],
                "repair_trace": r["repair_trace"],
                "final_answer": r["final_answer"],
            }
            s = json.dumps(out, ensure_ascii=False)
            json.loads(s)
        except Exception as e:
            fail(f"LF roundtrip failed for {r['id']}: {e}", failures)
            break


def build_report(train: list[dict[str, Any]], eval_: list[dict[str, Any]]):
    def by(records, key):
        return dict(Counter(r[key] for r in records))

    def samples(records, n=5):
        by_cell = defaultdict(list)
        for r in records:
            if len(by_cell[r["cell"]]) < n:
                by_cell[r["cell"]].append({
                    "id": r["id"],
                    "cell": r["cell"],
                    "surface_type": r["surface_type"],
                    "relation_family": r["relation_family"],
                    "problem": r["problem"],
                    "tentative_answer": r["tentative_answer"],
                    "gold_answer": r["gold_answer"],
                    "planted_wrong_answer": r.get("planted_wrong_answer"),
                    "repair_trace": r["repair_trace"],
                    "oracle_facts": r["oracle_facts"],
                })
        return dict(by_cell)

    return {
        "counts": {
            "train_total": len(train), "eval_total": len(eval_),
            "train_by_cell": by(train, "cell"),
            "eval_by_cell":  by(eval_, "cell"),
            "train_by_family": by(train, "relation_family"),
            "eval_by_family":  by(eval_, "relation_family"),
            "train_by_surface": by(train, "surface_type"),
            "eval_by_surface":  by(eval_, "surface_type"),
        },
        "held_out_templates": {
            "train_template_ids": sorted({r["template_id"] for r in train}),
            "eval_template_ids":  sorted({r["template_id"] for r in eval_}),
            "overlap": sorted(
                {r["template_id"] for r in train} & {r["template_id"] for r in eval_}
            ),
        },
        "held_out_entities": {
            "train_entity_count": len({(r["relation_family"], r["symbolic_facts"][0][0]) for r in train}),
            "eval_entity_count":  len({(r["relation_family"], r["symbolic_facts"][0][0]) for r in eval_}),
            "overlap_count": len(
                {(r["relation_family"], r["symbolic_facts"][0][0]) for r in train}
                & {(r["relation_family"], r["symbolic_facts"][0][0]) for r in eval_}
            ),
        },
        "duplicate_problems": {
            "train_problem_only": len([k for k, v in Counter(r["problem"] for r in train).items() if v > 1]),
            "eval_problem_only":  len([k for k, v in Counter(r["problem"] for r in eval_).items()  if v > 1]),
            "train_pair":         len([k for k, v in Counter((r["problem"], r["tentative_answer"]) for r in train).items() if v > 1]),
            "eval_pair":          len([k for k, v in Counter((r["problem"], r["tentative_answer"]) for r in eval_).items()  if v > 1]),
        },
        "samples": {
            "train": samples(train),
            "eval":  samples(eval_),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    train = load_jsonl(args.train)
    eval_ = load_jsonl(args.eval)

    failures: list[str] = []
    check_split(train, "train", failures)
    check_split(eval_, "eval",  failures)
    held_out_check(train, eval_, failures)
    llamafactory_roundtrip(train, failures)
    llamafactory_roundtrip(eval_, failures)

    report = build_report(train, eval_)
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\nstatus: {report['status']}  (failures={len(failures)})")
    print(f"report: {args.out}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
