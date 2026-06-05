#!/usr/bin/env python3
"""Score LLaMA-Factory predictions for the atomic-repair task.

Reads the `generated_predictions.jsonl` emitted by `llamafactory-cli ... do_predict`
and the held-out eval set, then reports per-cell accuracy, JSON-validity, the two
failure modes that matter (over-repair / under-repair), a repair-skill confusion
matrix, held-out generalization, and bootstrap CIs.

CPU only. Does not touch model weights. Run locally on the pulled prediction file.

Usage:
  python evaluate_predictions.py \
    --pred  output/qwen3_8b_repair_full_predict/generated_predictions.jsonl \
    --raw   data/repair_raw_eval.jsonl \
    --out   data/eval_report.json \
    --report data/eval_report.md

The prediction file is matched back to the raw eval rows by exact gold-output
(`label`) match first, falling back to line-order index — see `align()`.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Decision fields scored exactly against gold.
SCORE_FIELDS = ("diagnosis", "repair_skill", "final_answer")

CELLS = ["H-Aug", "H-Abl", "H-Cor", "K-Cor", "Clean"]

# The Clean cell's gold contract: no failure, keep the answer. A prediction
# counts as "no repair" when it emits either of these no-op signals.
NO_REPAIR_DIAGNOSIS = "no_failure_detected"
NO_REPAIR_SKILL = "keep_answer"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return rows


def load_pred(path: Path) -> list[dict]:
    """LLaMA-Factory writes one JSON object per line.

    Field names vary slightly across LF versions. We normalise to:
      pred_text  -- the model's generated string
      gold_text  -- the reference output string (LF 'label')
    """
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pred_text = obj.get("predict", obj.get("prediction", obj.get("output", "")))
        gold_text = obj.get("label", obj.get("reference", obj.get("response", "")))
        rows.append({"pred_text": pred_text, "gold_text": gold_text, "_raw": obj})
    return rows


# --------------------------------------------------------------------------- #
# Alignment: map each prediction back to its raw eval row (for the cell label)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    """Collapse whitespace so trivial reformatting doesn't break gold matching."""
    return re.sub(r"\s+", " ", (s or "").strip())


def align(preds: list[dict], raws: list[dict]) -> tuple[list[tuple[dict, dict]], dict]:
    """Pair each prediction with its raw eval row.

    Primary key: exact gold-output (LF `label`) match against the raw row's
    reconstructed gold JSON. Falls back to line index when the gold text is
    absent or ambiguous. Returns (pairs, alignment_stats).
    """
    # Build a lookup from normalised gold-output string -> raw row(s).
    gold_to_raw: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(raws):
        gold_obj = {
            "diagnosis": r["diagnosis"],
            "repair_skill": r["repair_skill"],
            "repair_trace": r["repair_trace"],
            "final_answer": r["final_answer"],
        }
        gold_to_raw[_norm(json.dumps(gold_obj, ensure_ascii=False))].append(i)

    pairs: list[tuple[dict, dict]] = []
    stats = {"by_gold_match": 0, "by_index": 0, "unmatched": 0}
    used = set()

    for j, p in enumerate(preds):
        key = _norm(p["gold_text"])
        cand = [i for i in gold_to_raw.get(key, []) if i not in used]
        if cand:
            i = cand[0]
            used.add(i)
            pairs.append((p, raws[i]))
            stats["by_gold_match"] += 1
        elif j < len(raws):
            pairs.append((p, raws[j]))  # positional fallback
            stats["by_index"] += 1
        else:
            stats["unmatched"] += 1
    return pairs, stats


# --------------------------------------------------------------------------- #
# Parsing model output
# --------------------------------------------------------------------------- #
def parse_pred_json(text: str) -> dict | None:
    """Best-effort extraction of the JSON object the model was asked to emit.

    Returns the parsed dict, or None if no valid JSON object is recoverable.
    """
    if not text:
        return None
    # Fast path: the whole string is JSON.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fallback: grab the first balanced {...} span and try that.
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : end + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def norm_answer(s) -> str:
    """Normalise a final_answer for comparison: trim, lowercase, drop trailing punct."""
    if s is None:
        return ""
    return re.sub(r"[\s.]+$", "", str(s).strip()).lower()


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #
def bootstrap_ci(hits: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a mean of 0/1 hits. Pure-stdlib LCG so the
    result is deterministic and needs no numpy."""
    n = len(hits)
    if n == 0:
        return (0.0, 0.0)
    state = seed | 1
    means = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            s += hits[state % n]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (round(lo, 4), round(hi, 4))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def evaluate(pairs: list[tuple[dict, dict]]) -> dict:
    per_cell_hits = {c: defaultdict(list) for c in CELLS}  # cell -> field -> [0/1]
    json_valid = {c: [] for c in CELLS}
    exact_all = {c: [] for c in CELLS}  # all 3 fields correct AND valid json
    over_repair = []   # Clean rows predicted as should_repair=True
    under_repair = []  # non-Clean rows predicted as should_repair=False
    skill_confusion = Counter()  # (gold_skill, pred_skill)
    heldout_hits = {"heldout": [], "seen": []}  # final_answer correctness by entity_split

    for pred, raw in pairs:
        cell = raw["cell"]
        if cell not in per_cell_hits:
            continue
        obj = parse_pred_json(pred["pred_text"])
        valid = obj is not None
        json_valid[cell].append(1 if valid else 0)

        gold = {f: raw[f] for f in SCORE_FIELDS}
        field_ok = {}
        for f in SCORE_FIELDS:
            if not valid:
                ok = 0
            elif f == "final_answer":
                ok = int(norm_answer(obj.get(f)) == norm_answer(gold[f]))
            else:
                ok = int(str(obj.get(f, "")).strip() == str(gold[f]).strip())
            field_ok[f] = ok
            per_cell_hits[cell][f].append(ok)

        exact_all[cell].append(int(valid and all(field_ok.values())))

        # Failure modes from the predicted no-op signal vs the gold contract.
        # Clean is the only no-repair cell (gold: diagnosis=no_failure_detected,
        # repair_skill=keep_answer). A prediction is "no repair" when it emits
        # either no-op signal. Invalid JSON counts as "attempted a repair" — the
        # model failed to produce the keep-answer no-op, which is itself a fault.
        pred_skill = str(obj.get("repair_skill", "")).strip() if valid else ""
        pred_diag = str(obj.get("diagnosis", "")).strip() if valid else ""
        pred_no_repair = valid and (
            pred_skill == NO_REPAIR_SKILL or pred_diag == NO_REPAIR_DIAGNOSIS
        )
        pred_should_repair = not pred_no_repair

        if cell == "Clean":
            over_repair.append(int(pred_should_repair))
        else:
            under_repair.append(int(not pred_should_repair))
            if valid:
                skill_confusion[(str(gold["repair_skill"]).strip(), pred_skill)] += 1

        # Held-out generalization on final_answer.
        bucket = "heldout" if raw.get("entity_split") == "eval" else "seen"
        heldout_hits[bucket].append(field_ok["final_answer"])

    def summ(hits):
        n = len(hits)
        acc = sum(hits) / n if n else 0.0
        lo, hi = bootstrap_ci(hits)
        return {"n": n, "acc": round(acc, 4), "ci95": [lo, hi]}

    report = {"per_cell": {}, "overall": {}, "failure_modes": {}, "generalization": {}}

    for c in CELLS:
        report["per_cell"][c] = {
            "n": len(exact_all[c]),
            "json_valid": summ(json_valid[c]),
            "exact_all3": summ(exact_all[c]),
            **{f: summ(per_cell_hits[c][f]) for f in SCORE_FIELDS},
        }

    # Overall (micro over all rows).
    all_exact = [h for c in CELLS for h in exact_all[c]]
    all_valid = [h for c in CELLS for h in json_valid[c]]
    report["overall"] = {
        "n": len(all_exact),
        "json_valid": summ(all_valid),
        "exact_all3": summ(all_exact),
        **{f: summ([h for c in CELLS for h in per_cell_hits[c][f]]) for f in SCORE_FIELDS},
    }

    report["failure_modes"] = {
        "over_repair_rate_on_clean": summ(over_repair),
        "under_repair_rate_on_nonclean": summ(under_repair),
        "skill_confusion": [
            {"gold": g, "pred": p, "count": n}
            for (g, p), n in sorted(skill_confusion.items(), key=lambda kv: -kv[1])
        ],
    }

    report["generalization"] = {
        "final_answer_heldout_entities": summ(heldout_hits["heldout"]),
        "final_answer_seen_entities": summ(heldout_hits["seen"]),
    }
    return report


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def render_md(report: dict, align_stats: dict) -> str:
    def pct(d):
        return f"{d['acc']*100:.1f}% [{d['ci95'][0]*100:.1f}, {d['ci95'][1]*100:.1f}] (n={d['n']})"

    L = ["# Atomic-repair eval report", ""]
    L.append(
        f"Alignment: {align_stats['by_gold_match']} by gold-match, "
        f"{align_stats['by_index']} by index, {align_stats['unmatched']} unmatched."
    )
    L.append("")
    o = report["overall"]
    L += [
        "## Overall",
        "",
        f"- JSON-valid: {pct(o['json_valid'])}",
        f"- Exact (all 3 fields): {pct(o['exact_all3'])}",
        f"- diagnosis: {pct(o['diagnosis'])}",
        f"- repair_skill: {pct(o['repair_skill'])}",
        f"- final_answer: {pct(o['final_answer'])}",
        "",
        "## Per cell",
        "",
        "| cell | n | json-valid | exact (3/3) | diagnosis | repair_skill | final_answer |",
        "|------|---|-----------|-------------|-----------|--------------|--------------|",
    ]
    for c in CELLS:
        d = report["per_cell"][c]
        L.append(
            f"| {c} | {d['n']} | {pct(d['json_valid'])} | {pct(d['exact_all3'])} | "
            f"{pct(d['diagnosis'])} | {pct(d['repair_skill'])} | {pct(d['final_answer'])} |"
        )

    fm = report["failure_modes"]
    L += [
        "",
        "## Failure modes",
        "",
        f"- **Over-repair on Clean** (model 'fixes' a correct answer): {pct(fm['over_repair_rate_on_clean'])}",
        f"- **Under-repair on non-Clean** (model misses a real failure): {pct(fm['under_repair_rate_on_nonclean'])}",
        "",
        "### Repair-skill confusion (gold -> pred, non-Clean, valid JSON only)",
        "",
        "| gold skill | predicted skill | count |",
        "|------------|-----------------|-------|",
    ]
    for row in fm["skill_confusion"][:20]:
        flag = "" if row["gold"] == row["pred"] else "  <-- mismatch"
        L.append(f"| {row['gold']} | {row['pred']} | {row['count']}{flag} |")

    g = report["generalization"]
    L += [
        "",
        "## Generalization (final_answer accuracy)",
        "",
        f"- Held-out entities: {pct(g['final_answer_heldout_entities'])}",
        f"- Seen entities: {pct(g['final_answer_seen_entities'])}",
        "",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path, help="generated_predictions.jsonl from LLaMA-Factory")
    ap.add_argument("--raw", required=True, type=Path, help="repair_raw_eval.jsonl (gold + cell labels)")
    ap.add_argument("--out", type=Path, default=Path("data/eval_report.json"))
    ap.add_argument("--report", type=Path, default=Path("data/eval_report.md"))
    args = ap.parse_args()

    raws = load_raw(args.raw)
    preds = load_pred(args.pred)
    print(f"loaded {len(preds)} predictions, {len(raws)} raw eval rows")
    if len(preds) != len(raws):
        print(f"WARNING: count mismatch ({len(preds)} preds vs {len(raws)} raw). "
              "Alignment falls back to gold-match where possible.")

    pairs, align_stats = align(preds, raws)
    print(f"aligned: {align_stats}")

    report = evaluate(pairs)
    report["_alignment"] = align_stats

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = render_md(report, align_stats)
    args.report.write_text(md)

    print(f"\nwrote {args.out} and {args.report}\n")
    print(md)


if __name__ == "__main__":
    main()
