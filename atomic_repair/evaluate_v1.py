"""Score atomic-repair v1 predictions. CPU-only, stdlib-only.

Two modes:
  --mode fact    : fact-QA accuracy vs fact_eval.jsonl. With --sanity it also acts
                   as the cleanliness/learned gate (prints leaked facts the model
                   already got right; PASS iff accuracy <= --sanity-thresh).
  --mode repair  : per-cell repair metrics vs repair_eval.jsonl, TOLERANT of
                   missing diagnosis/repair_skill (CoT-only & zero-shot have none).

Alignment is by ROW INDEX against --eval-source (zero-shot has no gold string to
match on; LF preserves dataset order, so index join is exact).

Skill-free repair signal (works for every condition):
  kept    := final_answer == tentative_answer   (model left the answer)
  changed := otherwise                          (model attempted a repair)
  over-repair (Clean)     = changed a correct answer
  under-repair (non-Clean)= failed to change a wrong answer
Accept-rates (skill-free):
  H-Cor wrong_bridge_accept = mean(final == planted_wrong_answer)
  K-Cor wrong_claim_accept  = mean(final == planted_wrong_answer)
Skill-based metrics (diagnosis/skill/confusion) are added only when present.

Each report stores per-row final_answer hit vectors so compare_conditions.py can
bootstrap deltas.

Usage:
  python evaluate_v1.py --mode fact   --pred P --eval-source data_v1/fact_eval.jsonl   --out R.json [--sanity --sanity-thresh 0.05]
  python evaluate_v1.py --mode repair --pred P --eval-source data_v1/repair_eval.jsonl --out R.json [--report R.md]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_predictions import load_pred, parse_pred_json, norm_answer, bootstrap_ci

CELLS = ["H-Aug", "H-Abl", "H-Cor", "K-Cor", "Clean"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_final_answer(text: str) -> str:
    """Pull a final answer from a model output in any of the 3 shapes:
    JSON {final_answer:...}, a 'Final answer: X' line, or a bare last line."""
    obj = parse_pred_json(text)
    if obj and "final_answer" in obj:
        return str(obj["final_answer"])
    m = re.search(r"final answer\s*[:\-]\s*(.+)", text or "", re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0]
    # bare: last non-empty line
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_skill(text: str) -> tuple[str | None, str | None]:
    """Return (diagnosis, repair_skill) if the output is JSON carrying them."""
    obj = parse_pred_json(text)
    if obj:
        return obj.get("diagnosis"), obj.get("repair_skill")
    return None, None


# --------------------------------------------------------------------------- #
def eval_fact(pred_path: Path, src_path: Path, out_path: Path,
              sanity: bool, thresh: float) -> None:
    preds = load_pred(pred_path)
    src = load_jsonl(src_path)
    n = min(len(preds), len(src))
    hits, by_hop, by_fam, leaked = [], defaultdict(list), defaultdict(list), []
    for i in range(n):
        gold = norm_answer(src[i]["answer"])
        got = norm_answer(extract_final_answer(preds[i]["pred_text"]))
        # fact answers are short; also accept exact-substring of the gold entity.
        hit = int(got == gold or (gold and gold in got))
        hits.append(hit)
        by_hop[src[i]["hop"]].append(hit)
        by_fam[src[i]["relation_family"]].append(hit)
        if hit and sanity:
            leaked.append({"q": src[i]["question"], "a": src[i]["answer"]})

    def stat(h):
        return {"n": len(h), "acc": round(sum(h) / len(h), 4) if h else 0.0,
                "ci95": bootstrap_ci(h)}

    acc = sum(hits) / len(hits) if hits else 0.0
    report = {
        "mode": "fact", "overall": stat(hits),
        "by_hop": {k: stat(v) for k, v in by_hop.items()},
        "by_family": {k: stat(v) for k, v in by_fam.items()},
        "hit_vector": hits,
    }
    if sanity:
        report["sanity"] = {
            "threshold": thresh,
            "passed": acc <= thresh,
            "leaked_count": len(leaked),
            "leaked_examples": leaked[:20],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    msg = f"[fact] acc={acc:.3f} (n={len(hits)})"
    if sanity:
        verdict = "PASS (clean)" if acc <= thresh else "FAIL (possible leakage)"
        msg += f"  sanity {verdict}; {len(leaked)} facts already known"
    print(msg, "->", out_path)


# --------------------------------------------------------------------------- #
def eval_repair(pred_path: Path, src_path: Path, out_path: Path,
                report_path: Path | None) -> None:
    preds = load_pred(pred_path)
    src = load_jsonl(src_path)
    n = min(len(preds), len(src))

    per_cell = {c: {"final": [], "json": [], "diag": [], "skill": []} for c in CELLS}
    over_repair, under_repair = [], []          # skill-free
    hcor_accept, kcor_accept = [], []
    skill_conf = Counter()
    has_skill_any = False

    for i in range(n):
        r = src[i]
        cell = r["cell"]
        text = preds[i]["pred_text"]
        gold = norm_answer(r["gold_answer"])
        tent = norm_answer(r["tentative_answer"])
        final = norm_answer(extract_final_answer(text))
        diag, skill = extract_skill(text)

        per_cell[cell]["final"].append(int(final == gold))
        per_cell[cell]["json"].append(int(parse_pred_json(text) is not None))

        changed = (final != tent)
        if cell == "Clean":
            over_repair.append(int(changed))         # changed a correct answer
        else:
            under_repair.append(int(not changed))    # failed to change a wrong one

        if cell == "H-Cor":
            hcor_accept.append(int(final == norm_answer(r["planted_wrong_answer"])))
        if cell == "K-Cor":
            kcor_accept.append(int(final == norm_answer(r["planted_wrong_answer"])))

        # skill-based, only when the model emitted them
        if diag is not None:
            has_skill_any = True
            per_cell[cell]["diag"].append(int(diag == r["diagnosis"]))
        if skill is not None:
            has_skill_any = True
            per_cell[cell]["skill"].append(int(skill == r["repair_skill"]))
            if cell != "Clean":
                skill_conf[(r["repair_skill"], skill)] += 1

    def stat(h):
        return {"n": len(h), "acc": round(sum(h) / len(h), 4) if h else 0.0,
                "ci95": bootstrap_ci(h)} if h else {"n": 0, "acc": None, "ci95": [0, 0]}

    overall_final = [x for c in CELLS for x in per_cell[c]["final"]]
    report = {
        "mode": "repair",
        "overall_final_answer": stat(overall_final),
        "per_cell": {c: {
            "n": len(per_cell[c]["final"]),
            "final_answer": stat(per_cell[c]["final"]),
            "json_valid": stat(per_cell[c]["json"]),
            "diagnosis": stat(per_cell[c]["diag"]),
            "repair_skill": stat(per_cell[c]["skill"]),
        } for c in CELLS},
        "failure_modes": {
            "over_repair_on_clean": stat(over_repair),
            "under_repair_on_nonclean": stat(under_repair),
            "hcor_wrong_bridge_accept": stat(hcor_accept),
            "kcor_wrong_claim_accept": stat(kcor_accept),
        },
        "has_skill_labels": has_skill_any,
        "skill_confusion": [{"gold": g, "pred": p, "count": c}
                            for (g, p), c in skill_conf.most_common()],
        "hit_vector_final": overall_final,
        "per_cell_final_vectors": {c: per_cell[c]["final"] for c in CELLS},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[repair] final_answer acc={report['overall_final_answer']['acc']} "
          f"(n={len(overall_final)}) skill_labels={has_skill_any} -> {out_path}")

    if report_path:
        _write_md(report, report_path)


def _pct(s):
    if s.get("acc") is None:
        return "n/a"
    return f"{s['acc']*100:.1f}% [{s['ci95'][0]*100:.1f}, {s['ci95'][1]*100:.1f}] (n={s['n']})"


def _write_md(report: dict, path: Path) -> None:
    L = ["# v1 repair eval", ""]
    L.append(f"Overall final_answer: {_pct(report['overall_final_answer'])}")
    L.append(f"Skill labels present: {report['has_skill_labels']}")
    L += ["", "## Per cell", "",
          "| cell | n | final_answer | json | diagnosis | repair_skill |",
          "|------|---|--------------|------|-----------|--------------|"]
    for c in CELLS:
        pc = report["per_cell"][c]
        L.append(f"| {c} | {pc['n']} | {_pct(pc['final_answer'])} | "
                 f"{_pct(pc['json_valid'])} | {_pct(pc['diagnosis'])} | {_pct(pc['repair_skill'])} |")
    fm = report["failure_modes"]
    L += ["", "## Failure modes", "",
          f"- Over-repair on Clean: {_pct(fm['over_repair_on_clean'])}",
          f"- Under-repair on non-Clean: {_pct(fm['under_repair_on_nonclean'])}",
          f"- H-Cor wrong-bridge accept: {_pct(fm['hcor_wrong_bridge_accept'])}",
          f"- K-Cor wrong-claim accept: {_pct(fm['kcor_wrong_claim_accept'])}"]
    if report["skill_confusion"]:
        L += ["", "## Skill confusion (non-Clean)", "", "| gold | pred | count |", "|---|---|---|"]
        for row in report["skill_confusion"]:
            L.append(f"| {row['gold']} | {row['pred']} | {row['count']} |")
    path.write_text("\n".join(L))
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fact", "repair"], required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--eval-source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--sanity-thresh", type=float, default=0.05)
    args = ap.parse_args()
    if args.mode == "fact":
        eval_fact(args.pred, args.eval_source, args.out, args.sanity, args.sanity_thresh)
    else:
        eval_repair(args.pred, args.eval_source, args.out, args.report)


if __name__ == "__main__":
    main()
