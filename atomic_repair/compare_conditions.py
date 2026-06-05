"""Build the v1 comparison report from per-condition eval JSON files.

Ingests the eval reports produced by evaluate_v1.py and emits comparison_v1.md
(+ .json). Computes the spec's contrasts and a two-gate banner.

Expected inputs (pass any subset; missing ones are skipped with a note):
  --fact-sanity-base   base/Instruct fact-QA sanity report (cleanliness gate)
  --fact-gate          Fact-only fact-QA report (learned gate)
  --zeroshot-direct    --zeroshot-cot   --zeroshot-skillcot   (condition A)
  --factonly-repair    (B on repair)
  --fact-then-cot      (C)
  --fact-then-skillcot (D)

Contrasts emitted:
  zero-shot CoT vs zero-shot Skill+CoT
  Fact-only(repair) vs zero-shot(best)
  Fact->CoT vs Fact-only
  Fact->Skill+CoT vs Fact->CoT
Plus a table of Clean over-repair, H-Cor accept, K-Cor accept across conditions.

Usage: python compare_conditions.py --out comparison_v1 [--<cond> report.json ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_predictions import bootstrap_ci


def load(path: Path | None) -> dict | None:
    if path and path.exists():
        return json.loads(path.read_text())
    return None


def final_acc(rep: dict | None) -> float | None:
    if not rep:
        return None
    return rep.get("overall_final_answer", {}).get("acc")


def delta_ci(a: dict | None, b: dict | None) -> dict:
    """Bootstrap CI on (a - b) final-answer accuracy using stored hit vectors."""
    if not a or not b:
        return {"delta": None, "ci95": None}
    va, vb = a.get("hit_vector_final", []), b.get("hit_vector_final", [])
    if not va or not vb:
        return {"delta": None, "ci95": None}
    da = sum(va) / len(va) - sum(vb) / len(vb)
    # paired-by-index where lengths match (same eval set), else unpaired means.
    if len(va) == len(vb):
        diffs = [va[i] - vb[i] for i in range(len(va))]
        # bootstrap mean of diffs (reuse 0/1-style CI on shifted values won't work;
        # do a simple percentile bootstrap on diffs).
        ci = _boot_mean(diffs)
    else:
        ci = None
    return {"delta": round(da, 4), "ci95": ci}


def _boot_mean(vals: list[float], n_boot: int = 2000, seed: int = 0):
    n = len(vals)
    if n == 0:
        return None
    state = seed | 1
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            s += vals[state % n]
        means.append(s / n)
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)]


def fm_acc(rep: dict | None, key: str) -> float | None:
    if not rep:
        return None
    return rep.get("failure_modes", {}).get(key, {}).get("acc")


def pct(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("comparison_v1"))
    ap.add_argument("--fact-sanity-base", type=Path)
    ap.add_argument("--fact-gate", type=Path)
    ap.add_argument("--zeroshot-direct", type=Path)
    ap.add_argument("--zeroshot-cot", type=Path)
    ap.add_argument("--zeroshot-skillcot", type=Path)
    ap.add_argument("--factonly-repair", type=Path)
    ap.add_argument("--fact-then-cot", type=Path)
    ap.add_argument("--fact-then-skillcot", type=Path)
    ap.add_argument("--sanity-thresh", type=float, default=0.05)
    ap.add_argument("--gate-thresh", type=float, default=0.80)
    args = ap.parse_args()

    sanity = load(args.fact_sanity_base)
    gate = load(args.fact_gate)
    zd = load(args.zeroshot_direct)
    zc = load(args.zeroshot_cot)
    zs = load(args.zeroshot_skillcot)
    fo = load(args.factonly_repair)
    fc = load(args.fact_then_cot)
    fs = load(args.fact_then_skillcot)

    conds = [
        ("A. zero-shot direct", zd), ("A. zero-shot CoT", zc),
        ("A. zero-shot Skill+CoT", zs), ("B. Fact-only", fo),
        ("C. Fact->CoT", fc), ("D. Fact->Skill+CoT", fs),
    ]

    # ---- gates ----
    sanity_acc = sanity.get("overall", {}).get("acc") if sanity else None
    sanity_pass = (sanity_acc is not None and sanity_acc <= args.sanity_thresh)
    gate_acc = gate.get("overall", {}).get("acc") if gate else None
    gate_pass = (gate_acc is not None and gate_acc >= args.gate_thresh)

    # best zero-shot for the Fact-only vs zero-shot contrast
    zs_reports = [r for r in (zd, zc, zs) if r]
    best_zs = max(zs_reports, key=lambda r: final_acc(r) or 0) if zs_reports else None

    contrasts = {
        "zeroshot_cot_vs_skillcot": {
            "a": "zero-shot Skill+CoT", "b": "zero-shot CoT", **delta_ci(zs, zc)},
        "factonly_vs_best_zeroshot": {
            "a": "Fact-only", "b": "best zero-shot", **delta_ci(fo, best_zs)},
        "fact_then_cot_vs_factonly": {
            "a": "Fact->CoT", "b": "Fact-only", **delta_ci(fc, fo)},
        "fact_then_skillcot_vs_cot": {
            "a": "Fact->Skill+CoT", "b": "Fact->CoT", **delta_ci(fs, fc)},
    }

    out_json = {
        "gates": {
            "cleanliness": {"base_fact_acc": sanity_acc, "thresh": args.sanity_thresh,
                            "passed": sanity_pass},
            "learned": {"factonly_fact_acc": gate_acc, "thresh": args.gate_thresh,
                        "passed": gate_pass},
        },
        "final_answer_by_condition": {name: final_acc(r) for name, r in conds},
        "contrasts": contrasts,
        "failure_modes_by_condition": {
            name: {
                "clean_over_repair": fm_acc(r, "over_repair_on_clean"),
                "hcor_accept": fm_acc(r, "hcor_wrong_bridge_accept"),
                "kcor_accept": fm_acc(r, "kcor_wrong_claim_accept"),
            } for name, r in conds},
    }
    args.out.with_suffix(".json").write_text(json.dumps(out_json, indent=2, ensure_ascii=False))

    # ---- markdown ----
    L = ["# Atomic-repair v1 — condition comparison", ""]
    L += ["## Gates (must pass before interpreting repair)", ""]
    sv = "PASS ✅" if sanity_pass else ("FAIL ⚠️" if sanity_acc is not None else "n/a")
    gv = "PASS ✅" if gate_pass else ("FAIL ⚠️" if gate_acc is not None else "n/a")
    L.append(f"- **Cleanliness** (base/Instruct fact-QA ≤ {pct(args.sanity_thresh)}): "
             f"{pct(sanity_acc)} → {sv}  "
             f"{'' if sanity_pass or sanity_acc is None else '— entities may be contaminated; repair numbers suspect'}")
    L.append(f"- **Learned** (Fact-only fact-QA ≥ {pct(args.gate_thresh)}): "
             f"{pct(gate_acc)} → {gv}  "
             f"{'' if gate_pass or gate_acc is None else '— knowledge stage failed; do NOT interpret repair'}")

    L += ["", "## Final-answer accuracy by condition", "",
          "| condition | final_answer acc |", "|---|---|"]
    for name, r in conds:
        L.append(f"| {name} | {pct(final_acc(r))} |")

    L += ["", "## Key contrasts (Δ = a − b on final_answer)", "",
          "| contrast | a | b | Δ | 95% CI on Δ |", "|---|---|---|---|---|"]
    labels = {
        "zeroshot_cot_vs_skillcot": "does a skill prompt help with NO training?",
        "factonly_vs_best_zeroshot": "does adding knowledge alone help?",
        "fact_then_cot_vs_factonly": "does CoT trajectory help use known facts?",
        "fact_then_skillcot_vs_cot": "does the skill label add over plain CoT?",
    }
    for key, c in contrasts.items():
        ci = c["ci95"]
        ci_s = "n/a" if ci is None else f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]"
        dval = "n/a" if c["delta"] is None else f"{c['delta']*100:+.1f}%"
        L.append(f"| {labels[key]} | {c['a']} | {c['b']} | {dval} | {ci_s} |")

    L += ["", "## Failure modes by condition (lower is better)", "",
          "| condition | Clean over-repair | H-Cor accept | K-Cor accept |",
          "|---|---|---|---|"]
    for name, r in conds:
        L.append(f"| {name} | {pct(fm_acc(r,'over_repair_on_clean'))} | "
                 f"{pct(fm_acc(r,'hcor_wrong_bridge_accept'))} | "
                 f"{pct(fm_acc(r,'kcor_wrong_claim_accept'))} |")

    L += ["", "_Δ CIs that exclude 0 indicate a statistically meaningful difference._"]
    args.out.with_suffix(".md").write_text("\n".join(L))
    print("wrote", args.out.with_suffix(".md"), "and", args.out.with_suffix(".json"))


if __name__ == "__main__":
    main()
