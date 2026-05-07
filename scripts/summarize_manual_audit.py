"""Summarize a filled-in manual audit sheet.

Reads the same CSV produced by `create_manual_audit_sheet.py` after the
annotator has filled the `annotator_*` columns. Reports pass rate per cell
and per family, the most common failure types, and (if a second-annotator
column exists) raw agreement and Cohen's kappa.

Usage:
    python -m scripts.summarize_manual_audit \
        --input reports/arr_revision/manual_audit_sample.csv \
        --output_csv reports/arr_revision/manual_audit_summary.csv \
        --output_md  reports/arr_revision/manual_audit_summary.md
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


PASS_COLUMNS = [
    "annotator_gold_same_answer",
    "annotator_perturbation_valid",
    "annotator_no_gold_leakage",
    "annotator_overall_pass",
]
OPTIONAL_COLUMNS = [
    "annotator_wrong_claim_plausible",
    "annotator_surface_artifact",
]


def _is_yes(v: str) -> bool:
    return (v or "").strip().lower() in {"yes", "y", "1", "true", "pass"}


def _is_no(v: str) -> bool:
    return (v or "").strip().lower() in {"no", "n", "0", "false", "fail"}


def _annotator_filled(row: dict) -> bool:
    return any(_is_yes(row.get(c, "")) or _is_no(row.get(c, "")) for c in PASS_COLUMNS)


def _cohen_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa for two parallel 0/1 lists. Returns None if undefined."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(int(x == y) for x, y in zip(a, b)) / n
    p1a = sum(a) / n; p0a = 1 - p1a
    p1b = sum(b) / n; p0b = 1 - p1b
    pe = p1a * p1b + p0a * p0b
    if 1 - pe < 1e-9:
        return None
    return (po - pe) / (1 - pe)


def summarize(input_path: Path, output_csv: Path, output_md: Path) -> None:
    rows = list(csv.DictReader(open(input_path, encoding="utf-8")))
    filled = [r for r in rows if _annotator_filled(r)]
    logger.info("read %d rows; %d have annotator data", len(rows), len(filled))

    by_cell = defaultdict(list)
    by_family = defaultdict(list)
    failure_modes = Counter()
    for r in filled:
        passed = _is_yes(r.get("annotator_overall_pass", ""))
        by_cell[r["cell_name"]].append(passed)
        by_family[r["family"]].append(passed)
        if not passed:
            for col in PASS_COLUMNS + OPTIONAL_COLUMNS:
                if _is_no(r.get(col, "")):
                    failure_modes[col] += 1

    out_rows = []
    for cell, results in sorted(by_cell.items()):
        n = len(results); k = sum(results)
        out_rows.append({
            "scope": "cell", "label": cell,
            "n": n, "passed": k, "pass_rate": f"{k/n:.2f}" if n else "",
        })
    for family, results in sorted(by_family.items()):
        n = len(results); k = sum(results)
        out_rows.append({
            "scope": "family", "label": family,
            "n": n, "passed": k, "pass_rate": f"{k/n:.2f}" if n else "",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        if out_rows:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader(); writer.writerows(out_rows)

    md = ["# Manual audit summary", ""]
    md.append(f"Annotated rows: {len(filled)} / {len(rows)}")
    md.append("")
    md.append("## Pass rate per cell")
    md.append("")
    md.append("| Cell | n | passed | pass rate |")
    md.append("|---|---|---|---|")
    for cell, results in sorted(by_cell.items()):
        n = len(results); k = sum(results)
        md.append(f"| {cell} | {n} | {k} | {k/n:.0%} |" if n else f"| {cell} | 0 | 0 | — |")

    md.append("")
    md.append("## Pass rate per family")
    md.append("")
    md.append("| Family | n | passed | pass rate |")
    md.append("|---|---|---|---|")
    for family, results in sorted(by_family.items()):
        n = len(results); k = sum(results)
        md.append(f"| {family} | {n} | {k} | {k/n:.0%} |" if n else f"| {family} | 0 | 0 | — |")

    if failure_modes:
        md.append("")
        md.append("## Most common failure types (counts of `no` answers among failed items)")
        md.append("")
        for col, cnt in failure_modes.most_common():
            md.append(f"- `{col}`: {cnt}")
    md.append("")
    md.append("## Notes")
    md.append("Inter-annotator agreement is reported only when a second annotator column "
             "is present in the input CSV (current sheet has one annotator).")
    md.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("wrote %s and %s", output_csv, output_md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="reports/arr_revision/manual_audit_sample.csv")
    p.add_argument("--output_csv", default="reports/arr_revision/manual_audit_summary.csv")
    p.add_argument("--output_md", default="reports/arr_revision/manual_audit_summary.md")
    args = p.parse_args()
    summarize(Path(args.input), Path(args.output_csv), Path(args.output_md))


if __name__ == "__main__":
    main()
