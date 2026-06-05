"""Convert atomic-repair v1 raw JSONL into LLaMA-Factory alpaca JSON.

Emits three training/eval formats plus the dataset_info.json that registers them:

  Fact-only   : input = fact question,            output = bare answer
  CoT-only    : input = Problem + Tentative,       output = {repair_trace, final_answer}
  Skill+CoT   : input = Problem + Tentative (SAME), output = {diagnosis, repair_skill,
                                                              repair_trace, final_answer}

The CoT-only and Skill+CoT inputs are built by ONE shared helper and asserted
byte-identical per row, so the only difference between conditions C and D is the
output supervision (whether the skill labels are present).

Usage:
  python convert_v1.py --data_dir data_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import zeroshot_prompts_v1 as zs

FACT_INSTRUCTION = (
    "Answer the question with the single correct entity. "
    "Reply with only the answer, no explanation."
)

# Shared repair instruction for BOTH CoT-only and Skill+CoT (identical input).
REPAIR_INSTRUCTION = (
    "You are given a problem and a tentative answer that may be wrong. Using the "
    "facts you know, reason about whether the tentative answer is correct, then "
    "give the corrected final answer. Return only valid JSON."
)


def repair_input(rec: dict) -> str:
    """The single source of truth for the repair input string (C and D share it)."""
    return f"Problem:\n{rec['problem']}\n\nTentative answer:\n{rec['tentative_answer']}"


def fact_alpaca(rec: dict) -> dict:
    return {
        "instruction": FACT_INSTRUCTION,
        "input": rec["question"],
        "output": rec["answer"],
    }


def cot_alpaca(rec: dict) -> dict:
    return {
        "instruction": REPAIR_INSTRUCTION,
        "input": repair_input(rec),
        "output": json.dumps(
            {"repair_trace": rec["repair_trace"], "final_answer": rec["final_answer"]},
            ensure_ascii=False),
    }


def skillcot_alpaca(rec: dict) -> dict:
    return {
        "instruction": REPAIR_INSTRUCTION,
        "input": repair_input(rec),
        "output": json.dumps(
            {"diagnosis": rec["diagnosis"], "repair_skill": rec["repair_skill"],
             "repair_trace": rec["repair_trace"], "final_answer": rec["final_answer"]},
            ensure_ascii=False),
    }


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"wrote {len(rows):5d} records -> {path}")


def write_dataset_info(out_dir: Path) -> None:
    cols = {"prompt": "instruction", "query": "input", "response": "output"}
    info = {
        "fact_v1_train":          {"file_name": "fact_lf_train.json", "columns": cols},
        "fact_v1_eval":           {"file_name": "fact_lf_eval.json", "columns": cols},
        "repair_cot_v1_train":    {"file_name": "cot_lf_train.json", "columns": cols},
        "repair_cot_v1_eval":     {"file_name": "cot_lf_eval.json", "columns": cols},
        "repair_skillcot_v1_train": {"file_name": "skillcot_lf_train.json", "columns": cols},
        "repair_skillcot_v1_eval":  {"file_name": "skillcot_lf_eval.json", "columns": cols},
        "repair_zeroshot_direct_v1":   {"file_name": "zeroshot_direct.json", "columns": cols},
        "repair_zeroshot_cot_v1":      {"file_name": "zeroshot_cot.json", "columns": cols},
        "repair_zeroshot_skillcot_v1": {"file_name": "zeroshot_skillcot.json", "columns": cols},
    }
    p = out_dir / "dataset_info.json"
    p.write_text(json.dumps(info, indent=2))
    print(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data_v1"))
    args = ap.parse_args()
    d = args.data_dir

    fact_train = load_jsonl(d / "fact_train.jsonl")
    fact_eval = load_jsonl(d / "fact_eval.jsonl")
    repair_train = load_jsonl(d / "repair_train.jsonl")
    repair_eval = load_jsonl(d / "repair_eval.jsonl")

    # Fact-only.
    write_json([fact_alpaca(r) for r in fact_train], d / "fact_lf_train.json")
    write_json([fact_alpaca(r) for r in fact_eval], d / "fact_lf_eval.json")

    # CoT-only and Skill+CoT: assert identical inputs row-by-row.
    cot_tr, sk_tr = [], []
    for r in repair_train:
        c, s = cot_alpaca(r), skillcot_alpaca(r)
        assert c["input"] == s["input"], f"input drift at {r['id']}"
        cot_tr.append(c); sk_tr.append(s)
    cot_ev, sk_ev = [], []
    for r in repair_eval:
        c, s = cot_alpaca(r), skillcot_alpaca(r)
        assert c["input"] == s["input"], f"input drift at {r['id']}"
        cot_ev.append(c); sk_ev.append(s)
    write_json(cot_tr, d / "cot_lf_train.json")
    write_json(cot_ev, d / "cot_lf_eval.json")
    write_json(sk_tr, d / "skillcot_lf_train.json")
    write_json(sk_ev, d / "skillcot_lf_eval.json")
    print("OK: cot_input == skillcot_input for every repair row")

    # Zero-shot eval prompt files (built from repair_eval; same body, 3 instructions).
    write_json([zs.direct_alpaca(r) for r in repair_eval], d / "zeroshot_direct.json")
    write_json([zs.cot_alpaca(r) for r in repair_eval], d / "zeroshot_cot.json")
    write_json([zs.skillcot_alpaca(r) for r in repair_eval], d / "zeroshot_skillcot.json")

    write_dataset_info(d)


if __name__ == "__main__":
    main()
