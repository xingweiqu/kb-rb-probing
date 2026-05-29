"""Convert raw atomic-repair JSONL into LLaMA-Factory alpaca-style JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

INSTRUCTION = (
    "You are an atomic repair agent. Given a problem and a tentative answer, "
    "diagnose whether there is an atomic-capacity failure. If there is a failure, "
    "choose the correct repair skill and repair the answer. If there is no failure, "
    "do not over-repair. Return only valid JSON."
)


def convert_record(r: dict) -> dict:
    inp = f"Problem:\n{r['problem']}\n\nTentative answer:\n{r['tentative_answer']}"
    out = {
        "diagnosis":    r["diagnosis"],
        "repair_skill": r["repair_skill"],
        "repair_trace": r["repair_trace"],
        "final_answer": r["final_answer"],
    }
    return {
        "instruction": INSTRUCTION,
        "input": inp,
        "output": json.dumps(out, ensure_ascii=False),
    }


def write_alpaca(records: list[dict], path: Path):
    out = [convert_record(r) for r in records]
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out):5d} records -> {path}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_dataset_info(out_dir: Path):
    info = {
        "atomic_repair_train": {
            "file_name": "repair_lf_train.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "atomic_repair_eval": {
            "file_name": "repair_lf_eval.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    p = out_dir / "dataset_info.json"
    p.write_text(json.dumps(info, indent=2))
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--eval",  required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(args.train)
    eval_ = load_jsonl(args.eval)

    write_alpaca(train, args.out_dir / "repair_lf_train.json")
    write_alpaca(eval_, args.out_dir / "repair_lf_eval.json")
    write_dataset_info(args.out_dir)


if __name__ == "__main__":
    main()
