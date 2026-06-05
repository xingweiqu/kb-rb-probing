"""Zero-shot prompt builders for condition A (base Qwen3-8B-Instruct, no training).

Three prompt styles over the SAME Problem/Tentative body, so we can ask: does
prompting alone repair, and does a skill prompt help with no training?

  direct    : ask for only the corrected final answer
  cot       : ask to think step by step, end with "Final answer: <answer>"
  skillcot  : ask for diagnosis + repair_skill + reasoning + final answer as JSON

The alpaca `output` is left empty (zero-shot has no gold target to train on);
the evaluator joins predictions back to repair_eval by row index, not by gold
string. The Problem/Tentative body matches convert_v1.repair_input exactly.
"""
from __future__ import annotations


def _body(rec: dict) -> str:
    return f"Problem:\n{rec['problem']}\n\nTentative answer:\n{rec['tentative_answer']}"


DIRECT_INSTRUCTION = (
    "Given the problem and a tentative answer that may be wrong, output ONLY the "
    "corrected final answer as a single short line. No explanation."
)

COT_INSTRUCTION = (
    "Given the problem and a tentative answer that may be wrong, think step by "
    "step about whether the tentative answer is correct using facts you know, "
    "then give the corrected answer. End with a line exactly of the form "
    "'Final answer: <answer>'."
)

SKILLCOT_INSTRUCTION = (
    "Given the problem and a tentative answer that may be wrong, first state a "
    "diagnosis and the repair skill, then reason step by step, then give the "
    "final answer. Respond as JSON with keys diagnosis, repair_skill, "
    "repair_trace, final_answer."
)


def direct_alpaca(rec: dict) -> dict:
    return {"instruction": DIRECT_INSTRUCTION, "input": _body(rec), "output": ""}


def cot_alpaca(rec: dict) -> dict:
    return {"instruction": COT_INSTRUCTION, "input": _body(rec), "output": ""}


def skillcot_alpaca(rec: dict) -> dict:
    return {"instruction": SKILLCOT_INSTRUCTION, "input": _body(rec), "output": ""}
