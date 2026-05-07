"""Derive 2x3 atomic capability labels + CoT state from model behavior.

Two judgment systems are emitted side-by-side:

1. ``binary`` — based on the soft-correct flag in model_outputs.jsonl.
   Provide axis    : ¬orig_correct ∧ variant_correct  -> True
   Block / Distract: orig_correct  ∧ ¬variant_correct -> True
   "True ∧ True" -> False; otherwise None (uninformative).
   Loses signal when both items are correct, which is what happens with
   strong models on easy data.

2. ``delta`` — based on Δlogprob of the gold answer:
       delta = mean_logprob(gold | variant) - mean_logprob(gold | original)
   Provide axis: delta > +tau -> True, delta < -tau -> False
   Block axis  : delta < -tau -> True, delta > +tau -> False
   Distract    : delta < -tau -> True, delta > +tau -> False
   Tau is in nats (default 0.1 ≈ ~10% probability change). If the model
   was already wildly wrong on both, delta still moves and produces a
   real label — that is the point.

Output schema:

    labels[family_id][cot_state] = {
        "binary":  {capability: True | False | null},
        "delta":   {capability: True | False | null},
        "delta_value": {capability: float},   # raw Δlogprob (mean per token)
    }

Capability map (mechanism × perturbation):
    KP  knowledge surfacing            (Knowledge × Provide)
    KB  retrieval dependence           (Knowledge × Block)
    KD  retrieval robustness           (Knowledge × Distract)
    RP  rule activation                (Reasoning × Provide)
    RB  rule dependence                (Reasoning × Block)
    RD  reasoning robustness           (Reasoning × Distract)
    CP  gap-filling                    (Composition × Provide)
    CB  bottleneck identification      (Composition × Block)

Variant -> primary capability:
    KB-hint                  -> KP        (provide)
    KB-wrongclaim            -> KD        (distract)
    KB-paraphrase            -> KB        (block)
    RB-scaffold              -> RP        (provide)
    RB-rule_removal          -> RB        (block)
    RB-wrong_intermediate    -> RD        (distract)
    Hybrid-explicit_fact     -> CP        (provide)
    Hybrid-retrieval_blocked -> CB        (block)
    Hybrid-both_blocked      -> CB-control(block; double-block lower bound)
"""

import argparse
import json
import logging
import math
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


VARIANT_TO_CAPABILITY = {
    "hint": ("KP", "provide"),
    "wrongclaim": ("KD", "distract"),
    "paraphrase": ("KB", "block"),
    "scaffold": ("RP", "provide"),
    "rule_removal": ("RB", "block"),
    "wrong_intermediate": ("RD", "distract"),
    "explicit_fact": ("CP", "provide"),
    "retrieval_blocked": ("CB", "block"),
    "both_blocked": ("CB-control", "block"),
}

CAPABILITIES = ["KP", "KD", "KB", "RP", "RD", "RB", "CP", "CB", "CB-control"]
COT_STATES = ["no_cot", "with_cot"]


def _judge_binary(orig_correct: bool, variant_correct: bool, axis: str) -> bool | None:
    if axis == "provide":
        if not orig_correct and variant_correct:
            return True
        if orig_correct and variant_correct:
            return False
        return None
    # block / distract
    if orig_correct and not variant_correct:
        return True
    if orig_correct and variant_correct:
        return False
    return None


def _judge_delta(delta: float, axis: str, tau: float) -> bool | None:
    """Continuous-aware label: returns True/False/None based on signed delta.

    delta is mean-per-token logprob change (variant - original). The sign
    convention assumes positive means the variant made the gold answer more
    likely.
    """
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return None
    if axis == "provide":
        if delta > tau:
            return True
        if delta < -tau:
            return False
        return None
    # block / distract: capability is "True" if perturbation hurt the model
    if delta < -tau:
        return True
    if delta > tau:
        return False
    return None


def _output_key(family_id: str, variant: str, mode: str, cot_state: str) -> str:
    return f"{family_id}_{variant}_{mode}_{cot_state}"


def _safe_get_lp(record: dict[str, Any]) -> float | None:
    v = record.get("gold_logprob_mean")
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) else None


def derive_atomic_labels(
    dataset_path: str,
    model_outputs_path: str,
    output_path: str,
    cot_states: list[str] | None = None,
    mode: str = "natural",
    tau: float = 0.1,
) -> dict[str, dict[str, dict[str, Any]]]:
    cot_states = cot_states or COT_STATES

    with open(dataset_path, encoding="utf-8") as f:
        items_by_key = {
            (it := json.loads(line))["family_id"] + "_" + it["variant"] + "_" + it.get("mode", "natural"): it
            for line in f
        }

    outputs_by_key: dict[str, dict[str, Any]] = {}
    with open(model_outputs_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            cot = o.get("cot_state", "no_cot")
            outputs_by_key[_output_key(o["family_id"], o["variant"], o.get("mode", "natural"), cot)] = o

    families = sorted({it["family_id"] for it in items_by_key.values()})
    labels: dict[str, dict[str, dict[str, Any]]] = {}
    has_logprob = False

    for family_id in families:
        labels[family_id] = {}
        for cot in cot_states:
            cell_binary: dict[str, bool | None] = {}
            cell_delta: dict[str, bool | None] = {}
            cell_value: dict[str, float | None] = {}
            orig = outputs_by_key.get(_output_key(family_id, "original", mode, cot))
            if not orig:
                labels[family_id][cot] = {
                    "binary": cell_binary, "delta": cell_delta, "delta_value": cell_value,
                }
                continue
            orig_correct = bool(orig.get("correct", False))
            orig_lp = _safe_get_lp(orig)

            for variant, (capability, axis) in VARIANT_TO_CAPABILITY.items():
                v = outputs_by_key.get(_output_key(family_id, variant, mode, cot))
                if not v:
                    continue
                v_lp = _safe_get_lp(v)
                cell_binary[capability] = _judge_binary(
                    orig_correct, bool(v.get("correct", False)), axis,
                )
                if orig_lp is not None and v_lp is not None:
                    has_logprob = True
                    delta = v_lp - orig_lp
                    cell_value[capability] = float(delta)
                    cell_delta[capability] = _judge_delta(delta, axis, tau)
                else:
                    cell_value[capability] = None
                    cell_delta[capability] = None

            labels[family_id][cot] = {
                "binary": cell_binary, "delta": cell_delta, "delta_value": cell_value,
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    _log_summary(labels, cot_states, has_logprob, tau)
    return labels


def _summary_block(
    labels: dict[str, dict[str, dict[str, Any]]],
    cot_states: list[str],
    judge_key: str,
    title: str,
) -> None:
    logger.info("[%s]", title)
    header = f"{'capability':14s} | " + " | ".join(f"{c:>22s}" for c in cot_states)
    logger.info(header)
    logger.info("-" * len(header))
    for cap in CAPABILITIES:
        cells = []
        for cot in cot_states:
            vals = []
            for fam in labels:
                bucket = labels[fam].get(cot, {}).get(judge_key, {})
                if cap in bucket:
                    vals.append(bucket[cap])
            pos = vals.count(True)
            neg = vals.count(False)
            drop = vals.count(None)
            cells.append(f"pos={pos:2d} neg={neg:2d} drop={drop:2d}")
        logger.info(f"{cap:14s} | " + " | ".join(f"{c:>22s}" for c in cells))


def _log_summary(
    labels: dict[str, dict[str, dict[str, Any]]],
    cot_states: list[str],
    has_logprob: bool,
    tau: float,
) -> None:
    logger.info("=" * 72)
    logger.info("LABEL SUMMARY (capability × cot_state)")
    logger.info("=" * 72)
    _summary_block(labels, cot_states, "binary", "binary judge (correct/wrong)")
    if has_logprob:
        logger.info("")
        _summary_block(labels, cot_states, "delta", f"delta judge (Δmean-logprob, tau={tau:.3f})")
    else:
        logger.info("")
        logger.info("[delta judge] no gold_logprob_mean field in model_outputs — skipped")
    logger.info("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset JSONL")
    p.add_argument("--model_outputs", required=True, help="Model outputs JSONL")
    p.add_argument("--output", required=True, help="Output JSON")
    p.add_argument("--mode", default="natural", help="natural | symbolic")
    p.add_argument("--cot-states", nargs="+", default=COT_STATES)
    p.add_argument("--tau", type=float, default=0.1,
                   help="Δlogprob threshold (in nats per token) for delta judge")
    args = p.parse_args()

    derive_atomic_labels(
        dataset_path=args.dataset,
        model_outputs_path=args.model_outputs,
        output_path=args.output,
        cot_states=args.cot_states,
        mode=args.mode,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
