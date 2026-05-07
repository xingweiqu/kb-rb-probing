"""Derive 2x3 atomic capability labels + CoT state from model behavior.

Taxonomy:
    Mechanism axis:  Knowledge / Reasoning / Composition
    Perturbation:    Provide / Block / Distract
    CoT axis:        with_cot / no_cot

Capability labels (mechanism letter + perturbation letter):
    KP  knowledge surfacing            (Knowledge × Provide)
    KB  retrieval dependence           (Knowledge × Block)
    KD  retrieval robustness           (Knowledge × Distract)
    RP  rule activation                (Reasoning × Provide)
    RB  rule dependence                (Reasoning × Block)
    RD  reasoning robustness           (Reasoning × Distract)
    CP  gap-filling                    (Composition × Provide)
    CB  bottleneck identification      (Composition × Block)
    CD  cross-axis interference        (Composition × Distract; not yet operationalized)

Variant -> primary capability mapping:
    KB-hint                  -> KP
    KB-wrongclaim            -> KD
    KB-paraphrase            -> KB
    RB-scaffold              -> RP
    RB-rule_removal          -> RB
    RB-wrong_intermediate    -> RD
    Hybrid-explicit_fact     -> CP
    Hybrid-retrieval_blocked -> CB
    Hybrid-both_blocked      -> CB-control  (lower-bound control, double-block)

Judgment rules (per-family, per-CoT-state):
    Provide-axis  (KP, RP, CP):  ¬orig_correct ∧ variant_correct  -> True
                                  orig_correct  ∧ variant_correct  -> False
                                  otherwise                        -> None
    Block-axis    (KB, RB, CB):  orig_correct  ∧ ¬variant_correct -> True
                                  orig_correct  ∧ variant_correct  -> False
                                  otherwise                        -> None
    Distract-axis (KD, RD):      orig_correct  ∧ ¬variant_correct -> True
                                  orig_correct  ∧ variant_correct  -> False
                                  otherwise                        -> None
"""

import json
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
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


def _judge(orig_correct: bool, variant_correct: bool, axis: str) -> bool | None:
    """Apply axis-specific judgment.

    Provide axis: capability is True if injection rescues a wrong-by-default item.
    Block/Distract axes: capability is True if perturbation breaks an originally-correct item.
    """
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


def _output_key(family_id: str, variant: str, mode: str, cot_state: str) -> str:
    return f"{family_id}_{variant}_{mode}_{cot_state}"


def derive_atomic_labels(
    dataset_path: str,
    model_outputs_path: str,
    output_path: str,
    cot_states: list[str] | None = None,
    mode: str = "natural",
) -> dict[str, dict[str, dict[str, bool | None]]]:
    """Derive capability labels per family, per CoT state.

    Returns nested dict:
        labels[family_id][cot_state][capability] -> True / False / None
    """
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
    labels: dict[str, dict[str, dict[str, bool | None]]] = {}

    for family_id in families:
        labels[family_id] = {}
        for cot in cot_states:
            cell: dict[str, bool | None] = {}
            orig = outputs_by_key.get(_output_key(family_id, "original", mode, cot))
            if not orig:
                labels[family_id][cot] = cell
                continue
            orig_correct = bool(orig.get("correct", False))

            for variant, (capability, axis) in VARIANT_TO_CAPABILITY.items():
                v = outputs_by_key.get(_output_key(family_id, variant, mode, cot))
                if not v:
                    continue
                cell[capability] = _judge(orig_correct, bool(v.get("correct", False)), axis)
            labels[family_id][cot] = cell

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    _log_summary(labels, cot_states)
    return labels


def _log_summary(
    labels: dict[str, dict[str, dict[str, bool | None]]],
    cot_states: list[str],
) -> None:
    logger.info("=" * 72)
    logger.info("LABEL SUMMARY (capability × cot_state)")
    logger.info("=" * 72)
    header = f"{'capability':14s} | " + " | ".join(f"{c:>22s}" for c in cot_states)
    logger.info(header)
    logger.info("-" * len(header))
    for cap in CAPABILITIES:
        cells = []
        for cot in cot_states:
            vals = [
                labels[fam][cot].get(cap)
                for fam in labels
                if cot in labels[fam] and cap in labels[fam][cot]
            ]
            pos = vals.count(True)
            neg = vals.count(False)
            drop = vals.count(None)
            cells.append(f"pos={pos:2d} neg={neg:2d} drop={drop:2d}")
        logger.info(f"{cap:14s} | " + " | ".join(f"{c:>22s}" for c in cells))
    logger.info("=" * 72)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset JSONL")
    parser.add_argument("--model_outputs", required=True, help="Model outputs JSONL (must contain cot_state field)")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--mode", default="natural", help="natural | symbolic")
    parser.add_argument("--cot-states", nargs="+", default=COT_STATES,
                        help="Which CoT states to derive labels for")
    args = parser.parse_args()

    derive_atomic_labels(
        dataset_path=args.dataset,
        model_outputs_path=args.model_outputs,
        output_path=args.output,
        cot_states=args.cot_states,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
