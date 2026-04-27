"""Derive atomic capability labels from model behavior."""

import json
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def derive_atomic_labels(
    dataset_path: str,
    model_outputs_path: str,
    output_path: str
) -> dict[str, dict[str, bool | None]]:
    """
    Derive atomic capability labels from model behavior.

    Label derivation rules:
    - wrong_claim_susceptible: original correct + wrongclaim wrong
    - retrieval_disambiguation: original wrong + hint correct
    - retrieval_robustness: original correct + paraphrase wrong
    - decomposition_sensitive: original wrong + scaffold correct
    - insufficient_info_awareness: original correct + removal wrong
    - intermediate_state_robustness: original correct + wrong_intermediate wrong
    - retrieval_operation_separable: original wrong + explicit_fact correct
    - reasoning_compensates_retrieval: original wrong + retrieval_blocked correct

    Returns:
        labels: dict[family_id] -> dict[atomic_capability] -> bool | None
    """
    # Load dataset
    items_by_key = {}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['family_id']}_{item['variant']}_{item.get('mode', 'natural')}"
            items_by_key[key] = item

    # Load model outputs
    outputs_by_key = {}
    with open(model_outputs_path, encoding="utf-8") as f:
        for line in f:
            output = json.loads(line)
            key = f"{output['family_id']}_{output['variant']}_{output.get('mode', 'natural')}"
            outputs_by_key[key] = output

    # Get unique families
    families = set(item["family_id"] for item in items_by_key.values())

    labels = {}

    for family_id in families:
        family_labels = {}

        # Get original behavior
        orig_key = f"{family_id}_original_natural"
        orig = outputs_by_key.get(orig_key)
        if not orig:
            continue

        orig_correct = orig.get("correct", False)

        # KB labels
        wc_key = f"{family_id}_wrongclaim_natural"
        if wc_key in outputs_by_key:
            wc = outputs_by_key[wc_key]
            if orig_correct and not wc.get("correct"):
                family_labels["wrong_claim_susceptible"] = True
            elif orig_correct and wc.get("correct"):
                family_labels["wrong_claim_susceptible"] = False
            else:
                family_labels["wrong_claim_susceptible"] = None

        hint_key = f"{family_id}_hint_natural"
        if hint_key in outputs_by_key:
            hint = outputs_by_key[hint_key]
            if not orig_correct and hint.get("correct"):
                family_labels["retrieval_disambiguation"] = True
            elif orig_correct and hint.get("correct"):
                family_labels["retrieval_disambiguation"] = False
            else:
                family_labels["retrieval_disambiguation"] = None

        para_key = f"{family_id}_paraphrase_natural"
        if para_key in outputs_by_key:
            para = outputs_by_key[para_key]
            if orig_correct and not para.get("correct"):
                family_labels["retrieval_robustness"] = True
            elif orig_correct and para.get("correct"):
                family_labels["retrieval_robustness"] = False
            else:
                family_labels["retrieval_robustness"] = None

        # RB labels
        sc_key = f"{family_id}_scaffold_natural"
        if sc_key in outputs_by_key:
            sc = outputs_by_key[sc_key]
            if not orig_correct and sc.get("correct"):
                family_labels["decomposition_sensitive"] = True
            elif orig_correct and sc.get("correct"):
                family_labels["decomposition_sensitive"] = False
            else:
                family_labels["decomposition_sensitive"] = None

        rm_key = f"{family_id}_rule_removal_natural"
        if rm_key in outputs_by_key:
            rm = outputs_by_key[rm_key]
            if orig_correct and not rm.get("correct"):
                family_labels["insufficient_info_awareness"] = True
            elif orig_correct and rm.get("correct"):
                family_labels["insufficient_info_awareness"] = False
            else:
                family_labels["insufficient_info_awareness"] = None

        wi_key = f"{family_id}_wrong_intermediate_natural"
        if wi_key in outputs_by_key:
            wi = outputs_by_key[wi_key]
            if orig_correct and not wi.get("correct"):
                family_labels["intermediate_state_robustness"] = True
            elif orig_correct and wi.get("correct"):
                family_labels["intermediate_state_robustness"] = False
            else:
                family_labels["intermediate_state_robustness"] = None

        # Hybrid labels
        ef_key = f"{family_id}_explicit_fact_natural"
        if ef_key in outputs_by_key:
            ef = outputs_by_key[ef_key]
            if not orig_correct and ef.get("correct"):
                family_labels["retrieval_operation_separable"] = True
            elif orig_correct and ef.get("correct"):
                family_labels["retrieval_operation_separable"] = False
            else:
                family_labels["retrieval_operation_separable"] = None

        rb_key = f"{family_id}_retrieval_blocked_natural"
        if rb_key in outputs_by_key:
            rb = outputs_by_key[rb_key]
            if not orig_correct and rb.get("correct"):
                family_labels["reasoning_compensates_retrieval"] = True
            elif orig_correct and rb.get("correct"):
                family_labels["reasoning_compensates_retrieval"] = False
            else:
                family_labels["reasoning_compensates_retrieval"] = None

        labels[family_id] = family_labels

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("LABEL SUMMARY")
    logger.info(f"{'='*60}")

    for capability in [
        "wrong_claim_susceptible",
        "retrieval_disambiguation",
        "retrieval_robustness",
        "decomposition_sensitive",
        "insufficient_info_awareness",
        "intermediate_state_robustness",
        "retrieval_operation_separable",
        "reasoning_compensates_retrieval"
    ]:
        values = [l.get(capability) for l in labels.values() if capability in l]
        if values:
            pos = values.count(True)
            neg = values.count(False)
            dropped = values.count(None)
            logger.info(f"{capability:40s}: pos={pos:3d}, neg={neg:3d}, dropped={dropped:3d}")

    logger.info(f"{'='*60}\n")

    return labels


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset JSONL")
    parser.add_argument("--model_outputs", required=True, help="Model outputs JSONL")
    parser.add_argument("--output", required=True, help="Output JSON")

    args = parser.parse_args()

    derive_atomic_labels(
        dataset_path=args.dataset,
        model_outputs_path=args.model_outputs,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
