"""MCQ generation — Stage 5 of the pipeline.

Generates 4-choice MCQ items, prioritizing symbolic families.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .api_client import APIClient

logger = logging.getLogger(__name__)

MCQ_SYSTEM = """You are a dataset designer. Generate MCQ (multiple choice question) distractors
for a symbolic item family.

Given:
- The symbolic question and gold answer
- The entity map (real entities → symbols)
- The underlying structure

Generate 3 distractors for a 4-choice MCQ. The distractors must be:
1. same_type: A symbol/entity of the same type as the gold answer (e.g., another capital if gold is a capital)
2. structurally_related: A symbol/entity that appears in the structure but is NOT the answer (e.g., the query entity)
3. wrongclaim_aligned: A plausible wrong answer that aligns with common misconceptions or the wrong claim from the wrongclaim_bare variant (if available)

Return a JSON object with:
- options: array of 4 strings [gold, same_type, structurally_related, wrongclaim_aligned]
- correct_index: 0 (gold is always first, we shuffle later)
- option_metadata: array of 4 objects, each with "role" and "source" fields

Return ONLY valid JSON."""

MCQ_USER = """Symbolic item:
- question: {question}
- gold_answer: {gold_answer}
- entity_map: {entity_map}
- underlying_structure: {structure}
- wrong_claim (if available): {wrong_claim}

Generate 3 distractors for a 4-choice MCQ."""


def build_mcq_prompt(family: dict[str, Any]) -> tuple[str, str]:
    sym = family.get("symbolic_variants", {})
    entity_map = sym.get("entity_map", {})

    sym_original = sym.get("original", {})
    question = sym_original.get("question", "") if isinstance(sym_original, dict) else ""

    gold = family.get("gold_answer", "")
    # Map gold answer to symbol if possible
    sym_gold = entity_map.get(gold, gold)

    wrong_claim = ""
    wc = sym.get("wrongclaim_bare", {})
    if isinstance(wc, dict):
        wc_meta = wc.get("metadata", {})
        wrong_claim = wc_meta.get("wrong_claim", "")

    user = MCQ_USER.format(
        question=question,
        gold_answer=sym_gold,
        entity_map=json.dumps(entity_map, ensure_ascii=False),
        structure=json.dumps(family.get("underlying_structure", {}), ensure_ascii=False),
        wrong_claim=wrong_claim,
    )
    return MCQ_SYSTEM, user


def generate_mcq(client: APIClient, family: dict[str, Any]) -> dict[str, Any]:
    """Generate MCQ for a single family's symbolic original variant.

    Returns a dict with the MCQ item data.
    """
    system, user = build_mcq_prompt(family)
    logger.info("Generating MCQ for %s...", family.get("family_id"))
    result = client.call_api_json(system, user)

    sym = family.get("symbolic_variants", {})
    sym_original = sym.get("original", {})
    question = sym_original.get("question", "") if isinstance(sym_original, dict) else ""

    return {
        "symbolic_original": {
            "question": question,
            "options": result.get("options", []),
            "correct_index": result.get("correct_index", 0),
            "option_metadata": result.get("option_metadata", []),
        }
    }


def generate_all_mcq(
    client: APIClient,
    families: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate MCQ for all families that have symbolic variants.

    Returns enriched family dicts.
    """
    enriched = []
    for family in families:
        sym = family.get("symbolic_variants", {})
        if sym and sym.get("original"):
            mcq = generate_mcq(client, family)
            family_copy = {**family, "mcq_variants": mcq}
        else:
            family_copy = {**family, "mcq_variants": {}}
        enriched.append(family_copy)
    return enriched
