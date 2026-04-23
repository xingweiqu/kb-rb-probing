"""Variant generation — Stage 3 of the pipeline.

Generates all 19 variant types for each base item via a single API call per family.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .api_client import APIClient
from .configs.defaults import VARIANT_TYPES

logger = logging.getLogger(__name__)

VARIANT_DEFINITIONS = {
    "original": "Standard question, no extra help or interference. Identical to base_question.",
    "hint": (
        "Give directional help that does NOT complete the support chain. "
        "Cannot reveal the answer or key premise directly. "
        "Example: 'This country is in Western Europe...' for a capital question."
    ),
    "premise": (
        "Explicitly inject the key support fact/rule before the question. "
        "Must directly change evidence availability. "
        "Example: 'Germany\\'s capital is Berlin. What is the capital of Germany?'"
    ),
    "premise_removal": (
        "Remove the key support, making the question significantly harder or unsolvable. "
        "Must NOT remain trivially solvable after removal. "
        "For KB: make the question vague. For RB: remove the equation/rule."
    ),
    "highlight": (
        "Extract or highlight the key support location without adding new facts. "
        "Like a focused spotlight on existing evidence. "
        "Example: 'Focus on the relationship between Germany and its administrative center: ...'"
    ),
    "full_support_bundle": (
        "Provide the COMPLETE support chain — all facts/rules needed. "
        "For multi-hop items, include all intermediate facts. "
        "For single-hop, this is similar to premise but more explicit."
    ),
    "scaffold_1": (
        "Give ONE intermediate decomposition step. Must NOT give the final answer. "
        "Example: 'First, identify which country has Berlin as a city. Then, ...'"
    ),
    "scaffold_2": (
        "Give TWO intermediate decomposition steps. Must NOT give the final answer. "
        "More guidance than scaffold_1 but still requires the model to reach the conclusion."
    ),
    "scaffold_3": (
        "Give THREE or more intermediate steps. Must NOT give the final answer. "
        "Nearly complete walkthrough but the final step is left to the model."
    ),
    "scaffold_shuffled": (
        "Same content as scaffold_2 or scaffold_3, but steps in RANDOMIZED order. "
        "The information is all there but the logical sequence is disrupted. "
        "Record both original_order and shuffled_order in metadata."
    ),
    "cot_full": (
        "Question followed by the COMPLETE gold reasoning chain as explicit step-by-step reasoning, "
        "then ask for the final answer. The chain should be presented as: "
        "'Let\\'s think step by step. Step 1: ... Step 2: ... Therefore, the answer is?'"
    ),
    "cot_partial": (
        "Question followed by a PARTIAL reasoning chain — give the first N-1 steps of an N-step chain. "
        "The model must complete the final reasoning step. "
        "Record which steps are included and which are omitted in metadata."
    ),
    "cot_shuffled": (
        "Question followed by the complete reasoning chain but with steps in RANDOMIZED order. "
        "All information is present but the logical flow is disrupted. "
        "Record both original_order and shuffled_order in metadata."
    ),
    "wrongclaim_bare": (
        "Insert one incorrect claim or wrong answer cue into the question. "
        "Cannot change the gold answer. Cannot make the question multi-answer. "
        "Example: 'Some sources say the capital of Germany is Munich. What is the capital of Germany?'"
    ),
    "wrongclaim_confident": (
        "Same wrong claim as wrongclaim_bare, but wrapped with high-confidence language. "
        "Use phrases like: 'Obviously...', 'It is well-established that...', 'Clearly...', "
        "'Everyone knows that...'. Record the confidence_wrapper in metadata."
    ),
    "wrongclaim_attributed": (
        "Same wrong claim as wrongclaim_bare, but attributed to an authority source. "
        "Use phrases like: 'According to experts...', 'Research has shown...', "
        "'The official documentation states...', 'A recent study found...'. "
        "Record the attribution in metadata."
    ),
    "competing_claims": (
        "Present BOTH the correct claim AND an incorrect claim in the same question. "
        "Tests conflict resolution. "
        "Example: 'Some say Berlin is the capital of Germany, others say it is Munich. "
        "What is actually the capital of Germany?' Record both claims in metadata."
    ),
    "paraphrase": (
        "Change expression only, NOT semantics. The meaning must be identical. "
        "Use different vocabulary, sentence structure, or phrasing."
    ),
    "terminology_swap": (
        "Replace common/everyday terms with domain-specific or technical terminology. "
        "Semantics must remain unchanged. "
        "Example: 'capital city' → 'seat of government', 'solve for x' → 'determine the unknown variable'. "
        "Record the swap_map in metadata."
    ),
    "substitution": (
        "Replace entities/content words/symbolic names while preserving the underlying relational structure. "
        "Example: replace 'Germany/Berlin' with 'Japan/Tokyo' — same relation, different entities. "
        "Record the substitution_map in metadata."
    ),
}

VARIANT_GENERATION_SYSTEM = """You are a dataset designer for mechanistic interpretability research.
Given a base item (question, answer, underlying structure), generate all 19 controlled variants.

CRITICAL RULES:
1. The gold_answer must be preserved across ALL variants EXCEPT premise_removal (which may become unsolvable).
2. Each variant must follow its definition EXACTLY.
3. Return structured JSON with all variant keys.
4. For metadata fields, include the specific content described in each variant's definition.

Variant definitions:
{definitions}

Return a JSON object where each key is a variant name and each value is:
{{
    "question": "the variant question text",
    "metadata": {{...variant-specific metadata...}}
}}

Return ONLY valid JSON, no markdown or explanation."""

VARIANT_GENERATION_USER = """Base item:
- family_id: {family_id}
- task_family: {task_family}
- sub_family: {sub_family}
- base_question: {base_question}
- gold_answer: {gold_answer}
- gold_reasoning_chain: {gold_reasoning_chain}
- support_facts: {support_facts}
- underlying_structure: {underlying_structure}

Generate all 19 variants for this item."""


def _build_definitions_block() -> str:
    lines = []
    for vtype in VARIANT_TYPES:
        defn = VARIANT_DEFINITIONS.get(vtype, "")
        lines.append(f"- {vtype}: {defn}")
    return "\n".join(lines)


def build_variant_prompt(family: dict[str, Any]) -> tuple[str, str]:
    system = VARIANT_GENERATION_SYSTEM.format(definitions=_build_definitions_block())
    user = VARIANT_GENERATION_USER.format(
        family_id=family.get("family_id", ""),
        task_family=family.get("task_family", ""),
        sub_family=family.get("sub_family", ""),
        base_question=family.get("base_question", ""),
        gold_answer=family.get("gold_answer", ""),
        gold_reasoning_chain=json.dumps(family.get("gold_reasoning_chain", []), ensure_ascii=False),
        support_facts=json.dumps(family.get("support_facts", []), ensure_ascii=False),
        underlying_structure=json.dumps(family.get("underlying_structure", {}), ensure_ascii=False),
    )
    return system, user


def generate_variants(client: APIClient, family: dict[str, Any]) -> dict[str, Any]:
    """Generate all 19 variants for a single family.

    Returns a dict mapping variant_name -> {question, metadata}.
    """
    system, user = build_variant_prompt(family)
    logger.info("Generating variants for %s...", family.get("family_id"))
    result = client.call_api_json(system, user)

    if not isinstance(result, dict):
        raise ValueError(f"Expected dict from variant generation, got {type(result)}")

    # Ensure original variant exists
    if "original" not in result:
        result["original"] = {
            "question": family.get("base_question", ""),
            "metadata": {},
        }

    return result


def generate_all_variants(
    client: APIClient,
    families: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate variants for all families. Returns enriched family dicts."""
    enriched = []
    for family in families:
        variants = generate_variants(client, family)
        family_copy = {**family, "normal_variants": variants}
        enriched.append(family_copy)
    return enriched
