"""Hybrid block: structure templates and prompts for hybrid items."""

from __future__ import annotations

from typing import Any

from ..api_client import APIClient

HYBRID_STRUCTURE_SYSTEM = """You are a dataset designer for mechanistic interpretability research.
Generate underlying structures for Hybrid item families.

Hybrid items require BOTH knowledge access AND composition/reasoning.
- Remove the retrieval component → unsolvable
- Remove the reasoning/composition component → not directly solvable

Return a JSON array of structure objects. Each object must have:
- sub_family: one of "two_hop_relational", "retrieve_transform", "retrieve_rule_apply"
- type: structure type (e.g. "two_hop_composition", "retrieve_and_compute", "retrieve_and_apply_rule")
- nodes: array of {id, label, role} where role is "query_entity", "intermediate", or "answer"
- edges: array of {source, target, relation}
- rules: array of any transformation/computation rules (can be empty for pure multi-hop)
- support_chain: array of string representations of the reasoning path
- gold_derivation: derivation type (e.g. "multi_hop_lookup", "retrieve_then_compute", "retrieve_then_apply")
- gold_answer: the correct answer
- support_facts: array of all supporting facts needed
- required_steps: number of steps (typically 2-3)

Distribute items across sub-families roughly evenly.
Ensure each item genuinely requires both knowledge and reasoning — not just multi-step knowledge lookup.
Return ONLY valid JSON, no markdown or explanation."""

HYBRID_BASE_ITEM_SYSTEM = """You are a dataset designer. Given an underlying structure for a Hybrid item,
generate a natural-language base question and verify the gold answer.

The question should naturally require both knowledge retrieval and reasoning/composition.
Do NOT state all the intermediate facts — the question should require the model to retrieve some knowledge.

Return a JSON object with:
- base_question: a natural question requiring both knowledge and reasoning
- gold_answer: the correct answer (must match the structure)
- gold_reasoning_chain: array of reasoning steps showing both retrieval and composition
- support_facts: array of all facts needed

Return ONLY valid JSON."""


def build_hybrid_structure_prompt(count: int) -> str:
    return (
        f"Generate EXACTLY {count} diverse Hybrid item structures. "
        "Distribute across sub-families: two_hop_relational, retrieve_transform, retrieve_rule_apply."
    )


def build_hybrid_base_item_prompt(structure: dict[str, Any]) -> str:
    import json
    return f"Generate a base question for this Hybrid structure:\n{json.dumps(structure, ensure_ascii=False)}"


def generate_hybrid_structures(client: APIClient, count: int) -> list[dict[str, Any]]:
    prompt = build_hybrid_structure_prompt(count)
    result = client.call_api_json(HYBRID_STRUCTURE_SYSTEM, prompt)
    if not isinstance(result, list):
        result = [result]
    return result


async def generate_hybrid_structures_async(client: APIClient, count: int) -> list[dict[str, Any]]:
    prompt = build_hybrid_structure_prompt(count)
    # Expect a JSON array, so do NOT force json_object response_format.
    result = await client.call_api_json_async(HYBRID_STRUCTURE_SYSTEM, prompt, response_format=None)
    if not isinstance(result, list):
        result = [result]
    return result


def generate_hybrid_base_item(client: APIClient, structure: dict[str, Any]) -> dict[str, Any]:
    prompt = build_hybrid_base_item_prompt(structure)
    return client.call_api_json(HYBRID_BASE_ITEM_SYSTEM, prompt)


async def generate_hybrid_base_item_async(client: APIClient, structure: dict[str, Any]) -> dict[str, Any]:
    prompt = build_hybrid_base_item_prompt(structure)
    return await client.call_api_json_async(
        HYBRID_BASE_ITEM_SYSTEM,
        prompt,
        response_format={"type": "json_object"},
    )
