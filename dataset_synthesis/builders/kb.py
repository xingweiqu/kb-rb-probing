"""KB block: structure templates and prompts for knowledge-based items."""

from __future__ import annotations

from typing import Any

from ..api_client import APIClient

KB_STRUCTURE_SYSTEM = """You are a dataset designer for mechanistic interpretability research.
Generate underlying structures for knowledge-based (KB) item families.

KB items test factual/associative binding — answers come from stored knowledge, not reasoning.
Support chains should be short (1 hop). Gold answer must be unique and unambiguous.

Return a JSON array of structure objects. Each object must have:
- sub_family: one of "country_capital", "work_author", "element_symbol", "entity_attribute"
- type: structure type (e.g. "single_hop_binding")
- nodes: array of {id, label, role} where role is "query_entity" or "answer"
- edges: array of {source, target, relation}
- support_chain: array of string representations like "n1 --relation--> n2"
- gold_derivation: "direct_lookup"
- gold_answer: the correct answer
- support_facts: array of supporting fact strings
- required_steps: 1

Distribute items across sub-families roughly evenly.
Make items diverse — avoid repetitive patterns within the same sub-family.
Return ONLY valid JSON, no markdown or explanation."""

KB_BASE_ITEM_SYSTEM = """You are a dataset designer. Given an underlying structure for a KB item,
generate a natural-language base question and verify the gold answer.

Return a JSON object with:
- base_question: a clear, natural question whose answer is the gold_answer
- gold_answer: the correct answer (must match the structure)
- gold_reasoning_chain: array of 1-2 reasoning steps in natural language
- support_facts: array of supporting facts

Return ONLY valid JSON."""


def build_kb_structure_prompt(count: int) -> str:
    return (
        f"Generate EXACTLY {count} diverse KB item structures. "
        "Distribute across sub-families: country_capital, work_author, element_symbol, entity_attribute."
    )


def build_kb_base_item_prompt(structure: dict[str, Any]) -> str:
    import json
    return f"Generate a base question for this KB structure:\n{json.dumps(structure, ensure_ascii=False)}"


def generate_kb_structures(client: APIClient, count: int) -> list[dict[str, Any]]:
    prompt = build_kb_structure_prompt(count)
    result = client.call_api_json(KB_STRUCTURE_SYSTEM, prompt)
    if not isinstance(result, list):
        result = [result]
    return result


async def generate_kb_structures_async(client: APIClient, count: int) -> list[dict[str, Any]]:
    prompt = build_kb_structure_prompt(count)
    # Expect a JSON array, so do NOT force json_object response_format.
    result = await client.call_api_json_async(KB_STRUCTURE_SYSTEM, prompt, response_format=None)
    if not isinstance(result, list):
        result = [result]
    return result


def generate_kb_base_item(client: APIClient, structure: dict[str, Any]) -> dict[str, Any]:
    prompt = build_kb_base_item_prompt(structure)
    return client.call_api_json(KB_BASE_ITEM_SYSTEM, prompt)


async def generate_kb_base_item_async(client: APIClient, structure: dict[str, Any]) -> dict[str, Any]:
    prompt = build_kb_base_item_prompt(structure)
    return await client.call_api_json_async(
        KB_BASE_ITEM_SYSTEM,
        prompt,
        response_format={"type": "json_object"},
    )
