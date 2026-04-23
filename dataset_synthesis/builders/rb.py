"""RB block: structure templates and prompts for rule-based items."""

from __future__ import annotations

from typing import Any

from ..api_client import APIClient

RB_STRUCTURE_SYSTEM = """You are a dataset designer for mechanistic interpretability research.
Generate underlying structures for rule-based (RB) item families.

RB items test explicit rule/structure derivation — answers come from applying given rules,
not from world knowledge. Rules must be explicitly stated. Gold answer must be programmatically verifiable.

Return a JSON array of structure objects. Each object must have:
- sub_family: one of "linear_equation", "syllogistic_logic", "sequence_pattern", "boolean_logic", "function_application"
- type: structure type (e.g. "algebraic_equation", "syllogism", "sequence_rule", "boolean_expression", "function_composition")
- variables: array of {name, role} where role is "unknown", "constant", "parameter"
- rules: array of rule strings (equations, logical statements, sequence rules)
- derivation_steps: array of step-by-step derivation strings
- gold_derivation: derivation type (e.g. "algebraic_manipulation", "transitive_inference", "pattern_extrapolation", "truth_table", "function_evaluation")
- gold_answer: the correct answer
- support_facts: array of the rules/premises needed
- required_steps: number of derivation steps

Distribute items across sub-families roughly evenly.
Make items diverse in difficulty and structure. Avoid trivially simple items.
Return ONLY valid JSON, no markdown or explanation."""

RB_BASE_ITEM_SYSTEM = """You are a dataset designer. Given an underlying structure for an RB item,
generate a natural-language base question and verify the gold answer.

The question must present the rules explicitly so the answer is derivable without world knowledge.

Return a JSON object with:
- base_question: a clear question that states the rules and asks for the answer
- gold_answer: the correct answer (must match the structure)
- gold_reasoning_chain: array of reasoning steps in natural language
- support_facts: array of the rules/premises

Return ONLY valid JSON."""


def build_rb_structure_prompt(count: int) -> str:
    return f"Generate {count} diverse RB item structures. Distribute across sub-families: linear_equation, syllogistic_logic, sequence_pattern, boolean_logic, function_application."


def build_rb_base_item_prompt(structure: dict[str, Any]) -> str:
    import json
    return f"Generate a base question for this RB structure:\n{json.dumps(structure, ensure_ascii=False)}"


def generate_rb_structures(client: APIClient, count: int) -> list[dict[str, Any]]:
    prompt = build_rb_structure_prompt(count)
    result = client.call_api_json(RB_STRUCTURE_SYSTEM, prompt)
    if not isinstance(result, list):
        result = [result]
    return result


def generate_rb_base_item(client: APIClient, structure: dict[str, Any]) -> dict[str, Any]:
    prompt = build_rb_base_item_prompt(structure)
    return client.call_api_json(RB_BASE_ITEM_SYSTEM, prompt)
