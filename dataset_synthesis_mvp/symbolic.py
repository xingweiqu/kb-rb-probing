"""Symbolic variant generation (Stage 4)."""

import re
from typing import Any

from .config import SYMBOL_POOL


def _extract_entities(structure: dict[str, Any], family: dict[str, Any]) -> list[str]:
    """Extract all entities from structure and family."""
    entities = []
    seen = set()

    # From nodes
    for node in structure.get("nodes", []):
        label = node.get("label", "")
        if label and label not in seen:
            entities.append(label)
            seen.add(label)

    # From edges
    for edge in structure.get("edges", []):
        rel = edge.get("relation", "")
        if rel and rel not in seen:
            entities.append(rel)
            seen.add(rel)

    # From variables (RB)
    for var in structure.get("variables", []):
        name = var.get("name", "")
        if name and name not in seen:
            entities.append(name)
            seen.add(name)

    # CRITICAL FIX: Add gold_answer
    gold = family.get("gold_answer", "")
    if gold and gold not in seen:
        entities.append(gold)
        seen.add(gold)

    return entities


def build_entity_map(structure: dict[str, Any], family: dict[str, Any]) -> dict[str, str]:
    """Build entity → symbol mapping."""
    entities = _extract_entities(structure, family)
    return {entity: SYMBOL_POOL[i % len(SYMBOL_POOL)] for i, entity in enumerate(entities)}


def apply_symbol_substitution(text: str, entity_map: dict[str, str]) -> str:
    """Replace entities with symbols."""
    sorted_entities = sorted(entity_map.keys(), key=len, reverse=True)
    result = text

    for entity in sorted_entities:
        symbol = entity_map[entity]
        # Token-aware replacement
        if re.fullmatch(r"[A-Za-z0-9_\-]+", entity):
            pattern = rf"\b{re.escape(entity)}\b"
            result = re.sub(pattern, symbol, result, flags=re.IGNORECASE)
        else:
            result = result.replace(entity, symbol)

    return result


def build_symbolic_preamble(entity_map: dict[str, str]) -> str:
    """Build preamble defining symbol mappings."""
    mappings = [f"{entity} = {symbol}" for entity, symbol in entity_map.items()]
    return "In this system, " + ", ".join(mappings) + "."


def generate_symbolic_variant(item: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    """Generate symbolic version of an item."""
    structure = family.get("underlying_structure", {})
    entity_map = build_entity_map(structure, family)
    preamble = build_symbolic_preamble(entity_map)

    question = item["question"]
    symbolic_question = apply_symbol_substitution(question, entity_map)
    symbolic_question = f"{preamble}\n{symbolic_question}"

    return {
        **item,
        "mode": "symbolic",
        "question": symbolic_question,
        "metadata": {
            **item.get("metadata", {}),
            "entity_map": entity_map,
            "symbolic_mode": True
        }
    }


def generate_all_symbolic(items: list[dict[str, Any]], families: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate symbolic variants for all natural items."""
    symbolic_items = []

    for item in items:
        if item.get("mode") == "natural":
            family_id = item["family_id"]
            family = families.get(family_id)
            if family:
                symbolic = generate_symbolic_variant(item, family)
                symbolic_items.append(symbolic)

    return symbolic_items
