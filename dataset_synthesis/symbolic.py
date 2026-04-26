"""Symbolic counterpart generation — Stage 4 of the pipeline.

Programmatic Unicode symbol replacement for all entities in the underlying structure,
then text substitution across all variant question strings.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .configs.defaults import SYMBOL_POOL, SYMBOLIC_VARIANT_TYPES

logger = logging.getLogger(__name__)


def _extract_entities(structure: dict[str, Any]) -> list[str]:
    """Extract all entity labels from an underlying structure.

    Handles both graph-based (nodes) and rule-based (variables) structures.
    Returns entities in a stable order for consistent symbol assignment.
    """
    entities: list[str] = []
    seen: set[str] = set()

    # Graph-based: nodes
    for node in structure.get("nodes", []):
        label = node.get("label", "")
        if label and label not in seen:
            entities.append(label)
            seen.add(label)

    # Graph-based: edge relations
    for edge in structure.get("edges", []):
        rel = edge.get("relation", "")
        if rel and rel not in seen:
            entities.append(rel)
            seen.add(rel)

    # Rule-based: variables
    for var in structure.get("variables", []):
        name = var.get("name", "")
        if name and name not in seen:
            entities.append(name)
            seen.add(name)

    return entities


def build_entity_map(structure: dict[str, Any], symbol_pool: list[str] | None = None) -> dict[str, str]:
    """Build a mapping from entity labels to Unicode symbols."""
    pool = symbol_pool or SYMBOL_POOL
    entities = _extract_entities(structure)

    if len(entities) > len(pool):
        logger.warning(
            "More entities (%d) than symbols (%d), some entities will not be mapped",
            len(entities),
            len(pool),
        )

    return {entity: pool[i] for i, entity in enumerate(entities) if i < len(pool)}


def apply_symbol_substitution(text: str, entity_map: dict[str, str]) -> str:
    """Replace all entity occurrences in text with their symbols.

    Replaces longer entities first to avoid partial matches.
    """
    def _pattern_and_flags(entity: str) -> tuple[str, int]:
        # For alphanumeric-ish entities (variables, relations like capital_of), avoid
        # accidental substitutions inside larger words.
        if re.fullmatch(r"[A-Za-z0-9_]+", entity):
            return rf"\b{re.escape(entity)}\b", re.IGNORECASE
        # For multi-token / non-alnum entities (e.g., "Pride and Prejudice"), do a
        # literal match.
        return re.escape(entity), re.IGNORECASE

    sorted_entities = sorted(entity_map.keys(), key=len, reverse=True)
    result = text
    for entity in sorted_entities:
        symbol = entity_map[entity]
        pat, flags = _pattern_and_flags(entity)
        result = re.sub(pat, symbol, result, flags=flags)
    return result


def deep_apply_symbol_substitution(obj: Any, entity_map: dict[str, str]) -> Any:
    """Recursively apply symbol substitution to all string leaves.

    - Preserves dict keys (schema keys), only transforms values.
    - Handles nested dict/list structures.
    """
    if isinstance(obj, str):
        return apply_symbol_substitution(obj, entity_map)
    if isinstance(obj, list):
        return [deep_apply_symbol_substitution(x, entity_map) for x in obj]
    if isinstance(obj, dict):
        return {k: deep_apply_symbol_substitution(v, entity_map) for k, v in obj.items()}
    return obj


def derive_symbolic_control_family(
    family: dict[str, Any],
    symbol_pool: list[str] | None = None,
) -> dict[str, Any] | None:
    """Derive a standalone SymbolicControl family from a normal family.

    This makes the symbolic content the *primary* surface form:
    - task_family = "SymbolicControl"
    - base_question / normal_variants are symbolic
    - entity_map is stored in metadata and source_family_id is tracked
    """
    source_id = family.get("family_id", "")
    if not source_id:
        return None

    structure = family.get("underlying_structure", {})
    sym_raw = family.get("symbolic_variants")
    if isinstance(sym_raw, dict) and sym_raw.get("entity_map"):
        entity_map = sym_raw.get("entity_map", {})
        sym_variants = {k: v for k, v in sym_raw.items() if k not in ("entity_map", "source_family_id")}
    else:
        entity_map = build_entity_map(structure, symbol_pool)
        sym_variants = generate_symbolic_variants(family, symbol_pool)
        sym_variants = {k: v for k, v in sym_variants.items() if k not in ("entity_map", "source_family_id")}

    preamble = build_symbolic_preamble(entity_map)
    base_q = family.get("base_question", "")
    sym_base_q = f"{preamble}\n{apply_symbol_substitution(base_q, entity_map)}" if base_q else ""

    gold = family.get("gold_answer", "")
    sym_gold = entity_map.get(gold, gold)

    sym_support_facts = [apply_symbol_substitution(x, entity_map) for x in family.get("support_facts", [])]
    sym_reasoning = [apply_symbol_substitution(x, entity_map) for x in family.get("gold_reasoning_chain", [])]

    sym_structure = deep_apply_symbol_substitution(structure, entity_map)

    # Use symbolic variants as the normal variants of the derived family.
    derived_normal_variants: dict[str, Any] = {}
    for k, v in sym_variants.items():
        if isinstance(v, dict) and "question" in v:
            derived_normal_variants[k] = v

    derived_id = f"sym_{source_id}"
    return {
        "family_id": derived_id,
        "task_family": "SymbolicControl",
        "sub_family": family.get("sub_family", ""),
        "base_item_id": f"{derived_id}_base",
        "underlying_structure": sym_structure,
        "base_question": sym_base_q,
        "gold_answer": sym_gold,
        "gold_reasoning_chain": sym_reasoning,
        "support_facts": sym_support_facts,
        "required_steps": family.get("required_steps", 1),
        "normal_variants": derived_normal_variants,
        "symbolic_variants": None,
        "mcq_variants": family.get("mcq_variants", {}),
        "metadata": {
            "source_family_id": source_id,
            "entity_map": entity_map,
        },
    }


def build_symbolic_preamble(entity_map: dict[str, str]) -> str:
    """Build a preamble string that defines the symbol mappings.

    Example: "In this system, Germany = ∆, Berlin = ◇, capital_of = ⊕."
    """
    mappings = [f"{entity} = {symbol}" for entity, symbol in entity_map.items()]
    return "In this system, " + ", ".join(mappings) + "."


def generate_symbolic_variants(
    family: dict[str, Any],
    symbol_pool: list[str] | None = None,
) -> dict[str, Any]:
    """Generate symbolic counterparts for all variants of a family.

    This is purely programmatic — no API calls needed.
    Returns a dict with entity_map, source_family_id, and all symbolic variants.
    """
    structure = family.get("underlying_structure", {})
    entity_map = build_entity_map(structure, symbol_pool)
    preamble = build_symbolic_preamble(entity_map)

    family_id = family.get("family_id", "")
    normal_variants = family.get("normal_variants", {})

    symbolic = {
        "entity_map": entity_map,
        "source_family_id": family_id,
    }

    for vtype in SYMBOLIC_VARIANT_TYPES:
        variant_data = normal_variants.get(vtype)
        if not variant_data:
            continue

        question = variant_data.get("question", "") if isinstance(variant_data, dict) else ""
        if not question:
            continue

        sym_question = apply_symbol_substitution(question, entity_map)
        sym_question = f"{preamble}\n{sym_question}"

        orig_meta = variant_data.get("metadata", {}) if isinstance(variant_data, dict) else {}
        sym_metadata = deep_apply_symbol_substitution(orig_meta, entity_map)

        symbolic[vtype] = {
            "question": sym_question,
            "metadata": sym_metadata,
        }

    return symbolic


def generate_all_symbolic(
    families: list[dict[str, Any]],
    symbol_pool: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate symbolic counterparts for all families. Returns enriched family dicts."""
    enriched = []
    for family in families:
        sym = generate_symbolic_variants(family, symbol_pool)
        family_copy = {**family, "symbolic_variants": sym}
        enriched.append(family_copy)
    return enriched
