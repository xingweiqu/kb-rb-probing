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
    sorted_entities = sorted(entity_map.keys(), key=len, reverse=True)
    result = text
    for entity in sorted_entities:
        symbol = entity_map[entity]
        result = re.sub(re.escape(entity), symbol, result, flags=re.IGNORECASE)
    return result


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

        sym_metadata = {}
        orig_meta = variant_data.get("metadata", ) if isinstance(variant_data, dict) else {}
        for k, v in orig_meta.items():
            if isinstance(v, str):
                sym_metadata[k] = apply_symbol_substitution(v, entity_map)
            elif isinstance(v, list):
                sym_metadata[k] = [
                    apply_symbol_substitution(item, entity_map) if isinstance(item, str) else item
                    for item in v
                ]
            elif isinstance(v, dict):
                sym_metadata[k] = {
                    apply_symbol_substitution(mk, entity_map) if isinstance(mk, str) else mk:
                    apply_symbol_substitution(mv, entity_map) if isinstance(mv, str) else mv
                    for mk, mv in v.items()
                }
            else:
                sym_metadata[k] = v

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
