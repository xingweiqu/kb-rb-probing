"""Structure generation (Stage 1) - simplified."""

import asyncio
import logging
from typing import Any

from .api_client import APIClient
from .config import FAMILY_COUNTS

logger = logging.getLogger(__name__)


_STRUCTURE_SYSTEM = (
    "You are a dataset designer that outputs ONLY a JSON array. "
    "Do not output any explanation, no markdown fences, no preamble. "
    "Your entire response must be a valid JSON array of objects."
)


KB_STRUCTURE_USER = """Generate {count} diverse KB (knowledge-based) structures using real-world facts \
(geography, science, history, culture). Each structure represents a single-hop fact \
entity1 --relation--> entity2.

Each object MUST have these fields: nodes, edges, gold_answer, support_facts.

Example object (illustrative, do not copy):
{
  "nodes": [
    {"id": "n1", "label": "France", "role": "query_entity"},
    {"id": "n2", "label": "Paris", "role": "answer"}
  ],
  "edges": [
    {"source": "n1", "target": "n2", "relation": "capital_of"}
  ],
  "gold_answer": "Paris",
  "support_facts": ["Paris is the capital of France"]
}

Output ONLY a JSON array of {count} such objects. No markdown, no commentary."""


RB_STRUCTURE_USER = """Generate {count} diverse RB (rule-based) structures for reasoning tasks \
(arithmetic, algebra, logic, sequences). Each structure has variables, rules, and a derivation.

Each object MUST have: variables, rules, gold_answer, support_facts, derivation_steps.

Example object (illustrative, do not copy):
{
  "variables": [{"name": "x", "role": "unknown"}],
  "rules": ["x + 5 = 12"],
  "gold_answer": "7",
  "support_facts": ["x + 5 = 12"],
  "derivation_steps": ["Subtract 5 from both sides", "x = 12 - 5", "x = 7"]
}

Output ONLY a JSON array of {count} such objects. No markdown, no commentary."""


HYBRID_STRUCTURE_USER = """Generate {count} diverse Hybrid structures requiring BOTH retrieval and reasoning. \
Each is a 2-hop chain (e.g., landmark -> country -> capital, work -> author -> nationality).

Each object MUST have: nodes, edges, gold_answer, support_facts, bridge_entity.

Example object (illustrative, do not copy):
{
  "nodes": [
    {"id": "n1", "label": "Eiffel Tower", "role": "query_entity"},
    {"id": "n2", "label": "France", "role": "intermediate"},
    {"id": "n3", "label": "Paris", "role": "answer"}
  ],
  "edges": [
    {"source": "n1", "target": "n2", "relation": "located_in"},
    {"source": "n2", "target": "n3", "relation": "capital_of"}
  ],
  "gold_answer": "Paris",
  "support_facts": ["Eiffel Tower is in France", "Paris is the capital of France"],
  "bridge_entity": "France"
}

Output ONLY a JSON array of {count} such objects. No markdown, no commentary."""


async def generate_structures(client: APIClient, counts: dict[str, int] = None) -> list[dict[str, Any]]:
    """Generate structures for all task families."""
    counts = counts or FAMILY_COUNTS
    all_structures = []

    # KB structures
    logger.info(f"Generating {counts['KB']} KB structures...")
    kb_user = KB_STRUCTURE_USER.replace("{count}", str(counts['KB']))
    kb_result = await client.call_api_json_async(_STRUCTURE_SYSTEM, kb_user, response_format=None)
    if isinstance(kb_result, list):
        for i, s in enumerate(kb_result):
            s["task_family"] = "KB"
            s["family_id"] = f"kb_{i+1:03d}"
        all_structures.extend(kb_result)

    # RB structures
    logger.info(f"Generating {counts['RB']} RB structures...")
    rb_user = RB_STRUCTURE_USER.replace("{count}", str(counts['RB']))
    rb_result = await client.call_api_json_async(_STRUCTURE_SYSTEM, rb_user, response_format=None)
    if isinstance(rb_result, list):
        for i, s in enumerate(rb_result):
            s["task_family"] = "RB"
            s["family_id"] = f"rb_{i+1:03d}"
        all_structures.extend(rb_result)

    # Hybrid structures
    logger.info(f"Generating {counts['Hybrid']} Hybrid structures...")
    hybrid_user = HYBRID_STRUCTURE_USER.replace("{count}", str(counts['Hybrid']))
    hybrid_result = await client.call_api_json_async(_STRUCTURE_SYSTEM, hybrid_user, response_format=None)
    if isinstance(hybrid_result, list):
        for i, s in enumerate(hybrid_result):
            s["task_family"] = "Hybrid"
            s["family_id"] = f"hybrid_{i+1:03d}"
        all_structures.extend(hybrid_result)

    logger.info(f"Generated {len(all_structures)} structures total")
    return all_structures
