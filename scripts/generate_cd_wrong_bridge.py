"""Generate the missing Wrong-Bridge Drop (CD) variant for Hybrid items.

The existing dataset (`runs/full_25/output/dataset.jsonl`) has eight hybrid
items per backbone: original + explicit_fact + retrieval_blocked + both_blocked
in natural and symbolic surface modes (4 variants × 2 modes × 25 backbones =
200 hybrid items). The 2x3 capacity grid expects a ninth Distract-on-Composition
cell. We add it as `wrong_bridge`: a plausible but wrong bridge fact `B'`
prepended to the original question, with the rule `R` and gold answer `A`
unchanged.

We do NOT replace `both_blocked` here. We append wrong_bridge as a new variant
so existing hidden-state extraction and probe runs remain valid; the paper
moves both_blocked to an appendix lower-bound control. After this script
runs, hybrid backbones have 5 variants × 2 modes = 10 items each (kb / rb
families unchanged at 4 × 2).

Usage (server, with API access):

    python -m scripts.generate_cd_wrong_bridge \
        --structures runs/full_25/checkpoints/01_structures.json \
        --base_items runs/full_25/checkpoints/02_base_items.json \
        --existing_dataset runs/full_25/output/dataset.jsonl \
        --output_jsonl runs/full_25/output/dataset_with_cd.jsonl

Outputs:
    A new combined dataset.jsonl that contains every existing item plus 50
    wrong_bridge items (25 natural, 25 symbolic). The symbolic mode reuses
    the symbolic-mode entity map already produced for each backbone; if the
    symbolic side cannot be constructed (entity map missing), only natural
    items are emitted and the script logs a warning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

# Bring in the existing API client / config so the model and retry policy match
# the rest of the data construction.
from dataset_synthesis_mvp.api_client import APIClient
from dataset_synthesis_mvp.config import (
    ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, MODEL, SYMBOL_POOL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


WRONG_BRIDGE_SYSTEM = (
    "You are a dataset designer that outputs ONLY a JSON object. "
    "No markdown, no code fences, no preamble. Output a single JSON object only."
)


WRONG_BRIDGE_USER = """Generate a `wrong_bridge` variant for a 2-hop hybrid item.

The original task chains a query entity Q to an intermediate bridge entity B
to a final answer A via two relations. The `wrong_bridge` variant prepends
a plausible but FALSE bridge claim to the question, asserting that Q is
linked to a different bridge entity B' (same type as B). The reasoning rule
R that maps a bridge entity to an answer is unchanged. The original gold
answer A is still the correct final answer; if the model accepts the false
bridge B', it would derive a different answer A'.

Inputs:
- Original question: {base_question}
- Gold answer (unchanged): {gold_answer}
- True bridge entity B: {bridge_entity}
- A list of plausible alternative bridge entities of the same type:
  {alt_bridges}

Choose ONE alternative B' from the list (or propose your own if none fit).
Phrase a single short sentence asserting that Q is linked to B' (NOT B).
Then prepend that sentence to the original question. The variant question
must keep the original interrogative; do not replace it.

Output JSON object with EXACTLY these keys:
- question: string. The wrong-bridge sentence followed by the original question.
- metadata: object with keys
    wrong_bridge: string (the chosen B')
    wrong_bridge_claim: string (the inserted false sentence)
    wrong_bridge_implied_answer: string (the answer A' that B' + R would yield;
                                          MUST be different from A; otherwise
                                          pick a different B')
    true_bridge: string (B, copied from input)
- variant_type: must be the literal string "wrong_bridge".

Hard constraints:
- The variant question MUST NOT contain the gold answer A literally.
- The wrong_bridge claim MUST NOT mention A.
- B' MUST be different from B and of the same type (e.g., country-for-country,
  author-for-author).
- CRITICAL: the implied answer A' (what B' + R yields) MUST be different
  from A. For example, if A is a nationality and B' shares the same
  nationality as B (e.g. both Hemingway and Fitzgerald are American), pick
  a different B' whose nationality is NOT American (e.g. Camus, Tolstoy,
  Mishima). If A is a currency and B' shares the same currency zone as B
  (e.g. France/Belgium both Euro), pick a B' whose currency differs.
- The original question text after the prepended claim MUST be preserved
  verbatim.

Output the JSON object only."""


def _alt_bridge_candidates(structure: dict[str, Any]) -> list[str]:
    """Best-effort list of plausible alternative bridge entities. We do not
    have a knowledge graph at hand; we lean on the structure to give us
    enough type signal that Claude can pick a sensible B'."""
    intermediate = next((n for n in structure.get("nodes", [])
                         if n.get("role") == "intermediate"), None)
    if not intermediate:
        return []
    label = intermediate.get("label", "")
    # Cheap heuristic alternatives: nearby entities of the same kind. We
    # don't try to enumerate exhaustively; Claude will refine.
    geographic_alts = ["Belgium", "Switzerland", "Austria", "Portugal",
                       "Greece", "Norway", "Sweden", "Finland", "Croatia"]
    if label and label[0].isupper():
        return [a for a in geographic_alts if a != label][:5]
    return []


async def generate_one(client: APIClient,
                       structure: dict[str, Any],
                       base_item: dict[str, Any]) -> dict[str, Any] | None:
    """Generate one wrong_bridge variant for a hybrid backbone."""
    bridge = structure.get("bridge_entity") or ""
    alts = _alt_bridge_candidates(structure)
    user = WRONG_BRIDGE_USER.format(
        base_question=base_item["base_question"],
        gold_answer=base_item["gold_answer"],
        bridge_entity=bridge,
        alt_bridges=json.dumps(alts) if alts else "[]",
    )
    try:
        result = await client.call_api_json_async(
            WRONG_BRIDGE_SYSTEM, user, response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning("API call failed for %s: %s", base_item.get("family_id"), e)
        return None

    question = (result.get("question") or "").strip()
    metadata = result.get("metadata") or {}
    if not question:
        logger.warning("%s: empty question; skipping", base_item.get("family_id"))
        return None
    if base_item["gold_answer"].lower() in question.lower():
        logger.warning("%s: gold answer leak in wrong_bridge; skipping",
                       base_item.get("family_id"))
        return None

    return {
        "family_id": base_item["family_id"],
        "task_family": "Hybrid",
        "variant": "wrong_bridge",
        "mode": "natural",
        "question": question,
        "gold_answer": base_item["gold_answer"],
        "metadata": {
            "variant_type": "wrong_bridge",
            "wrong_bridge": metadata.get("wrong_bridge", ""),
            "wrong_bridge_claim": metadata.get("wrong_bridge_claim", ""),
            "wrong_bridge_implied_answer": metadata.get("wrong_bridge_implied_answer", ""),
            "true_bridge": metadata.get("true_bridge", bridge),
        },
    }


def _build_symbolic(natural_item: dict[str, Any],
                    symbolic_originals: dict[tuple[str, str], dict],
                    symbol_pool: list[str]) -> dict | None:
    """Construct a symbolic-mode wrong_bridge by extending the existing
    entity_map with the wrong_bridge entity (and its implied wrong answer
    if available), then substituting throughout. Also rebuilds the
    "In this system, X = ⊕..." preamble so the symbol legend is complete.

    Without this extension the wrong_bridge entity would leak as the
    natural-language string into symbolic mode, defeating the purpose of
    the symbolic surface.
    """
    fid = natural_item["family_id"]
    sym_orig = symbolic_originals.get((fid, "symbolic"))
    if not sym_orig:
        return None
    base_map: dict[str, str] = dict((sym_orig.get("metadata") or {}).get("entity_map") or {})
    if not base_map:
        return None

    extended_map = dict(base_map)
    used_symbols = set(extended_map.values())
    available = [s for s in symbol_pool if s not in used_symbols]

    meta = natural_item["metadata"]
    wrong_bridge = (meta.get("wrong_bridge") or "").strip()
    wrong_implied = (meta.get("wrong_bridge_implied_answer") or "").strip()
    if wrong_bridge and wrong_bridge not in extended_map:
        if not available:
            return None
        extended_map[wrong_bridge] = available.pop(0)
    if wrong_implied and wrong_implied not in extended_map:
        if not available:
            available = [chr(0x2600 + i) for i in range(20)]  # extra fallback symbols
        extended_map[wrong_implied] = available.pop(0)

    # Build the preamble in the same shape as symbolic.py:
    # "In this system, A = α, B = β, ...\n<question>"
    # We must list every entity in extended_map.
    mappings = ", ".join(f"{k} = {v}" for k, v in extended_map.items())
    preamble = f"In this system, {mappings}.\n"

    # Take the natural question (which itself has no preamble), substitute,
    # then prepend the preamble.
    nat_q = natural_item["question"]
    sym_q = nat_q
    for k in sorted(extended_map.keys(), key=len, reverse=True):
        v = extended_map[k]
        sym_q = sym_q.replace(k, v)
    sym_gold = extended_map.get(natural_item["gold_answer"], natural_item["gold_answer"])

    return {
        "family_id": fid,
        "task_family": "Hybrid",
        "variant": "wrong_bridge",
        "mode": "symbolic",
        "question": preamble + sym_q,
        "gold_answer": sym_gold,
        "metadata": {
            **natural_item["metadata"],
            "entity_map": extended_map,
            "symbolic_mode": True,
            "derived_from_natural": True,
        },
    }


async def run(structures_path: Path, base_items_path: Path,
              existing_dataset_path: Path, output_path: Path,
              concurrency: int = 8) -> None:
    structures = json.load(open(structures_path))
    base_items = json.load(open(base_items_path))
    existing = [json.loads(line) for line in open(existing_dataset_path, encoding="utf-8")]

    hybrid_structures = {s["family_id"]: s for s in structures
                         if s.get("task_family") == "Hybrid"}
    hybrid_bases = {b["family_id"]: b for b in base_items
                    if b.get("task_family") == "Hybrid"}
    logger.info("hybrid backbones: structures=%d, bases=%d",
                len(hybrid_structures), len(hybrid_bases))

    # Index symbolic originals so we can mirror the variant in symbolic mode.
    symbolic_originals = {
        (it["family_id"], it["mode"]): it
        for it in existing
        if it.get("variant") == "original" and it.get("mode") == "symbolic"
    }

    client = APIClient(model=MODEL, mock=False,
                       base_url=ANTHROPIC_BASE_URL,
                       api_key=ANTHROPIC_AUTH_TOKEN)
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(fid: str) -> dict | None:
        async with sem:
            structure = hybrid_structures.get(fid)
            base = hybrid_bases.get(fid)
            if structure is None or base is None:
                return None
            return await generate_one(client, structure, base)

    try:
        tasks = [_bounded(fid) for fid in sorted(hybrid_structures)]
        natural_items = [r for r in await asyncio.gather(*tasks) if r is not None]
    finally:
        await client.aclose()

    logger.info("generated %d natural wrong_bridge items (target=%d)",
                len(natural_items), len(hybrid_structures))

    symbolic_items = []
    for nat in natural_items:
        sym = _build_symbolic(nat, symbolic_originals, list(SYMBOL_POOL))
        if sym:
            symbolic_items.append(sym)
    logger.info("derived %d symbolic wrong_bridge items", len(symbolic_items))

    # Drop any pre-existing wrong_bridge rows (idempotent re-runs).
    existing = [it for it in existing if it.get("variant") != "wrong_bridge"]
    combined = existing + natural_items + symbolic_items

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for it in combined:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    logger.info("wrote %d items to %s", len(combined), output_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--structures", default="runs/full_25/checkpoints/01_structures.json")
    p.add_argument("--base_items", default="runs/full_25/checkpoints/02_base_items.json")
    p.add_argument("--existing_dataset", default="runs/full_25/output/dataset.jsonl")
    p.add_argument("--output_jsonl", default="runs/full_25/output/dataset_with_cd.jsonl")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    asyncio.run(run(Path(args.structures), Path(args.base_items),
                    Path(args.existing_dataset), Path(args.output_jsonl),
                    args.concurrency))


if __name__ == "__main__":
    main()
