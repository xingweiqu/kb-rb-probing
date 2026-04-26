"""Variant generation — Stage 3 of the pipeline.

Generates all 19 variant types for each base item via a single API call per family.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .api_client import APIClient
from .configs.defaults import BATCH_SIZE, CONCURRENCY, VARIANT_TYPES

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

HARD CONSTRAINTS (machine-checked downstream):
- For `premise`: the question MUST include at least one of `support_facts` verbatim, and `metadata.injected_premise` MUST be a non-empty list.
- For `wrongclaim_bare`: `metadata.wrong_claim` MUST be a non-empty string and MUST appear verbatim in the question text.
- For `wrongclaim_confident`: include the SAME `metadata.wrong_claim` and include `metadata.confidence_wrapper`.
- For `wrongclaim_attributed`: include the SAME `metadata.wrong_claim` and include `metadata.attribution`.
- For `competing_claims`: include BOTH `metadata.correct_claim` and `metadata.wrong_claim`, and both MUST appear in the question text.
- For `scaffold_shuffled` and `cot_shuffled`: include `metadata.original_order` and `metadata.shuffled_order` as arrays.
- For `cot_partial`: include `metadata.included_steps` and `metadata.omitted_steps` as arrays.

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


async def generate_variants_async(client: APIClient, family: dict[str, Any]) -> dict[str, Any]:
    system, user = build_variant_prompt(family)
    logger.info("Generating variants for %s...", family.get("family_id"))
    result_any = await client.call_api_json_async(system, user, response_format={"type": "json_object"})
    result = _coerce_variants_payload(result_any)

    if not isinstance(result, dict):
        # If the model returned a JSON array (common failure mode), ask it to convert
        # to the required dict-of-variants schema.
        fix_system = (
            "You are a JSON schema converter. Convert the given JSON into a JSON object "
            "mapping each variant name to {question, metadata}. "
            "The required keys are exactly: "
            + ", ".join(VARIANT_TYPES)
            + ". Return ONLY valid JSON (an object), no markdown."
        )
        fix_user = (
            "BASE_ITEM:\n"
            + json.dumps(
                {
                    "family_id": family.get("family_id"),
                    "base_question": family.get("base_question"),
                    "gold_answer": family.get("gold_answer"),
                    "support_facts": family.get("support_facts", []),
                },
                ensure_ascii=False,
            )
            + "\n\nINPUT_JSON:\n"
            + json.dumps(result_any, ensure_ascii=False)
        )
        fixed_any = await client.call_api_json_async(
            fix_system,
            fix_user,
            response_format={"type": "json_object"},
        )
        fixed = _coerce_variants_payload(fixed_any)
        if isinstance(fixed, dict):
            result = fixed
        else:
            raise ValueError(f"Expected dict from variant generation, got {type(result_any)}")

    result = _ensure_variant_keys(result, family)

    # Lightweight schema/constraint validation; if it fails, ask model to repair once.
    issues = _validate_variant_constraints(result, family)
    if issues:
        repair_system = (
            "You are a dataset JSON fixer. You will be given a JSON object of 19 variants. "
            "Fix it so it satisfies the constraints. Return ONLY valid JSON."
        )
        repair_user = (
            "CONSTRAINT FAILURES:\n- "
            + "\n- ".join(issues)
            + "\n\nBASE ITEM:\n"
            + json.dumps(
                {
                    "family_id": family.get("family_id"),
                    "base_question": family.get("base_question"),
                    "gold_answer": family.get("gold_answer"),
                    "support_facts": family.get("support_facts", []),
                },
                ensure_ascii=False,
            )
            + "\n\nCURRENT JSON:\n"
            + json.dumps(result, ensure_ascii=False)
        )
        result2_any = await client.call_api_json_async(
            repair_system,
            repair_user,
            response_format={"type": "json_object"},
        )
        result2 = _coerce_variants_payload(result2_any)
        if isinstance(result2, dict):
            result = _ensure_variant_keys(result2, family)

    # Last-resort cleanup for premise_removal leaking key support facts.
    result = _sanitize_premise_removal(result, family)

    # Last-resort enforcement for machine-checkable constraints.
    result = _enforce_machine_constraints(result, family)

    return result


def _ensure_variant_keys(result: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    """Ensure all 19 variant keys exist and have non-empty question strings."""
    base_q = family.get("base_question", "") or ""
    out: dict[str, Any] = dict(result)

    # Ensure original always matches base.
    out.setdefault("original", {"question": base_q, "metadata": {}})
    if isinstance(out.get("original"), dict):
        if not out["original"].get("question"):
            out["original"]["question"] = base_q

    for vt in VARIANT_TYPES:
        if vt not in out or not isinstance(out.get(vt), dict):
            out[vt] = {"question": "", "metadata": {"auto_filled": True}}
        q = out[vt].get("question", "")
        if not isinstance(q, str) or not q.strip():
            # Minimal non-empty fallback; avoids breaking downstream.
            if vt == "premise_removal":
                out[vt]["question"] = "Answer the question without using any provided premises: " + base_q
            elif vt.startswith("scaffold_"):
                out[vt]["question"] = "Follow the steps, then answer: " + base_q
            elif vt.startswith("cot_"):
                out[vt]["question"] = base_q + " Let's think step by step. Therefore, the answer is?"
            else:
                out[vt]["question"] = base_q
    return out


def _sanitize_premise_removal(result: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    support_facts = family.get("support_facts", []) or []
    if not support_facts:
        return result
    pr = result.get("premise_removal")
    if not isinstance(pr, dict):
        return result
    q = pr.get("question", "")
    if not isinstance(q, str) or not q:
        return result
    # Remove any support fact strings if they appear verbatim.
    for sf in support_facts:
        if isinstance(sf, str) and sf and sf in q:
            q = q.replace(sf, "")
    pr["question"] = " ".join(q.split())
    result["premise_removal"] = pr
    return result


def _enforce_machine_constraints(result: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    """Programmatic fixes for constraints we can enforce without semantics."""

    base_q = family.get("base_question", "") or ""
    support_facts = family.get("support_facts", []) or []
    sf0 = support_facts[0] if support_facts and isinstance(support_facts[0], str) else ""

    def _get(vt: str) -> dict[str, Any]:
        v = result.get(vt)
        if not isinstance(v, dict):
            v = {"question": "", "metadata": {}}
            result[vt] = v
        v.setdefault("metadata", {})
        if not isinstance(v.get("metadata"), dict):
            v["metadata"] = {}
        return v

    # Premise must include at least one support fact verbatim (use the first).
    if sf0:
        prem = _get("premise")
        q = prem.get("question", "")
        meta = prem.get("metadata", {})
        if isinstance(q, str) and sf0 not in q:
            prem["question"] = f"{sf0} {base_q}".strip()
        inj = meta.get("injected_premise")
        if not (isinstance(inj, list) and len(inj) > 0):
            meta["injected_premise"] = [sf0]

    # wrongclaim_* must include wrong_claim string.
    wc_bare = _get("wrongclaim_bare")
    wc = wc_bare.get("metadata", {}).get("wrong_claim")
    if isinstance(wc, str) and wc.strip():
        if wc.strip() not in str(wc_bare.get("question", "")):
            wc_bare["question"] = f"Some sources say {wc.strip()} {base_q}".strip()

        # Confident/attributed must reuse same wrong_claim.
        wc_conf = _get("wrongclaim_confident")
        wc_attr = _get("wrongclaim_attributed")

        wc_conf_meta = wc_conf.get("metadata", {})
        wc_attr_meta = wc_attr.get("metadata", {})
        wc_conf_meta["wrong_claim"] = wc
        wc_attr_meta["wrong_claim"] = wc

        wrapper = wc_conf_meta.get("confidence_wrapper")
        if not isinstance(wrapper, str) or not wrapper.strip():
            wrapper = "Obviously"
            wc_conf_meta["confidence_wrapper"] = wrapper
        if wc.strip() not in str(wc_conf.get("question", "")):
            wc_conf["question"] = f"{wrapper.strip()}, {wc.strip()} {base_q}".strip()

        attr = wc_attr_meta.get("attribution")
        if not isinstance(attr, str) or not attr.strip():
            attr = "experts"
            wc_attr_meta["attribution"] = attr
        if wc.strip() not in str(wc_attr.get("question", "")):
            wc_attr["question"] = f"According to {attr.strip()}, {wc.strip()} {base_q}".strip()

        # Competing claims must include both claims.
        cc = _get("competing_claims")
        cc_meta = cc.get("metadata", {})
        gold = family.get("gold_answer", "")
        correct_claim = cc_meta.get("correct_claim")
        wrong_claim = cc_meta.get("wrong_claim")
        if not isinstance(correct_claim, str) or not correct_claim.strip():
            # Best-effort correct claim.
            if isinstance(gold, str) and gold.strip():
                correct_claim = f"the answer is {gold.strip()}"
            else:
                correct_claim = "the first claim is correct"
            cc_meta["correct_claim"] = correct_claim
        if not isinstance(wrong_claim, str) or not wrong_claim.strip():
            wrong_claim = wc
            cc_meta["wrong_claim"] = wrong_claim

        cc_q = str(cc.get("question", ""))
        if correct_claim.strip() not in cc_q or str(wrong_claim).strip() not in cc_q:
            cc["question"] = f"Some say {correct_claim.strip()}, others say {str(wrong_claim).strip()}. {base_q}".strip()

    return result


def _coerce_variants_payload(payload: Any) -> Any:
    """Coerce common model failure modes into the expected dict-of-variants."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        # Sometimes returned as a 1-element list containing the dict.
        if len(payload) == 1 and isinstance(payload[0], dict):
            return payload[0]
        # Sometimes returned as a list of {name, question, metadata} items.
        out: dict[str, Any] = {}
        ok = True
        for item in payload:
            if not isinstance(item, dict):
                ok = False
                break
            name = item.get("name") or item.get("variant") or item.get("variant_name")
            if not isinstance(name, str) or not name:
                ok = False
                break
            q = item.get("question", "")
            meta = item.get("metadata", {})
            out[name] = {"question": q, "metadata": meta}
        if ok and out:
            return out
    return payload


def _validate_variant_constraints(result: dict[str, Any], family: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    support_facts = family.get("support_facts", []) or []
    sf0 = support_facts[0] if support_facts and isinstance(support_facts[0], str) else ""

    def _q(vt: str) -> str:
        v = result.get(vt, {})
        return v.get("question", "") if isinstance(v, dict) else ""

    def _m(vt: str) -> dict[str, Any]:
        v = result.get(vt, {})
        meta = v.get("metadata", {}) if isinstance(v, dict) else {}
        return meta if isinstance(meta, dict) else {}

    if sf0:
        prem_q = _q("premise")
        injected = _m("premise").get("injected_premise")
        if sf0 not in prem_q:
            issues.append("premise.question must include support_facts[0] verbatim")
        if not (isinstance(injected, list) and len(injected) > 0):
            issues.append("premise.metadata.injected_premise must be a non-empty list")

        pr_q = _q("premise_removal")
        if sf0 in pr_q:
            issues.append("premise_removal.question must NOT include support_facts[0]")

    # Ensure all expected keys are present.
    missing = [vt for vt in VARIANT_TYPES if vt not in result]
    if missing:
        issues.append("missing required variant keys: " + ", ".join(missing))

    empty = [vt for vt in VARIANT_TYPES if not isinstance(_q(vt), str) or not _q(vt).strip()]
    if empty:
        issues.append("empty variant questions: " + ", ".join(empty))

    wc = _m("wrongclaim_bare").get("wrong_claim")
    wc_q = _q("wrongclaim_bare")
    if not (isinstance(wc, str) and wc.strip()):
        issues.append("wrongclaim_bare.metadata.wrong_claim must be a non-empty string")
    elif wc.strip() not in wc_q:
        issues.append("wrongclaim_bare.question must contain metadata.wrong_claim verbatim")

    # Confident/attributed should reuse same wrong_claim.
    for vt, key in [("wrongclaim_confident", "confidence_wrapper"), ("wrongclaim_attributed", "attribution")]:
        meta = _m(vt)
        if wc and meta.get("wrong_claim") != wc:
            issues.append(f"{vt}.metadata.wrong_claim must equal wrongclaim_bare.metadata.wrong_claim")
        if key not in meta or not isinstance(meta.get(key), str) or not meta.get(key):
            issues.append(f"{vt}.metadata.{key} must be a non-empty string")

    cc = _m("competing_claims")
    if not (isinstance(cc.get("correct_claim"), str) and cc.get("correct_claim")):
        issues.append("competing_claims.metadata.correct_claim must be a non-empty string")
    if not (isinstance(cc.get("wrong_claim"), str) and cc.get("wrong_claim")):
        issues.append("competing_claims.metadata.wrong_claim must be a non-empty string")
    if isinstance(cc.get("correct_claim"), str) and cc.get("correct_claim") not in _q("competing_claims"):
        issues.append("competing_claims.question must include metadata.correct_claim verbatim")
    if isinstance(cc.get("wrong_claim"), str) and cc.get("wrong_claim") not in _q("competing_claims"):
        issues.append("competing_claims.question must include metadata.wrong_claim verbatim")

    return issues


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


async def generate_all_variants_async(
    client: APIClient,
    families: list[dict[str, Any]],
    *,
    concurrency: int = CONCURRENCY,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Generate variants for all families with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def one(family: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            variants = await generate_variants_async(client, family)
        return {**family, "normal_variants": variants}

    enriched: list[dict[str, Any]] = []
    for i in range(0, len(families), batch_size):
        batch = families[i : i + batch_size]
        out = await asyncio.gather(*(one(f) for f in batch))
        enriched.extend(out)
    return enriched
