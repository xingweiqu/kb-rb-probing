#!/usr/bin/env python3
"""Rebuild probe-ready paired dataset.

Outputs (default):
  - data/probe_ready_90_natural.jsonl
  - data/probe_ready_90_symbolic.jsonl
  - data/probe_ready_180_paired.jsonl
  - reports/probe_ready_validation_report.json

Design goals:
  - 90 naturalized families: KB 30 + RB 30 + Hybrid 30 (diverse sub_family)
  - 90 strict_symbolic families paired 1:1 with naturalized
  - Exactly these 10 variants per family:
      original, hint, premise, premise_removal, highlight,
      wrongclaim_bare, competing_claims, paraphrase, scaffold_1, scaffold_2
  - No CoT variants, no scaffold_3, no MCQ.
  - Validation enforces P0=0 and writes a report.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_VARIANTS: list[str] = [
    "original",
    "hint",
    "premise",
    "premise_removal",
    "highlight",
    "wrongclaim_bare",
    "competing_claims",
    "paraphrase",
    "scaffold_1",
    "scaffold_2",
]

STRICT_TOKENS: tuple[str, ...] = ("E1", "E2", "R1", "C1", "V1")

# Expected family schema for outputs in this script.
ALLOWED_TOP_LEVEL_KEYS: set[str] = {
    "family_id",
    "task_family",
    "sub_family",
    "base_item_id",
    "underlying_structure",
    "base_question",
    "gold_answer",
    "gold_reasoning_chain",
    "support_facts",
    "required_steps",
    "normal_variants",
    "metadata",
}

ALLOWED_TOP_LEVEL_KEYS_WITH_MCQ: set[str] = set(ALLOWED_TOP_LEVEL_KEYS) | {"mcq_variants"}

# Variants that MUST NOT include the gold answer (P0).
# Note: numeric golds can legitimately appear as constants in RB prompts, so we
# only enforce strong leakage constraints for entity-like golds plus the
# specifically high-risk variants.
NEVER_GOLD_VARIANTS: set[str] = {"wrongclaim_bare", "scaffold_1", "scaffold_2"}


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _token_contains(haystack: str, needle: str) -> bool:
    """Token-ish containment with word boundary fallback."""
    if not needle:
        return False
    h = haystack or ""
    n = needle
    if re.fullmatch(r"[A-Za-z0-9_\-]+", n):
        return re.search(rf"\b{re.escape(n)}\b", h) is not None
    return n in h


def _token_contains_any(haystack: str, needles: Iterable[str]) -> bool:
    for n in needles:
        if n and _token_contains(haystack, n):
            return True
    return False


def _has_letters(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def _answer_type(gold: str) -> str:
    g = (gold or "").strip()
    if not g:
        return "unknown"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", g):
        return "number"
    if _norm(g) in {"true", "false", "yes", "no"}:
        return "boolean"
    return "entity"


def _gold_aliases(gold: str) -> list[str]:
    """Generate a conservative alias set for leakage checks."""
    g = (gold or "").strip()
    if not g:
        return []
    out: list[str] = []

    def _add(x: str) -> None:
        x = (x or "").strip()
        if not x:
            return
        if x not in out:
            out.append(x)

    _add(g)
    # Strip article
    if re.match(r"(?i)^the\s+", g):
        _add(re.sub(r"(?i)^the\s+", "", g).strip())
    # Remove parentheticals
    if "(" in g and ")" in g:
        _add(re.sub(r"\s*\([^)]*\)\s*", " ", g).strip())
    # Comma truncation
    if "," in g:
        _add(g.split(",", 1)[0].strip())
    # Common suffix drops for entity answers
    if g.endswith(" River"):
        _add(g[: -len(" River")].strip())
    if g.endswith(" City"):
        _add(g[: -len(" City")].strip())
    # Boolean normalization
    ng = _norm(g)
    if ng in {"yes", "true"}:
        _add("yes")
        _add("true")
    if ng in {"no", "false"}:
        _add("no")
        _add("false")
    # Numeric normalization
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", g):
        _add(str(float(g)) if "." in g else str(int(float(g))))
    # Filter trivially short aliases
    return [x for x in out if len(_norm(x)) >= 2]


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e
            if isinstance(obj, dict):
                items.append(obj)
    return items


def _jsonl_write(items: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _jsonl_concat(in_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for p in in_paths:
            with open(p, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line.rstrip("\n") + "\n")


def _json_write(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _extract_structure(source: dict[str, Any]) -> dict[str, Any]:
    """Prefer `underlying_structure`; fall back to a shallow structure view."""
    s = source.get("underlying_structure")
    if isinstance(s, dict) and s:
        return s
    # Fallback: preserve a minimal set of fields that describe the structure.
    out: dict[str, Any] = {}
    for k in (
        "sub_family",
        "type",
        "nodes",
        "edges",
        "variables",
        "rules",
        "derivation_steps",
        "required_steps",
        "support_chain",
    ):
        if k in source:
            out[k] = source.get(k)
    return out


def _derive_support_chain(source: dict[str, Any], structure: dict[str, Any]) -> list[str]:
    sc = source.get("support_chain")
    if isinstance(sc, list) and all(isinstance(x, str) for x in sc):
        return list(sc)
    sc2 = structure.get("support_chain")
    if isinstance(sc2, list) and all(isinstance(x, str) for x in sc2):
        return list(sc2)
    # Try from edges/nodes
    nodes = structure.get("nodes") if isinstance(structure.get("nodes"), list) else []
    edges = structure.get("edges") if isinstance(structure.get("edges"), list) else []
    id2label: dict[str, str] = {}
    for n in nodes:
        if isinstance(n, dict) and isinstance(n.get("id"), str) and isinstance(n.get("label"), str):
            id2label[n["id"]] = n["label"]
    out: list[str] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = id2label.get(str(e.get("source", "")), str(e.get("source", "")))
        tgt = id2label.get(str(e.get("target", "")), str(e.get("target", "")))
        rel = str(e.get("relation", ""))
        if src and tgt and rel:
            out.append(f"{src} --{rel}--> {tgt}")
    return out


def _sanitize_base_question(base_q: str, gold: str) -> str:
    """Remove obvious appended-gold patterns from base questions."""
    q = (base_q or "").strip()
    g = (gold or "").strip()
    if not q:
        return q
    if g and _token_contains(q, g):
        # Only do aggressive removal for non-numeric gold.
        if _has_letters(g) or len(g) >= 8:
            # Common pattern: question ends with the conclusion / answer statement.
            # Remove the last occurrence of gold.
            q2 = q
            idx = q2.rfind(g)
            if idx != -1:
                q2 = (q2[:idx] + q2[idx + len(g) :]).strip()
            # Clean repeated punctuation / whitespace.
            q2 = re.sub(r"\s+", " ", q2).strip()
            q2 = re.sub(r"\s*([?.!])\s*", r"\1 ", q2).strip()
            # Ensure it still reads like a question.
            if not re.search(r"[?]$", q2):
                q2 = q2.rstrip(".") + "?"
            q = q2
    return q


def _remove_aliases_from_text(text: str, aliases: list[str]) -> str:
    """Best-effort removal of gold/alias tokens from a prompt."""
    out = text
    for a in sorted({x for x in aliases if x}, key=len, reverse=True):
        if re.fullmatch(r"[A-Za-z0-9_\-]+", a):
            out = re.sub(rf"\b{re.escape(a)}\b", "[REDACTED]", out)
        else:
            out = out.replace(a, "[REDACTED]")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _pick_distractors(pool: list[str], gold: str, k: int, rng: random.Random) -> list[str]:
    aliases = _gold_aliases(gold)
    pool2 = [
        x
        for x in pool
        if isinstance(x, str)
        and x.strip()
        and _norm(x) != _norm(gold)
        and not _token_contains_any(x, aliases)
    ]
    rng.shuffle(pool2)
    out: list[str] = []
    for x in pool2:
        if _norm(x) in {_norm(y) for y in out}:
            continue
        out.append(x)
        if len(out) >= k:
            break
    return out


def _numeric_distractors(gold: str) -> list[str]:
    g = (gold or "").strip()
    if not re.fullmatch(r"[-+]?\d+", g):
        return []
    v = int(g)
    cands = [v + 1, v - 1, v + 2, v - 2, v + 3]
    out: list[str] = []
    for x in cands:
        if x == v:
            continue
        out.append(str(x))
        if len(out) >= 3:
            break
    return out


def build_natural_variants(
    *,
    task_family: str,
    sub_family: str,
    base_question: str,
    gold: str,
    support_facts: list[str],
    distractors: list[str],
    rng: random.Random,
) -> dict[str, dict[str, Any]]:
    """Create 10 probe-ready natural variants."""
    base = base_question.strip()
    aliases = _gold_aliases(gold)

    def _safe_premise() -> str:
        # Prefer a support fact that does not contain gold/aliases.
        for f in support_facts:
            if isinstance(f, str) and f.strip() and not _token_contains_any(f, aliases):
                return f.strip()
        # Generic premises by family
        if task_family == "KB":
            return "Use the relevant relation in a knowledge base to retrieve the requested value."
        if task_family == "RB":
            return "Apply the standard rule system implied by the statements to derive the conclusion."
        return "Use the provided relations/rules and perform the minimal composition needed."
    wrong = distractors[0] if distractors else ("Berlin" if task_family == "KB" else "0")
    wrong2 = distractors[1] if len(distractors) >= 2 else ("Rome" if task_family == "KB" else "1")
    # Ensure wrong options don't accidentally match aliases.
    if _token_contains_any(wrong, aliases):
        wrong = "Madrid" if task_family == "KB" else "2"
    if _token_contains_any(wrong2, aliases) or _norm(wrong2) == _norm(wrong):
        wrong2 = "Lisbon" if task_family == "KB" else "3"

    premise_fact = _safe_premise()

    # Paraphrase templates to reduce cross-family collisions.
    paraphrase_tpls = [
        "Rephrase: {q}",
        "In other words, {q}",
        "Put differently: {q}",
    ]
    paraphrase_q = rng.choice(paraphrase_tpls).format(q=base)

    # Hint templates avoid including the gold; keep them generic.
    hint_tpls = [
        "Hint: use the information implied by the prompt. {q}",
        "Hint: identify the key entity and relation being asked about. {q}",
        "Hint: this is a direct retrieval / short reasoning question. {q}",
    ]
    hint_q = rng.choice(hint_tpls).format(q=base)

    # Premise removal: explicitly remove any added premise.
    premise_removal_q = f"Answer the question without relying on any added premises: {base}"

    # Highlight: emphasize the (possibly leaky) premise fact.
    highlight_q = f"Focus on the key relation being asked about (do not assume any extra facts). {base}"

    # Premise: provide one supporting fact (often includes gold in KB/Hybrid).
    premise_q = f"Premise: {premise_fact} {base}".strip()

    # Wrong-claim: must NOT include gold.
    wrongclaim_q = f"Some sources claim the answer is {wrong}. {base}"

    # Competing claims: MUST NOT include gold/aliases.
    competing_q = f"Some claim the answer is {wrong}, while others claim it is {wrong2}. {base}"

    # Scaffolds: must NOT include gold.
    scaffold_1_q = (
        "Step 1: Identify what the question is asking for. "
        "Step 2: Use the appropriate fact/rule to compute or retrieve it. "
        f"Question: {base}"
    )
    scaffold_2_q = (
        "Step 1: Extract the key entities/variables from the question. "
        "Step 2: Apply the relevant relation/rule. "
        "Step 3: Return the final result only. "
        f"Question: {base}"
    )

    # Ensure all variant questions are strings.
    variants: dict[str, str] = {
        "original": base,
        "hint": hint_q,
        "premise": premise_q,
        "premise_removal": premise_removal_q,
        "highlight": highlight_q,
        "wrongclaim_bare": wrongclaim_q,
        "competing_claims": competing_q,
        "paraphrase": paraphrase_q,
        "scaffold_1": scaffold_1_q,
        "scaffold_2": scaffold_2_q,
    }

    # Hard safety pass: remove aliases if they accidentally appear.
    for k in list(variants.keys()):
        if _token_contains_any(variants[k], aliases):
            variants[k] = _remove_aliases_from_text(variants[k], aliases)

    out: dict[str, dict[str, Any]] = {}
    for k in REQUIRED_VARIANTS:
        out[k] = {"question": variants[k], "metadata": {"variant": k, "task_family": task_family, "sub_family": sub_family}}
    return out


def _kb_symbolic_map(structure: dict[str, Any]) -> dict[str, str]:
    # Map query_entity -> E1, answer -> E2, relation -> R1, first intermediate -> C1.
    out: dict[str, str] = {}
    nodes = structure.get("nodes") if isinstance(structure.get("nodes"), list) else []
    edges = structure.get("edges") if isinstance(structure.get("edges"), list) else []
    q_label = None
    a_label = None
    interm: list[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        lab = n.get("label")
        role = n.get("role")
        if not isinstance(lab, str) or not lab:
            continue
        if role == "query_entity" and q_label is None:
            q_label = lab
        elif role == "answer" and a_label is None:
            a_label = lab
        elif role == "intermediate":
            interm.append(lab)
    if q_label:
        out[q_label] = "E1"
    if a_label:
        out[a_label] = "E2"
    if interm:
        out[interm[0]] = "C1"
    if edges:
        rels: list[str] = []
        for e in edges:
            if isinstance(e, dict) and isinstance(e.get("relation"), str) and e["relation"]:
                rels.append(e["relation"])
        if rels:
            out[rels[0]] = "R1"
        if len(rels) >= 2:
            # second relation token uses V1 (allowed token set)
            out[rels[1]] = "V1"
    return out


def _sanitize_structure_tokens(structure: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Return a structure copy where any labels/relations are replaced with strict tokens."""
    if not isinstance(structure, dict):
        return {}
    out: dict[str, Any] = {}
    # Keep only structural fields.
    for k in ("sub_family", "type", "required_steps"):
        if k in structure:
            out[k] = structure.get(k)
    nodes = structure.get("nodes") if isinstance(structure.get("nodes"), list) else []
    edges = structure.get("edges") if isinstance(structure.get("edges"), list) else []
    out_nodes: list[dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nn = {k: n.get(k) for k in ("id", "role") if k in n}
        lab = n.get("label")
        if isinstance(lab, str) and lab:
            nn["label"] = mapping.get(lab, "C1")
        out_nodes.append(nn)
    out_edges: list[dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        ee = {k: e.get(k) for k in ("source", "target") if k in e}
        rel = e.get("relation")
        if isinstance(rel, str) and rel:
            ee["relation"] = mapping.get(rel, "R1")
        out_edges.append(ee)
    if out_nodes:
        out["nodes"] = out_nodes
    if out_edges:
        out["edges"] = out_edges
    return out


def _rewrite_gold_with_map(text: str, mapping: dict[str, str]) -> str:
    out = text
    for k, v in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        if not k:
            continue
        if re.fullmatch(r"[A-Za-z0-9_\-]+", k):
            out = re.sub(rf"\b{re.escape(k)}\b", v, out)
        else:
            out = out.replace(k, v)
    return out


def build_strict_symbolic_variants(
    *,
    task_family: str,
    sub_family: str,
    scenario_tag: str,
    gold_symbolic: str,
    distractors_symbolic: list[str],
    support_chain_symbolic: list[str],
    rng: random.Random,
) -> dict[str, dict[str, Any]]:
    """Build strict-symbolic variant questions using only E1/E2/R1/C1/V1 tokens."""

    # For strict leak policy, prompts must NOT contain the gold token.
    wrong = distractors_symbolic[0] if distractors_symbolic else "C1"
    wrong2 = distractors_symbolic[1] if len(distractors_symbolic) >= 2 else "V1"
    wrong = wrong if wrong in STRICT_TOKENS and wrong != gold_symbolic else "C1"
    wrong2 = wrong2 if wrong2 in STRICT_TOKENS and wrong2 not in {gold_symbolic, wrong} else "V1"

    # Task/subfamily flavored gloss to reduce prompt collisions while remaining abstract.
    flavor = {
        ("KB", "country_capital"): "a single-hop attribute query",
        ("KB", "work_author"): "a single-hop author lookup",
        ("KB", "element_symbol"): "a single-hop symbol lookup",
        ("KB", "entity_attribute"): "a single-hop attribute lookup",
        ("Hybrid", "two_hop_relational"): "a two-hop relational composition",
        ("Hybrid", "retrieve_transform"): "retrieve + transform",
        ("Hybrid", "retrieve_rule_apply"): "retrieve + rule application",
    }.get((task_family, sub_family), "an abstract reasoning task")

    base = f"Scenario {scenario_tag} ({flavor}): What is the value of relation R1 for entity E1?"
    hint_tpls = [
        "Hint: identify E1 and the relation token R1, then retrieve the linked entity.",
        "Hint: treat this as a short symbolic lookup using R1.",
        "Hint: focus on the role of R1 connecting E1 to its target.",
    ]
    hint_q = f"Scenario {scenario_tag}: {rng.choice(hint_tpls)} {base}"

    # Premise/highlight must NOT include the gold token.
    premise_q = f"Scenario {scenario_tag}: Premise: E1 has exactly one target under relation R1. {base}"
    highlight_q = f"Scenario {scenario_tag}: Focus on the relation token R1 and the query entity E1. {base}"

    premise_removal_q = f"Scenario {scenario_tag}: {base} (No additional premises.)"
    wrongclaim_q = f"Scenario {scenario_tag}: Some sources claim E1 --R1--> {wrong}. {base}"
    competing_q = f"Scenario {scenario_tag}: One claim says E1 --R1--> {wrong}, another says E1 --R1--> {wrong2}. {base}"
    paraphrase_q = f"Scenario {scenario_tag}: Which entity is linked to E1 by relation R1?"

    scaffold_1_q = (
        f"Scenario {scenario_tag}: Step 1: Identify E1 and R1. "
        f"Step 2: Retrieve the unique target of R1 for E1. "
        f"Question: {base}"
    )
    scaffold_2_q = (
        f"Scenario {scenario_tag}: Step 1: Read the symbolic relation tokens. "
        f"Step 2: Use R1 to map from E1 to its target. "
        f"Step 3: Output the target token only. "
        f"Question: {base}"
    )

    variants: dict[str, str] = {
        "original": base,
        "hint": hint_q,
        "premise": premise_q,
        "premise_removal": premise_removal_q,
        "highlight": highlight_q,
        "wrongclaim_bare": wrongclaim_q,
        "competing_claims": competing_q,
        "paraphrase": paraphrase_q,
        "scaffold_1": scaffold_1_q,
        "scaffold_2": scaffold_2_q,
    }
    out: dict[str, dict[str, Any]] = {}
    for k in REQUIRED_VARIANTS:
        out[k] = {
            "question": variants[k],
            "metadata": {
                "variant": k,
                "symbolic_mode": "strict_symbolic",
                "task_family": task_family,
                "sub_family": sub_family,
                "support_chain": support_chain_symbolic,
            },
        }
    return out


@dataclass
class Issue:
    severity: str  # P0 | P1 | P2
    code: str
    family_id: str
    variant: str | None = None
    message: str = ""


def _balanced_correct_indices(n: int, rng: random.Random) -> list[int]:
    """Return a near-uniform multiset of indices 0..3 with max-min <= 1."""
    base = n // 4
    rem = n % 4
    indices: list[int] = []
    for i in range(4):
        indices.extend([i] * base)
    # distribute remainder randomly but deterministically
    extras = list(range(4))
    rng.shuffle(extras)
    for i in extras[:rem]:
        indices.append(i)
    rng.shuffle(indices)
    return indices


def build_strict_symbolic_mcq(question: str, gold_token: str, correct_index: int) -> dict[str, Any]:
    """Build MCQ for strict_symbolic: options are strict tokens only."""
    assert gold_token in STRICT_TOKENS
    distractors = [t for t in STRICT_TOKENS if t != gold_token]
    # Need 3 distractors
    opts = [None, None, None, None]
    opts[correct_index] = gold_token
    di = 0
    for j in range(4):
        if opts[j] is not None:
            continue
        opts[j] = distractors[di]
        di += 1
    return {
        "symbolic_original": {
            "question": question,
            "options": opts,
            "correct_index": int(correct_index),
            "option_metadata": [
                {"role": "gold" if k == correct_index else "distractor", "token": str(opts[k])}
                for k in range(4)
            ],
        }
    }


def validate_mcq_uniformity(symbolic_families: list[dict[str, Any]]) -> list[Issue]:
    """Validate strict_symbolic MCQ schema + A/B/C/D balance."""
    issues: list[Issue] = []
    idx_counts = Counter()
    for f in symbolic_families:
        fid = str(f.get("family_id", ""))
        meta = f.get("metadata") if isinstance(f.get("metadata"), dict) else {}
        if str(meta.get("mode", "")) != "strict_symbolic":
            continue
        mcq = f.get("mcq_variants")
        if not isinstance(mcq, dict):
            issues.append(Issue("P0", "missing_mcq", fid, None, "mcq_variants missing"))
            continue
        item = mcq.get("symbolic_original")
        if not isinstance(item, dict):
            issues.append(Issue("P0", "mcq_schema_invalid", fid, None, "mcq_variants.symbolic_original missing"))
            continue
        q = item.get("question")
        opts = item.get("options")
        ci = item.get("correct_index")
        if not isinstance(q, str) or not q.strip():
            issues.append(Issue("P0", "mcq_empty_question", fid, None, "MCQ question empty"))
            continue
        if not isinstance(opts, list) or len(opts) != 4 or not all(isinstance(x, str) for x in opts):
            issues.append(Issue("P0", "mcq_options_invalid", fid, None, "MCQ options invalid"))
            continue
        if len(set(opts)) != 4:
            issues.append(Issue("P0", "mcq_options_duplicate", fid, None, "MCQ options duplicate"))
        if any(x not in STRICT_TOKENS for x in opts):
            issues.append(Issue("P0", "mcq_option_not_token", fid, None, "MCQ option not strict token"))
        if not isinstance(ci, int) or ci < 0 or ci > 3:
            issues.append(Issue("P0", "mcq_correct_index_invalid", fid, None, "correct_index invalid"))
            continue
        gold = str(f.get("gold_answer", ""))
        if gold not in STRICT_TOKENS:
            issues.append(Issue("P0", "symbolic_gold_not_token", fid, None, "gold_answer not strict token"))
        else:
            if opts[ci] != gold:
                issues.append(Issue("P0", "mcq_gold_mismatch", fid, None, "correct option != gold_answer"))
        idx_counts[ci] += 1

        # HARD GATE: MCQ question must not contain gold token
        aliases = _gold_aliases(gold)
        if aliases and _token_contains_any(q, aliases):
            issues.append(Issue("P0", "mcq_question_leaks_gold", fid, None, "MCQ question contains gold/alias"))

    if idx_counts:
        vals = list(idx_counts.values())
        if max(vals) - min(vals) > 1:
            issues.append(Issue("P0", "mcq_correct_index_not_uniform", "__dataset__", None, f"counts={dict(idx_counts)}"))
    return issues


def validate_dataset(
    natural: list[dict[str, Any]],
    symbolic: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    *,
    require_quality_fields: bool,
    allow_mcq: bool,
) -> dict[str, Any]:
    issues: list[Issue] = []
    issues_by_family: dict[str, list[Issue]] = defaultdict(list)

    def _check_family(f: dict[str, Any]) -> None:
        fid = str(f.get("family_id", ""))

        allowed_keys = ALLOWED_TOP_LEVEL_KEYS_WITH_MCQ if allow_mcq else ALLOWED_TOP_LEVEL_KEYS
        extra_top = sorted([k for k in f.keys() if k not in allowed_keys])
        if extra_top:
            issues.append(Issue("P0", "unexpected_top_level_keys", fid, None, f"extra keys: {extra_top}"))
        meta = f.get("metadata")
        if not isinstance(meta, dict):
            issues.append(Issue("P0", "missing_metadata", fid, None, "family.metadata missing"))
            meta = {}

        mode = str(meta.get("mode", ""))
        gold = str(f.get("gold_answer", "") or "")
        aliases = _gold_aliases(gold)
        if not gold.strip():
            issues.append(Issue("P0", "empty_gold", fid, None, "gold_answer empty"))

        nv = f.get("normal_variants")
        if not isinstance(nv, dict):
            issues.append(Issue("P0", "missing_variants", fid, None, "normal_variants missing"))
            return
        missing = [k for k in REQUIRED_VARIANTS if k not in nv]
        extra = [k for k in nv.keys() if k not in REQUIRED_VARIANTS]
        if missing:
            issues.append(Issue("P0", "missing_variant", fid, None, f"missing variants: {missing}"))
        if extra:
            issues.append(Issue("P0", "extra_variant", fid, None, f"extra variants present: {extra}"))

        # No MCQ / CoT / scaffold_3
        if not allow_mcq:
            if "mcq_variants" in f and f.get("mcq_variants"):
                issues.append(Issue("P0", "mcq_present", fid, None, "mcq_variants must be absent/empty"))
        for bad in ("cot_full", "cot_partial", "cot_shuffled", "scaffold_3", "scaffold_shuffled"):
            if isinstance(nv, dict) and bad in nv:
                issues.append(Issue("P0", "forbidden_variant", fid, bad, f"forbidden variant present: {bad}"))

        # Leakage / constraints
        for vt in REQUIRED_VARIANTS:
            v = nv.get(vt)
            q = v.get("question", "") if isinstance(v, dict) else ""
            if not isinstance(q, str) or not q.strip():
                issues.append(Issue("P0", "empty_prompt", fid, vt, "variant question empty"))
                continue

            # HARD GATE: any prompt/question containing gold_answer or any alias is P0.
            if aliases and _token_contains_any(q, aliases):
                issues.append(Issue("P0", "gold_or_alias_leak", fid, vt, "prompt contains gold/alias"))

            # wrongclaim must actually include a wrong target that differs from gold.
            if vt == "wrongclaim_bare" and gold:
                if aliases and _token_contains_any(q, aliases):
                    issues.append(Issue("P0", "wrongclaim_contains_gold", fid, vt, "wrongclaim contains gold/alias"))

            # competing_claims MUST NOT contain gold claim.
            if vt == "competing_claims" and aliases and _token_contains_any(q, aliases):
                issues.append(Issue("P0", "competing_claims_contains_gold", fid, vt, "competing_claims contains gold/alias"))

            # strict_symbolic must not contain obvious real-world entities (heuristic: no mapping keys + no Unicode symbols)
            if mode == "strict_symbolic":
                # Disallow common overlay symbols.
                if re.search(r"[∆◇⊕Ω★⊗▽◆⊙⊞⊘⊛⊜⊝⊟⊠]", q):
                    issues.append(Issue("P0", "symbolic_contains_overlay_symbols", fid, vt, "contains unicode overlay symbols"))

                # Gold/distractors must be strict tokens only.
                if str(f.get("gold_answer", "")) not in STRICT_TOKENS:
                    issues.append(Issue("P0", "symbolic_gold_not_token", fid, None, "gold_answer not strict token"))
                atc = meta.get("atomic_target_candidates") if isinstance(meta, dict) else None
                if not isinstance(atc, list) or not atc:
                    issues.append(Issue("P0", "symbolic_missing_candidates", fid, None, "atomic_target_candidates missing"))
                else:
                    bad = [x for x in atc if not isinstance(x, str) or x not in STRICT_TOKENS]
                    if bad:
                        issues.append(Issue("P0", "symbolic_candidate_not_token", fid, None, f"bad candidates: {bad}"))

                # Do not allow real-world entity strings from underlying_structure to appear in prompts.
                st = f.get("underlying_structure") if isinstance(f.get("underlying_structure"), dict) else {}
                forb: list[str] = []
                for n in (st.get("nodes") or []):
                    if isinstance(n, dict) and isinstance(n.get("label"), str):
                        lab = n["label"]
                        if _has_letters(lab) and len(_norm(lab)) >= 3 and lab not in STRICT_TOKENS:
                            forb.append(lab)
                for e in (st.get("edges") or []):
                    if isinstance(e, dict) and isinstance(e.get("relation"), str):
                        rel = e["relation"]
                        if _has_letters(rel) and len(_norm(rel)) >= 3 and rel not in STRICT_TOKENS:
                            forb.append(rel)
                if forb and _token_contains_any(q, forb):
                    issues.append(Issue("P0", "symbolic_real_entity_leak", fid, vt, "prompt contains real-world label"))

        # Required metadata fields (quality fields can be enforced after validator-derived annotation)
        for key in (
            "task_family",
            "mode",
            "source_family_id",
            "paired_family_id",
            "answer_type",
            "support_chain",
            "atomic_target_candidates",
            "variant_names",
        ):
            if key not in meta:
                issues.append(Issue("P0", "missing_metadata_field", fid, None, f"metadata missing {key}"))

        if require_quality_fields:
            for key in ("quality_grade", "usable_main_experiment"):
                if key not in meta:
                    issues.append(Issue("P0", "missing_quality_field", fid, None, f"metadata missing {key}"))
            qg = str(meta.get("quality_grade", ""))
            usable = meta.get("usable_main_experiment")
            if qg and qg not in {"A", "B", "C", "D"}:
                issues.append(Issue("P1", "invalid_quality_grade", fid, None, f"quality_grade={qg}"))
            if usable not in {True, False}:
                issues.append(Issue("P1", "invalid_usable_flag", fid, None, f"usable_main_experiment={usable!r}"))
            if usable is True and qg == "D":
                issues.append(Issue("P1", "quality_usable_inconsistent", fid, None, "usable but grade D"))

    # Counts
    def _count_by_tf(items: list[dict[str, Any]]) -> Counter[str]:
        c: Counter[str] = Counter()
        for f in items:
            c[str(f.get("task_family", ""))] += 1
        return c

    if len(natural) != 90:
        issues.append(Issue("P0", "count_natural", "__dataset__", None, f"natural count={len(natural)}"))
    if len(symbolic) != 90:
        issues.append(Issue("P0", "count_symbolic", "__dataset__", None, f"symbolic count={len(symbolic)}"))
    if len(paired) != 180:
        issues.append(Issue("P0", "count_paired", "__dataset__", None, f"paired count={len(paired)}"))

    nat_tf = _count_by_tf(natural)
    sym_tf = _count_by_tf(symbolic)
    if nat_tf != Counter({"KB": 30, "RB": 30, "Hybrid": 30}):
        issues.append(Issue("P0", "natural_tf_mismatch", "__dataset__", None, f"{dict(nat_tf)}"))
    if sym_tf != Counter({"KB": 30, "RB": 30, "Hybrid": 30}):
        issues.append(Issue("P0", "symbolic_tf_mismatch", "__dataset__", None, f"{dict(sym_tf)}"))

    # Pair alignment
    nat_by_id = {str(f.get("family_id")): f for f in natural}
    sym_by_id = {str(f.get("family_id")): f for f in symbolic}
    aligned = 0
    for nf in natural:
        nm = nf.get("metadata") if isinstance(nf.get("metadata"), dict) else {}
        sid = str(nm.get("paired_family_id", ""))
        sf = sym_by_id.get(sid)
        if sf is None:
            issues.append(Issue("P0", "pair_missing", str(nf.get("family_id")), None, f"missing symbolic pair {sid}"))
            continue
        sm = sf.get("metadata") if isinstance(sf.get("metadata"), dict) else {}
        if str(sm.get("paired_family_id", "")) != str(nf.get("family_id")):
            issues.append(Issue("P0", "pair_mismatch", str(nf.get("family_id")), None, "paired_family_id not reciprocal"))
            continue
        if str(sf.get("task_family", "")) != str(nf.get("task_family", "")) or str(sf.get("sub_family", "")) != str(
            nf.get("sub_family", "")
        ):
            issues.append(Issue("P0", "pair_schema_mismatch", str(nf.get("family_id")), None, "task/sub mismatch"))
            continue
        aligned += 1

    # Duplicate prompts (global)
    seen: dict[str, tuple[str, str]] = {}
    dup_count = 0
    for f in paired:
        fid = str(f.get("family_id", ""))
        nv = f.get("normal_variants")
        if not isinstance(nv, dict):
            continue
        for vt in REQUIRED_VARIANTS:
            v = nv.get(vt)
            q = v.get("question", "") if isinstance(v, dict) else ""
            key = _norm(q)
            if not key:
                continue
            if key in seen:
                dup_count += 1
                ofid, ovt = seen[key]
                issues.append(Issue("P0", "duplicate_prompt", fid, vt, f"duplicates {ofid}:{ovt}"))
            else:
                seen[key] = (fid, vt)

    for f in paired:
        if isinstance(f, dict):
            before = len(issues)
            _check_family(f)
            after = len(issues)
            if after > before:
                fid = str(f.get("family_id", ""))
                issues_by_family[fid].extend(issues[before:after])

    sev_counts = Counter(i.severity for i in issues)
    code_counts = Counter(i.code for i in issues)
    leak_by_variant = Counter(i.variant for i in issues if i.code in {"gold_leak", "gold_in_never_variant", "wrongclaim_contains_gold"} and i.variant)

    # Variant coverage summary
    var_coverage: dict[str, int] = Counter()
    for f in paired:
        nv = f.get("normal_variants")
        if not isinstance(nv, dict):
            continue
        for k in REQUIRED_VARIANTS:
            v = nv.get(k)
            q = v.get("question") if isinstance(v, dict) else None
            if isinstance(q, str) and q.strip():
                var_coverage[k] += 1

    # Count by task_family & mode
    ctm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in paired:
        tf = str(f.get("task_family", ""))
        meta = f.get("metadata") if isinstance(f.get("metadata"), dict) else {}
        mode = str(meta.get("mode", ""))
        ctm[tf][mode] += 1

    quality_grades = Counter(
        str((f.get("metadata") or {}).get("quality_grade", ""))
        for f in paired
        if isinstance(f.get("metadata"), dict)
    )
    usable_count = sum(
        1
        for f in paired
        if isinstance(f.get("metadata"), dict) and f["metadata"].get("usable_main_experiment") is True
    )

    report = {
        "summary": {
            "counts": {"natural": len(natural), "symbolic": len(symbolic), "paired": len(paired)},
            "count_by_task_family_mode": {k: dict(v) for k, v in ctm.items()},
            "variant_coverage": dict(var_coverage),
            "severity_counts": dict(sev_counts),
            "issue_counts": dict(code_counts),
            "leakage_count_by_variant": dict(leak_by_variant),
            "pair_alignment_pass_rate": aligned / max(1, len(natural)),
            "quality_grade_count": dict(quality_grades),
            "usable_count": usable_count,
            "duplicate_prompt_violations": dup_count,
        },
        "issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "family_id": i.family_id,
                **({"variant": i.variant} if i.variant else {}),
                "message": i.message,
            }
            for i in issues
        ],
        "issues_by_family": {
            fid: [
                {
                    "severity": i.severity,
                    "code": i.code,
                    **({"variant": i.variant} if i.variant else {}),
                    "message": i.message,
                }
                for i in lst
            ]
            for fid, lst in issues_by_family.items()
        },
    }
    return report


def _apply_quality_from_issues(families: list[dict[str, Any]], issues_by_family: dict[str, list[dict[str, Any]]]) -> None:
    """Set quality_grade / usable_main_experiment from validator results."""
    by_id: dict[str, list[str]] = {}
    for fid, lst in issues_by_family.items():
        sevs = [str(i.get("severity", "")) for i in lst if isinstance(i, dict)]
        by_id[fid] = sevs

    def grade_from_sevs(sevs: list[str]) -> str:
        if any(s == "P0" for s in sevs):
            return "D"
        if any(s == "P1" for s in sevs):
            return "C"
        if any(s == "P2" for s in sevs):
            return "B"
        return "A"

    for f in families:
        fid = str(f.get("family_id", ""))
        sevs = by_id.get(fid, [])
        meta = f.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            f["metadata"] = meta
        qg = grade_from_sevs(sevs)
        meta["quality_grade"] = qg
        meta["usable_main_experiment"] = (qg in {"A", "B"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("synthesis_output_120b/output/dataset.jsonl"),
        help="Input dataset.jsonl to sample underlying structures from",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_natural", type=Path, default=Path("data/probe_ready_90_natural.jsonl"))
    ap.add_argument("--out_symbolic", type=Path, default=Path("data/probe_ready_90_symbolic.jsonl"))
    ap.add_argument("--out_paired", type=Path, default=Path("data/probe_ready_180_paired.jsonl"))
    ap.add_argument(
        "--out_report",
        type=Path,
        default=Path("reports/probe_ready_validation_report.json"),
    )
    ap.add_argument("--out_symbolic_mcq", type=Path, default=Path("data/probe_ready_90_symbolic_mcq.jsonl"))
    ap.add_argument("--out_paired_mcq", type=Path, default=Path("data/probe_ready_180_paired_mcq.jsonl"))
    ap.add_argument(
        "--out_report_mcq",
        type=Path,
        default=Path("reports/probe_ready_validation_report_mcq.json"),
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    src = _jsonl_load(args.input)
    # Keep only the 90 canonical families.
    src = [
        d
        for d in src
        if isinstance(d, dict)
        and str(d.get("task_family", "")) in {"KB", "RB", "Hybrid"}
        and isinstance(d.get("family_id"), str)
    ]

    # Build distractor pools by (task_family, sub_family)
    by_tf_sf: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    gold_pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    for d in src:
        tf = str(d.get("task_family"))
        sf = str(d.get("sub_family"))
        by_tf_sf[(tf, sf)].append(d)
        ga = d.get("gold_answer")
        if isinstance(ga, str) and ga.strip():
            gold_pool[(tf, sf)].append(ga.strip())

    # Deterministic ordering for reproducibility.
    src_sorted = sorted(src, key=lambda d: (str(d.get("task_family")), str(d.get("sub_family")), str(d.get("family_id"))))

    natural_out: list[dict[str, Any]] = []
    symbolic_out: list[dict[str, Any]] = []

    # Scenario tags: unique, stable per paired family.
    scenario_counter: dict[str, int] = {"KB": 0, "RB": 0, "Hybrid": 0}

    for d in src_sorted:
        tf = str(d.get("task_family"))
        sf = str(d.get("sub_family"))
        source_fid = str(d.get("family_id"))
        nat_id = f"probe_nat_{source_fid}"
        sym_id = f"probe_sym_{source_fid}"
        scenario_counter[tf] += 1
        scenario_tag = f"{tf.lower()}-{scenario_counter[tf]:02d}"

        gold = str(d.get("gold_answer", "") or "").strip()
        structure = _extract_structure(d)
        support_chain = _derive_support_chain(d, structure)
        support_facts = d.get("support_facts") if isinstance(d.get("support_facts"), list) else []
        support_facts = [str(x) for x in support_facts if isinstance(x, str) and x.strip()]

        base_q_src = str(d.get("base_question", "") or "")
        base_q = _sanitize_base_question(base_q_src, gold)

        # Distractors
        pool = gold_pool.get((tf, sf), [])
        distractors = []
        if _answer_type(gold) == "number":
            distractors = _numeric_distractors(gold)
        if not distractors:
            distractors = _pick_distractors(pool, gold, k=3, rng=rng)

        # Natural variants
        per_rng = random.Random((_norm(nat_id) + str(args.seed)).encode("utf-8"))
        nv = build_natural_variants(
            task_family=tf,
            sub_family=sf,
            base_question=base_q,
            gold=gold,
            support_facts=support_facts,
            distractors=distractors,
            rng=per_rng,
        )

        # Symbolic mapping from structure
        sym_map = _kb_symbolic_map(structure)
        # STRICT symbolic answers/distractors must be tokens only.
        gold_symbolic = "E2"
        distractors_symbolic = ["C1", "V1", "E1"]

        # Symbolic support chain: rewrite support_chain with tokens.
        support_chain_symbolic = [_rewrite_gold_with_map(s, sym_map) for s in support_chain]
        if not support_chain_symbolic:
            support_chain_symbolic = ["E1 --R1--> E2"]

        sym_variants = build_strict_symbolic_variants(
            task_family=tf,
            sub_family=sf,
            scenario_tag=scenario_tag,
            gold_symbolic=gold_symbolic if gold_symbolic else "E2",
            distractors_symbolic=distractors_symbolic,
            support_chain_symbolic=support_chain_symbolic,
            rng=random.Random((_norm(sym_id) + str(args.seed)).encode("utf-8")),
        )

        # Atomic target candidates
        candidates = [gold] + [x for x in distractors if x and _norm(x) != _norm(gold)]
        candidates = candidates[:4] if candidates else []
        if _answer_type(gold) == "number" and gold:
            # Ensure at least 4 numeric candidates if possible.
            while len(candidates) < 4:
                candidates.append(str(int(gold) + len(candidates)))

        candidates_sym = ["E2", "C1", "V1", "E1"]

        nat_meta = {
            "task_family": tf,
            "mode": "naturalized",
            "source_family_id": source_fid,
            "paired_family_id": sym_id,
            "gold_answer": gold,
            "answer_type": _answer_type(gold),
            "support_chain": support_chain,
            "atomic_target_candidates": candidates,
            "variant_names": list(REQUIRED_VARIANTS),
            "symbolic_map": sym_map,
        }
        sym_meta = {
            "task_family": tf,
            "mode": "strict_symbolic",
            "source_family_id": source_fid,
            "paired_family_id": nat_id,
            "gold_answer": "E2",
            "answer_type": "entity",
            "support_chain": support_chain_symbolic,
            "atomic_target_candidates": candidates_sym,
            "variant_names": list(REQUIRED_VARIANTS),
            # Do NOT store real-world label mappings in strict_symbolic metadata.
            "symbolic_map": {"E1": "E1", "E2": "E2", "R1": "R1", "C1": "C1", "V1": "V1"},
        }

        nat_family = {
            "family_id": nat_id,
            "task_family": tf,
            "sub_family": sf,
            "base_item_id": f"{nat_id}_base",
            "underlying_structure": structure,
            "base_question": base_q,
            "gold_answer": gold,
            "gold_reasoning_chain": d.get("gold_reasoning_chain", []) if isinstance(d.get("gold_reasoning_chain"), list) else [],
            "support_facts": support_facts,
            "required_steps": int(d.get("required_steps", structure.get("required_steps", 1)) or 1),
            "normal_variants": nv,
            "metadata": nat_meta,
        }

        sym_structure = _sanitize_structure_tokens(structure, sym_map)
        sym_family = {
            "family_id": sym_id,
            "task_family": tf,
            "sub_family": sf,
            "base_item_id": f"{sym_id}_base",
            "underlying_structure": sym_structure,
            "base_question": sym_variants["original"]["question"],
            "gold_answer": "E2",
            "gold_reasoning_chain": [],
            "support_facts": [],
            "required_steps": int(d.get("required_steps", structure.get("required_steps", 1)) or 1),
            "normal_variants": sym_variants,
            "metadata": sym_meta,
        }

        natural_out.append(nat_family)
        symbolic_out.append(sym_family)

    paired_out = natural_out + symbolic_out

    # Phase 1 validate (quality fields not required yet)
    report1 = validate_dataset(natural_out, symbolic_out, paired_out, require_quality_fields=False, allow_mcq=False)
    _apply_quality_from_issues(paired_out, report1.get("issues_by_family", {}))
    # Phase 2 validate (quality fields required)
    report = validate_dataset(natural_out, symbolic_out, paired_out, require_quality_fields=True, allow_mcq=False)
    sev_counts = report.get("summary", {}).get("severity_counts", {})
    if int(sev_counts.get("P0", 0)) != 0:
        _json_write(report, args.out_report)
        raise SystemExit(f"Validation failed: P0={sev_counts.get('P0')} (see {args.out_report})")

    # HARD GATE PASSED -> write final outputs
    _jsonl_write(natural_out, args.out_natural)
    _jsonl_write(symbolic_out, args.out_symbolic)
    _jsonl_concat([args.out_natural, args.out_symbolic], args.out_paired)

    # Post-write validation on disk (guards against accidental schema drift)
    nat2 = _jsonl_load(args.out_natural)
    sym2 = _jsonl_load(args.out_symbolic)
    pair2 = _jsonl_load(args.out_paired)
    report_disk = validate_dataset(nat2, sym2, pair2, require_quality_fields=True, allow_mcq=False)
    sev_disk = report_disk.get("summary", {}).get("severity_counts", {})
    if int(sev_disk.get("P0", 0)) != 0:
        _json_write(report_disk, args.out_report)
        raise SystemExit(f"Post-write validation failed: P0={sev_disk.get('P0')} (see {args.out_report})")

    _json_write(report_disk, args.out_report)

    # === MCQ version (strict_symbolic only) ===
    symbolic_mcq_out: list[dict[str, Any]] = []
    # Deterministic balanced correct_index assignment.
    sym_sorted = sorted(symbolic_out, key=lambda x: str(x.get("family_id", "")))
    idxs = _balanced_correct_indices(len(sym_sorted), rng=random.Random(args.seed + 101))
    for fam, ci in zip(sym_sorted, idxs):
        f2 = dict(fam)
        f2_meta = f2.get("metadata") if isinstance(f2.get("metadata"), dict) else {}
        f2["metadata"] = dict(f2_meta)
        gold_tok = str(f2.get("gold_answer", "E2"))
        q = str(((f2.get("normal_variants") or {}).get("original") or {}).get("question") or f2.get("base_question") or "")
        f2["mcq_variants"] = build_strict_symbolic_mcq(q, gold_tok, int(ci))
        symbolic_mcq_out.append(f2)

    # Validate MCQ families (schema + uniformity)
    mcq_issues = validate_mcq_uniformity(symbolic_mcq_out)
    if any(i.severity == "P0" for i in mcq_issues):
        rep_mcq = {
            "summary": {
                "severity_counts": dict(Counter(i.severity for i in mcq_issues)),
                "issue_counts": dict(Counter(i.code for i in mcq_issues)),
            },
            "issues": [
                {"severity": i.severity, "code": i.code, "family_id": i.family_id, "message": i.message}
                for i in mcq_issues
            ],
        }
        _json_write(rep_mcq, args.out_report_mcq)
        raise SystemExit(f"MCQ validation failed: P0 (see {args.out_report_mcq})")

    # Write MCQ files and validate (allow_mcq=True)
    _jsonl_write(symbolic_mcq_out, args.out_symbolic_mcq)
    _jsonl_concat([args.out_natural, args.out_symbolic_mcq], args.out_paired_mcq)
    nat3 = _jsonl_load(args.out_natural)
    sym3 = _jsonl_load(args.out_symbolic_mcq)
    pair3 = _jsonl_load(args.out_paired_mcq)
    # dataset-level validator (permits mcq key)
    rep3 = validate_dataset(nat3, sym3, pair3, require_quality_fields=True, allow_mcq=True)
    sev3 = rep3.get("summary", {}).get("severity_counts", {})
    if int(sev3.get("P0", 0)) != 0:
        _json_write(rep3, args.out_report_mcq)
        raise SystemExit(f"MCQ post-write validation failed: P0={sev3.get('P0')} (see {args.out_report_mcq})")
    # Attach MCQ uniformity summary into report_mcq
    rep3["mcq_uniformity"] = {
        "severity_counts": dict(Counter(i.severity for i in mcq_issues)),
        "issue_counts": dict(Counter(i.code for i in mcq_issues)),
    }
    _json_write(rep3, args.out_report_mcq)

    print(f"Wrote: {args.out_natural}")
    print(f"Wrote: {args.out_symbolic}")
    print(f"Wrote: {args.out_paired}")
    print(f"Wrote: {args.out_report}")
    print(f"Wrote: {args.out_symbolic_mcq}")
    print(f"Wrote: {args.out_paired_mcq}")
    print(f"Wrote: {args.out_report_mcq}")
    print(
        f"Validation: P0={sev_disk.get('P0', 0)} P1={sev_disk.get('P1', 0)} P2={sev_disk.get('P2', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
