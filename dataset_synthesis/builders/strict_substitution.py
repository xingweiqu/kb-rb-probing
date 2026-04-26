"""Strict substitution builder (structure-preserving).

Goal: fix the `substitution` variant so it does NOT become a different question.

Rules:
- KB: prefer pseudoword replacement for entities (no real-world entity swap)
- RB: variable renaming only (structure unchanged)
- Hybrid: pseudoword replacement for graph nodes/relations while preserving roles

This module is deterministic and model-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_PSEUDO_SYLLABLES = [
    "dax",
    "wug",
    "blick",
    "zorp",
    "tarn",
    "plin",
    "mep",
    "fep",
    "kiv",
    "lorp",
    "narg",
    "sprol",
]


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _extract_graph_entities(structure: dict[str, Any]) -> list[tuple[str, str]]:
    """Return list of (label, role) for graph nodes; role used for stability."""
    out: list[tuple[str, str]] = []
    for n in structure.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        label = n.get("label")
        role = n.get("role")
        if isinstance(label, str) and label:
            out.append((label, str(role or "")))
    return out


def _extract_relations(structure: dict[str, Any]) -> list[str]:
    rels: list[str] = []
    for e in structure.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        r = e.get("relation")
        if isinstance(r, str) and r:
            rels.append(r)
    return rels


def _extract_rb_variables(structure: dict[str, Any]) -> list[str]:
    vars_: list[str] = []
    for v in structure.get("variables", []) or []:
        if not isinstance(v, dict):
            continue
        name = v.get("name")
        if isinstance(name, str) and name and re.fullmatch(r"[A-Za-z]+", name):
            vars_.append(name)
    # also try infer from rules
    for rule in structure.get("rules", []) or []:
        if isinstance(rule, str):
            for m in re.findall(r"\b[A-Za-z]\b", rule):
                vars_.append(m)
    # unique stable
    seen: set[str] = set()
    out: list[str] = []
    for x in vars_:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _regex_replace_token(text: str, old: str, new: str) -> str:
    if not old:
        return text
    if re.fullmatch(r"[A-Za-z0-9_\-]+", old):
        pat = rf"\b{re.escape(old)}\b"
        return re.sub(pat, new, text)
    return text.replace(old, new)


def build_pseudoword_map(labels: list[str]) -> dict[str, str]:
    """Map each label to a lowercase pseudoword."""
    out: dict[str, str] = {}
    for i, lbl in enumerate(labels):
        syl = _PSEUDO_SYLLABLES[i % len(_PSEUDO_SYLLABLES)]
        # add suffix to avoid accidental collisions
        out[lbl] = f"{syl}{i+1}"
    return out


def rewrite_text_with_map(text: str, subst_map: dict[str, str]) -> str:
    """Apply token-aware replacements. Longest-first to avoid partial overlaps."""
    items = sorted(subst_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    out = text
    for old, new in items:
        out = _regex_replace_token(out, old, new)
    return out


def build_strict_substitution(
    family: dict[str, Any],
    *,
    variant_key: str = "substitution",
) -> dict[str, Any]:
    """Return a new {question, metadata} for the substitution variant."""
    structure = family.get("underlying_structure", {})
    if not isinstance(structure, dict):
        structure = {}
    task = str(family.get("task_family", ""))
    base_q = str(family.get("base_question", "") or "")

    if task in {"KB", "Hybrid"}:
        ents = _extract_graph_entities(structure)
        rels = _extract_relations(structure)
        labels = [lbl for (lbl, _role) in ents]
        # Also include relations that may appear in text.
        labels += [r for r in rels if r not in labels]
        subst_map = build_pseudoword_map(labels)
        q = rewrite_text_with_map(base_q, subst_map)
        q = _norm_ws(q)
        return {
            "question": q,
            "metadata": {
                "substitution_map": subst_map,
                "strict": True,
                "strategy": "pseudoword",
            },
        }

    # RB
    if task == "RB":
        vars_ = _extract_rb_variables(structure)
        # Map each variable to a different variable name deterministically.
        target_pool = [c for c in "xyzuvwst" if c not in {v.lower() for v in vars_}]
        subst_map: dict[str, str] = {}
        for i, v in enumerate(vars_):
            new = target_pool[i % len(target_pool)]
            subst_map[v] = new
        q = rewrite_text_with_map(base_q, subst_map)
        q = _norm_ws(q)
        return {
            "question": q,
            "metadata": {
                "substitution_map": subst_map,
                "strict": True,
                "strategy": "variable_rename",
            },
        }

    # Fallback: no-op but marked
    return {
        "question": base_q,
        "metadata": {"strict": False, "strategy": "noop"},
    }

