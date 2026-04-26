"""Strict symbolic builder.

`symbolic_overlay` (existing) uses preamble "France = ∆" which leaks real entities.
`strict_symbolic` aims to remove real-world entity dependence while preserving roles.

Strategy:
- Extract entities/relations from underlying structure.
- Assign abstract role tokens (Q1/A1/I1/R1/...) independent of original labels.
- Map role tokens to symbols (SYMBOL_POOL).
- Replace original labels in question text with symbols, prepend a role-token preamble.

Note:
- This is still natural language around relations (e.g. "capital") but removes entity names.
- It is deterministic and model-free.
"""

from __future__ import annotations

import re
from typing import Any

from ..configs.defaults import SYMBOL_POOL, VARIANT_TYPES


def _regex_replace_token(text: str, old: str, new: str) -> str:
    if not old:
        return text
    if re.fullmatch(r"[A-Za-z0-9_\-]+", old):
        pat = rf"\b{re.escape(old)}\b"
        return re.sub(pat, new, text)
    return text.replace(old, new)


def _extract_role_labels(structure: dict[str, Any]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {"query_entity": [], "intermediate": [], "answer": [], "relation": []}
    for n in structure.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        label = n.get("label")
        role = n.get("role")
        if isinstance(label, str) and label:
            r = str(role or "")
            if r in roles:
                roles[r].append(label)
            else:
                roles.setdefault("other", []).append(label)
    for e in structure.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        rel = e.get("relation")
        if isinstance(rel, str) and rel:
            roles["relation"].append(rel)
    # RB variables treated as entities (but not real-world)
    for v in structure.get("variables", []) or []:
        if not isinstance(v, dict):
            continue
        name = v.get("name")
        if isinstance(name, str) and name:
            roles.setdefault("variable", []).append(name)
    return roles


def build_strict_symbol_maps(structure: dict[str, Any]) -> dict[str, Any]:
    """Build mapping objects for strict symbolic.

    Returns:
      - label_to_role_token: original label -> role token (Q1/A1/I1/R1/...)
      - role_token_to_symbol: role token -> symbol
    """
    roles = _extract_role_labels(structure)

    label_to_role_token: dict[str, str] = {}
    role_tokens: list[str] = []

    def _assign(prefix: str, labels: list[str]) -> None:
        seen: set[str] = set()
        for lbl in labels:
            if lbl in seen:
                continue
            seen.add(lbl)
            tok = f"{prefix}{len(role_tokens)+1}"
            label_to_role_token[lbl] = tok
            role_tokens.append(tok)

    _assign("Q", roles.get("query_entity", []))
    _assign("I", roles.get("intermediate", []))
    _assign("A", roles.get("answer", []))
    _assign("R", roles.get("relation", []))
    # variables (RB) last
    _assign("V", roles.get("variable", []))

    role_token_to_symbol: dict[str, str] = {}
    for i, tok in enumerate(role_tokens):
        role_token_to_symbol[tok] = SYMBOL_POOL[i % len(SYMBOL_POOL)]

    return {
        "label_to_role_token": label_to_role_token,
        "role_token_to_symbol": role_token_to_symbol,
    }


def build_strict_symbolic_preamble(role_token_to_symbol: dict[str, str]) -> str:
    parts = [f"{k} = {v}" for k, v in role_token_to_symbol.items()]
    return "In this system, " + ", ".join(parts) + "."


def rewrite_to_strict_symbolic(text: str, label_to_role_token: dict[str, str], role_token_to_symbol: dict[str, str]) -> str:
    # Replace labels -> symbols (via role token indirection).
    # Longest-first to avoid partial overlaps.
    items = sorted(label_to_role_token.items(), key=lambda kv: len(kv[0]), reverse=True)
    out = text
    for lbl, tok in items:
        sym = role_token_to_symbol.get(tok)
        if sym:
            out = _regex_replace_token(out, lbl, sym)
    return out


def build_strict_symbolic_variants(family: dict[str, Any]) -> dict[str, Any]:
    structure = family.get("underlying_structure", {})
    if not isinstance(structure, dict):
        structure = {}

    maps = build_strict_symbol_maps(structure)
    label_to_role_token = maps["label_to_role_token"]
    role_token_to_symbol = maps["role_token_to_symbol"]
    preamble = build_strict_symbolic_preamble(role_token_to_symbol)

    nv = family.get("normal_variants", {})
    base_q = str(family.get("base_question", "") or "")

    variants: dict[str, Any] = {}
    for vt in VARIANT_TYPES:
        src_q = base_q
        src_meta: dict[str, Any] = {}
        if isinstance(nv, dict) and isinstance(nv.get(vt), dict):
            src_q = str(nv[vt].get("question", "") or base_q)
            m = nv[vt].get("metadata", {})
            if isinstance(m, dict):
                src_meta = dict(m)

        q2 = rewrite_to_strict_symbolic(src_q, label_to_role_token, role_token_to_symbol)
        q2 = preamble + "\n" + q2
        variants[vt] = {
            "question": q2,
            "metadata": {
                "symbolic_mode": "strict_symbolic",
                "source_variant": vt,
                **({"original_metadata": src_meta} if src_meta else {}),
            },
        }

    return {
        "symbolic_mode": "strict_symbolic",
        "label_to_role_token": label_to_role_token,
        "role_token_to_symbol": role_token_to_symbol,
        "variants": variants,
    }

