"""MCQ repair builder: enforce option type consistency + improve distractors.

This module is deterministic and model-free.
"""

from __future__ import annotations

import re
from typing import Any

from ..configs.defaults import SYMBOL_POOL


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _guess_type(x: str) -> str:
    s = x.strip()
    if not s:
        return "empty"
    lo = s.lower()
    if lo in {"yes", "no", "true", "false"}:
        return "bool"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return "number"
    if s in SYMBOL_POOL or (len(s) == 1 and ("\u2200" <= s <= "\u22ff" or "\u2190" <= s <= "\u21ff")):
        return "symbol"
    return "entity"


def _extract_wrongclaim_answer(family: dict[str, Any]) -> str | None:
    nv = family.get("normal_variants")
    if not isinstance(nv, dict):
        return None
    wc = nv.get("wrongclaim_bare")
    if not isinstance(wc, dict):
        return None
    meta = wc.get("metadata", {})
    if not isinstance(meta, dict):
        return None
    wrong_claim = meta.get("wrong_claim")
    if not isinstance(wrong_claim, str) or not wrong_claim.strip():
        return None
    # Try extract answer-like token from common patterns.
    m = re.search(r"\b(?:is|=)\s*([^\.\n\r\t]+)", wrong_claim)
    if m:
        cand = m.group(1).strip().strip('"').strip("'")
        # cut at comma
        cand = cand.split(",")[0].strip()
        if cand:
            return cand
    return None


def _get_structurally_related(family: dict[str, Any]) -> str | None:
    st = family.get("underlying_structure", {})
    if not isinstance(st, dict):
        return None
    # Prefer query entity label.
    for n in st.get("nodes", []) or []:
        if isinstance(n, dict) and n.get("role") == "query_entity" and isinstance(n.get("label"), str):
            return n["label"]
    # Otherwise any non-answer node.
    for n in st.get("nodes", []) or []:
        if isinstance(n, dict) and isinstance(n.get("label"), str):
            return n["label"]
    return None


def _same_type_distractor(gold: str, t: str) -> str:
    if t == "bool":
        lo = gold.strip().lower()
        return "No" if lo in {"yes", "true"} else "Yes"
    if t == "number":
        try:
            if "." in gold:
                x = float(gold)
                return str(x + 1.0)
            x = int(gold)
            return str(x + 1)
        except Exception:
            return "0"
    if t == "symbol":
        for s in SYMBOL_POOL:
            if s != gold:
                return s
        return SYMBOL_POOL[0]
    # entity
    return gold + "_alt"


def _coerce_to_type(x: str, t: str) -> str:
    s = str(x).strip()
    if t == "bool":
        lo = s.lower()
        if lo in {"yes", "true"}:
            return "Yes"
        if lo in {"no", "false"}:
            return "No"
        return "No"
    if t == "number":
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        return m.group(0) if m else "0"
    if t == "symbol":
        # If already a symbol, keep; otherwise pick a stable symbol.
        if s in SYMBOL_POOL:
            return s
        return SYMBOL_POOL[0]
    return s


def repair_mcq(family: dict[str, Any]) -> dict[str, Any] | None:
    """Repair `mcq_variants.symbolic_original` in-place-ish.

    Returns updated mcq_variants dict or None if not present.
    """
    mcq = family.get("mcq_variants")
    if not isinstance(mcq, dict) or not mcq:
        return None
    item = mcq.get("symbolic_original")
    if not isinstance(item, dict):
        return None

    opts = item.get("options")
    if not isinstance(opts, list) or len(opts) != 4:
        opts = [str(x) for x in (opts or [])]
        # pad
        while len(opts) < 4:
            opts.append("")
        opts = opts[:4]

    ci = item.get("correct_index")
    if not isinstance(ci, int) or not (0 <= ci < 4):
        ci = 0

    gold = str(opts[ci]).strip()
    t_gold = _guess_type(gold)
    if t_gold == "empty":
        # fallback to gold_answer
        gold = str(family.get("gold_answer", "") or "").strip()
        t_gold = _guess_type(gold)
        if t_gold == "empty":
            t_gold = "entity"

    target_type = t_gold

    # Build candidates.
    same_type = _same_type_distractor(gold, target_type)
    related = _get_structurally_related(family) or same_type
    wrong = _extract_wrongclaim_answer(family) or same_type

    related = _coerce_to_type(related, target_type)
    wrong = _coerce_to_type(wrong, target_type)
    same_type = _coerce_to_type(same_type, target_type)

    new_opts = [gold, same_type, related, wrong]
    def _unique_alt(seen_norm: set[str], *, start_k: int = 1) -> str:
        """Generate a type-consistent alternative that's not in `seen_norm`."""
        # Prefer a deterministic pool for symbols.
        if target_type == "symbol":
            for s in SYMBOL_POOL:
                if _norm(s) not in seen_norm:
                    return s
            # Shouldn't happen; fallback
            return SYMBOL_POOL[0]

        if target_type == "bool":
            for s in ("Yes", "No"):
                if _norm(s) not in seen_norm:
                    return s
            return "No"

        if target_type == "number":
            # Try to anchor around a numeric gold; otherwise just use 0,1,2,...
            base = 0.0
            m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(gold).strip())
            if m:
                try:
                    base = float(m.group(0))
                except Exception:
                    base = 0.0
            k = start_k
            while k < 1000:
                cand = str(int(base + k)) if float(int(base)) == base else str(base + float(k))
                if _norm(cand) not in seen_norm:
                    return cand
                k += 1
            return "0"

        # entity (default)
        k = start_k
        while k < 1000:
            cand = f"{gold}_alt{k}"
            if _norm(cand) not in seen_norm:
                return cand
            k += 1
        return f"{gold}_alt"

    # Ensure uniqueness + non-empty
    seen: set[str] = set()
    deduped: list[str] = []
    for idx, o in enumerate(new_opts):
        o = str(o).strip()
        n = _norm(o)
        if (not o) or (n in seen):
            o = _unique_alt(seen, start_k=idx + 1)
            n = _norm(o)
        seen.add(n)
        deduped.append(o)

    new_meta = [
        {"role": "gold", "source": "gold"},
        {"role": "same_type", "source": "generated"},
        {"role": "structurally_related", "source": "structure"},
        {"role": "wrongclaim_aligned", "source": "wrongclaim"},
    ]

    mcq["symbolic_original"] = {
        "question": str(item.get("question", "") or ""),
        "options": deduped,
        "correct_index": 0,
        "option_metadata": new_meta,
        "metadata": {
            "target_type": target_type,
            "repaired": True,
        },
    }
    return mcq
