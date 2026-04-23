"""Dataclass definitions for the item-family dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Variant:
    question: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCQItem:
    question: str
    options: list[str]
    correct_index: int
    option_metadata: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SymbolicVariants:
    entity_map: dict[str, str]
    source_family_id: str
    variants: dict[str, Variant] = field(default_factory=dict)


@dataclass
class Family:
    family_id: str
    task_family: str  # KB | RB | Hybrid | SymbolicControl
    sub_family: str
    base_item_id: str

    underlying_structure: dict[str, Any]

    base_question: str
    gold_answer: str
    gold_reasoning_chain: list[str] = field(default_factory=list)
    support_facts: list[str] = field(default_factory=list)
    required_steps: int = 1

    normal_variants: dict[str, Variant] = field(default_factory=dict)
    symbolic_variants: SymbolicVariants | None = None
    mcq_variants: dict[str, MCQItem] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "family_id": self.family_id,
            "task_family": self.task_family,
            "sub_family": self.sub_family,
            "base_item_id": self.base_item_id,
            "underlying_structure": self.underlying_structure,
            "base_question": self.base_question,
            "gold_answer": self.gold_answer,
            "gold_reasoning_chain": self.gold_reasoning_chain,
            "support_facts": self.support_facts,
            "required_steps": self.required_steps,
            "normal_variants": {
                k: {"question": v.question, "metadata": v.metadata}
                for k, v in self.normal_variants.items()
            },
            "symbolic_variants": None,
            "mcq_variants": {},
            "metadata": self.metadata,
        }
        if self.symbolic_variants:
            d["symbolic_variants"] = {
                "entity_map": self.symbolic_variants.entity_map,
                "source_family_id": self.symbolic_variants.source_family_id,
                **{
                    k: {"question": v.question, "metadata": v.metadata}
                    for k, v in self.symbolic_variants.variants.items()
                },
            }
        for k, mcq in self.mcq_variants.items():
            d["mcq_variants"][k] = {
                "question": mcq.question,
                "options": mcq.options,
                "correct_index": mcq.correct_index,
                "option_metadata": mcq.option_metadata,
            }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Family:
        normal = {
            k: Variant(question=v["question"], metadata=v.get("metadata", {}))
            for k, v in d.get("normal_variants", {}).items()
        }
        sym_raw = d.get("symbolic_variants")
        sym = None
        if sym_raw and isinstance(sym_raw, dict) and "entity_map" in sym_raw:
            entity_map = sym_raw["entity_map"]
            source_id = sym_raw.get("source_family_id", "")
            sym_variants = {}
            for k, v in sym_raw.items():
                if k in ("entity_map", "source_family_id"):
                    continue
                if isinstance(v, dict) and "question" in v:
                    sym_variants[k] = Variant(
                        question=v["question"], metadata=v.get("metadata", {})
                    )
            sym = SymbolicVariants(
                entity_map=entity_map,
                source_family_id=source_id,
                variants=sym_variants,
            )
        mcq = {}
        for k, v in d.get("mcq_variants", {}).items():
            if isinstance(v, dict) and "options" in v:
                mcq[k] = MCQItem(
                    question=v["question"],
                    options=v["options"],
                    correct_index=v["correct_index"],
                    option_metadata=v.get("option_metadata", []),
                )
        return cls(
            family_id=d["family_id"],
            task_family=d["task_family"],
            sub_family=d["sub_family"],
            base_item_id=d["base_item_id"],
            underlying_structure=d.get("underlying_structure", {}),
            base_question=d.get("base_question", ""),
            gold_answer=d.get("gold_answer", ""),
            gold_reasoning_chain=d.get("gold_reasoning_chain", []),
            support_facts=d.get("support_facts", []),
            required_steps=d.get("required_steps", 1),
            normal_variants=normal,
            symbolic_variants=sym,
            mcq_variants=mcq,
            metadata=d.get("metadata", {}),
        )
