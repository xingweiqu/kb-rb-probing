# Pilot Item-Family Dataset Synthesis — Design Spec

## Goal

Build a modular, API-driven pipeline that synthesizes ~80-100 item families for behavioral probing. Each family has a latent structure, 19 variant types aligned to 12 atomic capabilities, symbolic counterparts (Unicode symbols), and MCQ versions (symbolic priority).

No probing, training, evaluation, or execution — code only.

## Atomic Capabilities & Variant Mapping

The dataset is designed to probe 12 atomic capabilities organized into 4 major categories:

| Category | Atomic Capability | Variant(s) |
|---|---|---|
| Premise-sensitive | evidence-access | `original`, `hint`, `premise`, `premise_removal` |
| Premise-sensitive | evidence-localization | `highlight` |
| Premise-sensitive | evidence-integration | `full_support_bundle` |
| Scaffold-sensitive | decomposition-sensitive | `scaffold_1`, `scaffold_2`, `scaffold_3` |
| Scaffold-sensitive | order-sensitive | `scaffold_shuffled` |
| Scaffold-sensitive | intermediate-state-sensitive | `cot_full`, `cot_partial` |
| Wrong-claim-susceptible | cue-susceptible | `wrongclaim_bare` |
| Wrong-claim-susceptible | authority-susceptible | `wrongclaim_confident`, `wrongclaim_attributed` |
| Wrong-claim-susceptible | conflict-resolution-weak | `competing_claims` |
| Substitution-fragile | lexical-fragile | `paraphrase` |
| Substitution-fragile | terminology-fragile | `terminology_swap` |
| Substitution-fragile | structure-misaligned | `substitution` |

Plus `cot_shuffled` (CoT chain with shuffled steps). Total: **19 variant types**.

## Blocks

| Block | Count | Purpose |
|---|---|---|
| KB | 20-30 families | Factual/associative binding (country→capital, work→author, element→symbol) |
| RB | 20-30 families | Explicit rule/structure derivation (logic, arithmetic, algebra, function composition) |
| Hybrid | 20-30 families | Knowledge access + composition (two-hop, retrieve+transform, retrieve+rule) |
| SymbolicControl | 10-20 families | Derived from normal families via Unicode symbol substitution |

## Family Schema

```json
{
    "family_id": "kb_001",
    "task_family": "KB",
    "sub_family": "country_capital",
    "base_item_id": "kb_001_base",

    "underlying_structure": {
        "type": "single_hop_binding",
        "nodes": [
            {"id": "n1", "label": "Germany", "role": "query_entity"},
            {"id": "n2", "label": "Berlin", "role": "answer"}
        ],
        "edges": [
            {"source": "n1", "target": "n2", "relation": "capital_of"}
        ],
        "support_chain": ["n1 --capital_of--> n2"],
        "gold_derivation": "direct_lookup"
    },

    "base_question": "What is the capital of Germany?",
    "gold_answer": "Berlin",
    "gold_reasoning_chain": ["Germany has a capital city.", "The capital of Germany is Berlin."],
    "support_facts": ["Germany's capital is Berlin."],
    "required_steps": 1,

    "normal_variants": {
        "original": {"question": "...", "metadata": {}},
        "hint": {"question": "...", "metadata": {"hint_content": "..."}},
        "premise": {"question": "...", "metadata": {"added_premise": "..."}},
        "premise_removal": {"question": "...", "metadata": {"removed_premise": "..."}},
        "highlight": {"question": "...", "metadata": {"highlighted_evidence": "..."}},
        "full_support_bundle": {"question": "...", "metadata": {"bundle_facts": ["..."]}},
        "scaffold_1": {"question": "...", "metadata": {"scaffold_steps": ["..."]}},
        "scaffold_2": {"question": "...", "metadata": {"scaffold_steps": ["...", "..."]}},
        "scaffold_3": {"question": "...", "metadata": {"scaffold_steps": ["...", "...", "..."]}},
        "scaffold_shuffled": {"question": "...", "metadata": {"original_order": [...], "shuffled_order": [...]}},
        "cot_full": {"question": "...", "metadata": {"cot_chain": ["..."]}},
        "cot_partial": {"question": "...", "metadata": {"cot_chain": ["..."], "omitted_steps": [...]}},
        "cot_shuffled": {"question": "...", "metadata": {"original_order": [...], "shuffled_order": [...]}},
        "wrongclaim_bare": {"question": "...", "metadata": {"wrong_claim": "..."}},
        "wrongclaim_confident": {"question": "...", "metadata": {"wrong_claim": "...", "confidence_wrapper": "..."}},
        "wrongclaim_attributed": {"question": "...", "metadata": {"wrong_claim": "...", "attribution": "..."}},
        "competing_claims": {"question": "...", "metadata": {"correct_claim": "...", "wrong_claim": "..."}},
        "paraphrase": {"question": "...", "metadata": {}},
        "terminology_swap": {"question": "...", "metadata": {"swap_map": {}}},
        "substitution": {"question": "...", "metadata": {"substitution_map": {}}}
    },

    "symbolic_variants": {
        "entity_map": {"Germany": "∆", "Berlin": "◇", "capital_of": "⊕"},
        "source_family_id": "kb_001",
        "original": {"question": "...", "metadata": {}},
        "premise": {"question": "...", "metadata": {}},
        "scaffold_1": {"question": "...", "metadata": {}},
        "scaffold_2": {"question": "...", "metadata": {}},
        "scaffold_3": {"question": "...", "metadata": {}},
        "wrongclaim_bare": {"question": "...", "metadata": {}},
        "premise_removal": {"question": "...", "metadata": {}},
        "substitution": {"question": "...", "metadata": {}},
        "cot_full": {"question": "...", "metadata": {}},
        "cot_partial": {"question": "...", "metadata": {}},
        "cot_shuffled": {"question": "...", "metadata": {}},
        "highlight": {"question": "...", "metadata": {}},
        "full_support_bundle": {"question": "...", "metadata": {}},
        "scaffold_shuffled": {"question": "...", "metadata": {}},
        "wrongclaim_confident": {"question": "...", "metadata": {}},
        "wrongclaim_attributed": {"question": "...", "metadata": {}},
        "competing_claims": {"question": "...", "metadata": {}},
        "paraphrase": {"question": "...", "metadata": {}},
        "terminology_swap": {"question": "...", "metadata": {}}
    },

    "mcq_variants": {
        "symbolic_original": {
            "question": "...",
            "options": ["◇", "⊗", "▽", "◆"],
            "correct_index": 0,
            "option_metadata": [
                {"role": "gold", "source": "correct_answer"},
                {"role": "same_type", "source": "another_symbolic_entity_of_same_type"},
                {"role": "structurally_related", "source": "related_node_in_structure"},
                {"role": "wrongclaim_aligned", "source": "from_wrongclaim_variant"}
            ]
        }
    },

    "metadata": {
        "block": "KB",
        "generation_model": "claude-sonnet-4-6",
        "generation_timestamp": "2026-04-23T...",
        "pipeline_stage_versions": {
            "structures": "v1",
            "base_items": "v1",
            "variants": "v1",
            "symbolic": "v1",
            "mcq": "v1"
        }
    }
}
```

### underlying_structure by block

**KB**: Graph with nodes (entities) and edges (relations). Typically 1-hop.
```json
{
    "type": "single_hop_binding",
    "nodes": [{"id": "n1", "label": "...", "role": "query_entity"}, {"id": "n2", "label": "...", "role": "answer"}],
    "edges": [{"source": "n1", "target": "n2", "relation": "..."}],
    "support_chain": ["n1 --rel--> n2"],
    "gold_derivation": "direct_lookup"
}
```

**RB**: Rules/expressions with variables and derivation steps.
```json
{
    "type": "algebraic_equation",
    "variables": [{"name": "x", "role": "unknown"}, {"name": "7", "role": "constant"}],
    "rules": ["x + 7 = 21"],
    "derivation_steps": ["x = 21 - 7", "x = 14"],
    "gold_derivation": "algebraic_manipulation"
}
```

**Hybrid**: Combined graph + rules.
```json
{
    "type": "two_hop_composition",
    "nodes": [
        {"id": "n1", "label": "Paris", "role": "query_entity"},
        {"id": "n2", "label": "France", "role": "intermediate"},
        {"id": "n3", "label": "French", "role": "answer"}
    ],
    "edges": [
        {"source": "n1", "target": "n2", "relation": "capital_of"},
        {"source": "n2", "target": "n3", "relation": "official_language"}
    ],
    "support_chain": ["n1 --capital_of--> n2", "n2 --official_language--> n3"],
    "gold_derivation": "multi_hop_lookup"
}
```

## Variant Definitions

### Premise-sensitive

**original**: Standard question, no extra help or interference.

**hint**: Directional help that does NOT complete the support chain. Cannot reveal the answer or key premise directly.

**premise**: Explicitly inject the key support fact/rule. Must directly change evidence availability.

**premise_removal**: Remove the key support, making the question significantly harder or unsolvable. Must not remain trivially solvable after removal.

**highlight**: Extract or highlight the key support location without adding new facts. Like a focused spotlight on existing evidence.

**full_support_bundle**: Provide the complete support chain — all facts/rules needed. For multi-hop items, this means all intermediate facts.

### Scaffold-sensitive

**scaffold_1**: One intermediate decomposition step. Must not give the final answer.

**scaffold_2**: Two intermediate steps. Must not give the final answer.

**scaffold_3**: Three or more intermediate steps. Must not give the final answer.

**scaffold_shuffled**: Same content as scaffold_2 or scaffold_3, but steps in randomized order.

### CoT variants

**cot_full**: Question followed by the complete gold reasoning chain, then asks for the final answer. The chain is presented as explicit step-by-step reasoning.

**cot_partial**: Question followed by a partial reasoning chain (first N-1 steps of an N-step chain). Model must complete the reasoning.

**cot_shuffled**: Question followed by the complete reasoning chain but with steps in randomized order.

### Wrong-claim-susceptible

**wrongclaim_bare**: Insert one incorrect claim/answer cue. Cannot change the gold answer or make the question multi-answer.

**wrongclaim_confident**: Same wrong claim as wrongclaim_bare, wrapped with high-confidence language ("Obviously...", "It is well-established that...", "Clearly...").

**wrongclaim_attributed**: Same wrong claim, attributed to an authority ("According to experts...", "Research has shown...", "The documentation states...").

**competing_claims**: Present both the correct claim and an incorrect claim in the same question. Tests conflict resolution.

### Substitution-fragile

**paraphrase**: Change expression only, not semantics.

**terminology_swap**: Replace common terms with domain-specific terminology. Semantics unchanged.

**substitution**: Replace entities/content words/symbolic names. Underlying relational structure preserved.

## Pipeline Architecture

5-stage pipeline with intermediate JSON checkpoints. Each stage is idempotent and resumable.

```
Stage 1: structures    → checkpoints/01_structures.json
Stage 2: base_items    → checkpoints/02_base_items.json
Stage 3: variants      → checkpoints/03_variants.json
Stage 4: symbolic      → checkpoints/04_symbolic.json
Stage 5: mcq           → checkpoints/05_mcq.json
Export:  merge all      → output/dataset.json, output/dataset.jsonl, output/stats.json
```

### Stage 1: Structure Generation
- Per block (KB/RB/Hybrid), send structured prompt to Claude API
- Request N underlying structures with typed JSON output
- Validate structure completeness (nodes, edges, rules, derivation)

### Stage 2: Base Item Generation
- For each structure, API generates base_question + gold_answer + support_facts + gold_reasoning_chain
- RB items get programmatic gold answer verification where possible

### Stage 3: Variant Generation
- For each base item, API generates all 19 variants in one call
- Prompt includes strict definitions for each variant type
- Validates that gold_answer is preserved across variants (except premise_removal)

### Stage 4: Symbolic Counterpart
- Programmatic: assign Unicode symbols from pool based on structure node order
- Symbol pool: `∆ ◇ ⊕ Ω ★ ⊗ ▽ ◆ ⊙ ⊞ ⊘ ⊛ ⊜ ⊝`
- Text replacement in all variant question strings
- SymbolicControl families record `source_family_id`
- All 19 variants get symbolic versions

### Stage 5: MCQ Generation
- Symbolic families priority, normal families optional
- 4-choice: 1 gold + 3 distractors
- Distractor types: same-type, structurally-related, wrongclaim-aligned
- API generates distractors given the structure and entity map

### Export
- Merge all checkpoints into complete family objects
- Write `dataset.json` (full dataset) and `dataset.jsonl` (one family per line)
- Generate `stats.json`: counts by block, sub_family, variant coverage, symbolic coverage, MCQ coverage

## Code Structure

```
dataset_synthesis/
  __init__.py
  schema.py              # Dataclasses: Family, Variant, Structure, MCQItem
  structures.py          # Structure generation logic + API prompts
  builders/
    __init__.py
    kb.py                # KB structure templates + sub_family definitions
    rb.py                # RB structure templates
    hybrid.py            # Hybrid structure templates
  variants.py            # 19 variant generators + API prompts
  symbolic.py            # Unicode symbol pool + programmatic replacement
  mcq.py                 # MCQ distractor generation
  pipeline.py            # 5-stage orchestrator with checkpoint management
  api_client.py          # Anthropic API wrapper (retry, backoff, JSON parse)
  export.py              # JSON/JSONL export + merge logic
  stats.py               # Statistics report generation
  configs/
    __init__.py
    defaults.py          # Default params (counts, model, symbol pool, etc.)
  README.md
```

## API Strategy

- Model: `claude-sonnet-4-6` (configurable)
- All prompts enforce JSON output
- Automatic retry with exponential backoff (3 retries, 1s/2s/4s)
- Rate limiting: configurable delay between calls
- Each stage writes checkpoint after each successful family/batch
- Resume: skip family_ids already present in checkpoint

## KB Sub-families

- `country_capital`: country → capital
- `work_author`: literary/artistic work → author
- `element_symbol`: chemical element → symbol
- `entity_attribute`: general factual bindings (person→birthyear, river→continent, etc.)

## RB Sub-families

- `linear_equation`: solve for x in ax + b = c
- `syllogistic_logic`: if-all-then transitive reasoning
- `sequence_pattern`: find nth term given a rule
- `boolean_logic`: evaluate logical expressions
- `function_application`: apply f(x) = expression, compute f(n)

## Hybrid Sub-families

- `two_hop_relational`: A→B→C lookup chain
- `retrieve_transform`: retrieve a fact, apply arithmetic/logic
- `retrieve_rule_apply`: retrieve a fact, apply an explicit rule

## Constraints

- Structure-first: underlying_structure must exist before any surface form
- Gold answer uniqueness: one correct answer per family
- Variant fidelity: gold_answer preserved across all variants except premise_removal
- Symbolic isomorphism: symbolic families must preserve support_chain, answer role, distractor role
- No execution, testing, or training code
