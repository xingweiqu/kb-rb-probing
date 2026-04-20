"""
Generate item families for KB/RB/Hybrid probing benchmark.
Uses Claude API to generate 30 item families (10 per category),
each with 9 controlled variants.

Usage:
  export ANTHROPIC_API_KEY=your_key
  python generate_items.py
"""

import json
import os
import anthropic

VARIANTS = [
    "original",
    "paraphrase",
    "counterfactual",
    "hint",
    "premise",
    "context_scaffolding",
    "premise_removal",
    "structure_substitution",
    "symbol_substitution",
]

SYSTEM_PROMPT = """You are a benchmark designer for NLP research. Your task is to generate controlled item families to study whether language models use knowledge retrieval or active reasoning.

Each item family must have:
- A single short gold_answer (1-5 words)
- 9 variants of the same underlying question

Variant definitions:
- original: standard question form
- paraphrase: same meaning, different surface wording
- counterfactual: inject a wrong claim as distractor inside the question (e.g. "Given that X is Y, what is actually X?")
- hint: add a useful clue that partially guides toward the answer without giving it away
- premise: explicitly state the key supporting premise before asking
- context_scaffolding: add 2-3 intermediate reasoning steps before the final question
- premise_removal: remove the key supporting context so the question becomes underspecified
- structure_substitution: replace content tokens with different but structurally equivalent ones (same answer type, different entities/numbers)
- symbol_substitution: replace key entities/values with symbols (∆, ◇, ⊕, etc.) and provide the mapping explicitly at the start (e.g. "In this world, France = ∆, Paris = ◇. What is the capital of ∆?")

Return ONLY a JSON array of item family objects. No explanation, no markdown, just raw JSON."""

KB_PROMPT = """Generate 10 KB (knowledge-based) item families. These should test factual/associative knowledge stored in model parameters: world facts, entity-attribute bindings, historical facts, scientific facts.

Each item must have:
- id: "kb_001" through "kb_010"
- category: "KB"
- gold_answer: short string (1-5 words)
- variants: object with all 9 variant keys

Make the items diverse: mix geography, science, history, culture. Avoid trivial or ambiguous facts.

Return a JSON array of 10 objects."""

RB_PROMPT = """Generate 10 RB (rule-based reasoning) item families. These should test explicit structural inference: arithmetic, algebra, formal logic, pattern completion, rule application. The answer must be derivable from the given structure alone, without world knowledge.

Each item must have:
- id: "rb_001" through "rb_010"
- category: "RB"
- gold_answer: short string (1-5 words, typically a number or logical value)
- variants: object with all 9 variant keys

For structure_substitution: change the numbers/variables but keep the same operation type.
For symbol_substitution: replace variables/numbers with symbols and provide the mapping.
For premise_removal: remove the equation or rule, leaving only "What is x?" or similar.

Make items diverse: mix arithmetic, algebra, logic, sequence patterns.

Return a JSON array of 10 objects."""

HYBRID_PROMPT = """Generate 10 Hybrid item families. These require BOTH stored knowledge AND active reasoning/composition: multi-hop fact chains, knowledge-supported word problems, causal reasoning with domain knowledge, analogical reasoning.

Each item must have:
- id: "hybrid_001" through "hybrid_010"
- category: "Hybrid"
- gold_answer: short string (1-5 words)
- variants: object with all 9 variant keys

Examples of hybrid tasks:
- Multi-hop: "What language is spoken in the country whose capital is Paris?" (needs France→Paris→French)
- Knowledge word problem: "A train leaves the capital of France at 9am..."
- Causal with knowledge: "Why does ice float on water?"

Return a JSON array of 10 objects."""


def generate_category(client: anthropic.Anthropic, prompt: str, category: str) -> list:
    print(f"Generating {category} items...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    items = json.loads(text)
    print(f"  Got {len(items)} {category} items")
    return items


def validate_item(item: dict) -> bool:
    required_keys = {"id", "category", "gold_answer", "variants"}
    if not required_keys.issubset(item.keys()):
        return False
    if not all(v in item["variants"] for v in VARIANTS):
        return False
    return True


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    all_items = []

    for prompt, category in [
        (KB_PROMPT, "KB"),
        (RB_PROMPT, "RB"),
        (HYBRID_PROMPT, "Hybrid"),
    ]:
        items = generate_category(client, prompt, category)
        valid = [item for item in items if validate_item(item)]
        if len(valid) < len(items):
            print(f"  Warning: {len(items) - len(valid)} items failed validation")
        all_items.extend(valid)

    output_path = os.path.join(os.path.dirname(__file__), "items.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_items)} item families to {output_path}")

    # Print a sample
    print("\nSample item:")
    print(json.dumps(all_items[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
