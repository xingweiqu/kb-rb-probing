"""Quick smoke test for core functionality."""

import sys
sys.path.insert(0, 'dataset_synthesis_mvp')

from validation.leakage import LeakageValidator
from validation.surface import SurfaceValidator
from repair.post_process import PostProcessor

print("Testing validators...")

# Test data
family = {
    "family_id": "kb_001",
    "task_family": "KB",
    "base_question": "What is the capital of France?",
    "gold_answer": "Paris",
    "underlying_structure": {
        "nodes": [
            {"label": "France", "role": "query_entity"},
            {"label": "Paris", "role": "answer"}
        ]
    }
}

# Test 1: Direct leak detection
print("\n1. Testing direct leak detection...")
item_with_leak = {
    "family_id": "kb_001",
    "variant": "wrongclaim",
    "question": "Some say the capital is Paris. What is the capital of France?",
    "metadata": {}
}

leakage_validator = LeakageValidator()
issues = leakage_validator.validate(item_with_leak, family)
assert len(issues) > 0, "Should detect direct leak"
assert issues[0]["code"] == "direct_answer_leak"
print("✓ Direct leak detected")

# Test 2: No leak
print("\n2. Testing clean question...")
item_clean = {
    "family_id": "kb_001",
    "variant": "wrongclaim",
    "question": "Some say the capital is Lyon. What is the capital of France?",
    "metadata": {"wrong_claim": "Lyon"}
}

issues = leakage_validator.validate(item_clean, family)
leak_issues = [i for i in issues if "leak" in i["code"]]
assert len(leak_issues) == 0, "Should not detect leak in clean question"
print("✓ Clean question passed")

# Test 3: Type mismatch detection
print("\n3. Testing type mismatch detection...")
item_type_mismatch = {
    "family_id": "kb_001",
    "variant": "wrongclaim",
    "question": "What is the capital of France?",
    "metadata": {"wrong_claim": "42"}  # Number instead of entity
}

surface_validator = SurfaceValidator()
issues = surface_validator.validate(item_type_mismatch, family)
assert len(issues) > 0, "Should detect type mismatch"
assert issues[0]["code"] == "type_mismatch"
print("✓ Type mismatch detected")

# Test 4: Post-processor repair
print("\n4. Testing post-processor...")
post_processor = PostProcessor()
item_with_leak_copy = {**item_with_leak}
repaired = post_processor.process_item(
    item_with_leak_copy,
    family,
    [{"code": "direct_answer_leak"}]
)
assert "Paris" not in repaired["question"], "Should mask answer"
assert "[MASK]" in repaired["question"], "Should add mask"
assert repaired["metadata"]["auto_repaired"] == True
print("✓ Post-processor repaired leak")

# Test 5: Symbolic entity extraction
print("\n5. Testing symbolic generation...")
from dataset_synthesis_mvp.symbolic import _extract_entities, build_entity_map

entities = _extract_entities(family["underlying_structure"], family)
assert "France" in entities, "Should extract France"
assert "Paris" in entities, "Should extract Paris (gold_answer)"
print(f"✓ Extracted entities: {entities}")

entity_map = build_entity_map(family["underlying_structure"], family)
assert "France" in entity_map, "Should map France"
assert "Paris" in entity_map, "Should map Paris"
print(f"✓ Built entity map: {entity_map}")

print("\n" + "="*60)
print("All smoke tests passed! ✓")
print("="*60)
