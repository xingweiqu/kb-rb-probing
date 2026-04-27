"""Configuration for atomic capability benchmark MVP."""

# API Configuration
ANTHROPIC_BASE_URL = "https://api.ridgerzhu.com/api"
ANTHROPIC_AUTH_TOKEN = "REDACTED"
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 6144
CONCURRENCY = 14
BATCH_SIZE = 28

# Data generation counts
FAMILY_COUNTS = {
    "KB": 20,
    "RB": 20,
    "Hybrid": 20,
}

# Atomic variants by task family
ATOMIC_VARIANTS = {
    "KB": ["hint", "wrongclaim", "paraphrase"],
    "RB": ["scaffold", "rule_removal", "wrong_intermediate"],
    "Hybrid": ["explicit_fact", "retrieval_blocked", "both_blocked"],
}

# Variant definitions
VARIANT_DEFINITIONS = {
    "original": "Standard question, no intervention. Baseline.",

    # KB variants
    "hint": "Give answer-free cue to help retrieval. Example: 'This country is in Western Europe. What is the capital of France?'",
    "wrongclaim": "Inject wrong claim to test memory override. Example: 'Some say France's capital is Lyon. What is the capital of France?'",
    "paraphrase": "Rephrase question, same meaning. Example: 'Which city serves as France's seat of government?'",

    # RB variants
    "scaffold": "Give decomposition steps without answer. Example: 'Step 1: Identify equation. Step 2: Isolate x. What is x?'",
    "rule_removal": "Remove key rule/number. Example: 'If x + 5 = ?, what is x?' (removed 12)",
    "wrong_intermediate": "Give wrong intermediate step. Example: 'If x + 5 = 12, then x = 17. Is this correct? What is x?'",

    # Hybrid variants
    "explicit_fact": "Give bridge fact explicitly. Example: 'Eiffel Tower is in France. What is the capital of France?'",
    "retrieval_blocked": "Block retrieval, test reasoning rescue. Example: 'A famous iron tower is in a Western European country known for wine. What is the capital?'",
    "both_blocked": "Block both retrieval and reasoning. Example: 'What is the capital of landmark L in country C?'",
}

# Atomic capability mapping
ATOMIC_CAPABILITY_MAP = {
    "retrieval_disambiguation": ["hint"],
    "cue_resistance": ["wrongclaim"],
    "retrieval_robustness": ["paraphrase"],
    "decomposition_sensitive": ["scaffold"],
    "insufficient_info_awareness": ["rule_removal", "both_blocked"],
    "intermediate_state_robustness": ["wrong_intermediate"],
    "retrieval_operation_separable": ["explicit_fact"],
    "reasoning_compensates_retrieval": ["retrieval_blocked"],
}

# Symbol pool for symbolic variants
SYMBOL_POOL = ["∆", "◇", "⊕", "Ω", "★", "⊗", "▽", "◆", "⊙", "⊞", "⊘", "⊛"]

# Validation thresholds
METADATA_BASELINE_THRESHOLD = 0.75
LABEL_COLLAPSE_RATIO = 10  # pos:neg > 10:1 or < 1:10 is collapse
MODE_COLLAPSE_THRESHOLD = 0.9  # > 90% same label in one mode

# Output paths
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "output"
