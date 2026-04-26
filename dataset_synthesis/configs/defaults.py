"""Default configuration for the synthesis pipeline."""

MODEL = "gpt-oss-120b"
# vLLM `max_tokens` for chat completions. Keep below the server's max_model_len budget.
MAX_TOKENS = 6144

# Family counts per block
FAMILY_COUNTS = {
    "KB": 25,
    "RB": 25,
    "Hybrid": 25,
}
SYMBOLIC_CONTROL_COUNT = 15

# API retry
MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]
# For local multi-endpoint vLLM, keep this at 0 unless you see queue buildup.
REQUEST_DELAY = 0.0  # seconds between API calls

# Async throughput controls
CONCURRENCY = 14
BATCH_SIZE = 28

# Unicode symbol pool — assigned in order to structure nodes
SYMBOL_POOL = [
    "∆", "◇", "⊕", "Ω", "★", "⊗", "▽", "◆",
    "⊙", "⊞", "⊘", "⊛", "⊜", "⊝", "⊟", "⊠",
]

# All 19 variant types
VARIANT_TYPES = [
    # Premise-sensitive
    "original",
    "hint",
    "premise",
    "premise_removal",
    "highlight",
    "full_support_bundle",
    # Scaffold-sensitive
    "scaffold_1",
    "scaffold_2",
    "scaffold_3",
    "scaffold_shuffled",
    # CoT
    "cot_full",
    "cot_partial",
    "cot_shuffled",
    # Wrong-claim-susceptible
    "wrongclaim_bare",
    "wrongclaim_confident",
    "wrongclaim_attributed",
    "competing_claims",
    # Substitution-fragile
    "paraphrase",
    "terminology_swap",
    "substitution",
]

# Symbolic variants — all 19 get symbolic versions
SYMBOLIC_VARIANT_TYPES = VARIANT_TYPES

# Atomic capability → variant mapping (for stats / validation)
# Names are aligned to the intended atomic capability taxonomy.
ATOMIC_CAPABILITY_MAP = {
    "access_sensitive": ["original", "hint", "premise", "premise_removal"],
    "localization_sensitive": ["highlight"],
    "integration_sensitive": ["full_support_bundle"],
    "decomposition_sensitive": ["scaffold_1", "scaffold_2", "scaffold_3"],
    "order_sensitive": ["scaffold_shuffled"],
    "intermediate_state_sensitive": ["cot_full", "cot_partial"],
    "cue_susceptible": ["wrongclaim_bare"],
    "authority_susceptible": ["wrongclaim_confident", "wrongclaim_attributed"],
    "conflict_resolution_weak": ["competing_claims"],
    "lexical_fragile": ["paraphrase"],
    "terminology_fragile": ["terminology_swap"],
    "structure_misaligned": ["substitution"],
}

# CoT shuffled is cross-cutting
COT_SHUFFLED_NOTE = "cot_shuffled tests order-sensitivity within CoT chains"

# KB sub-families
KB_SUB_FAMILIES = [
    "country_capital",
    "work_author",
    "element_symbol",
    "entity_attribute",
]

# RB sub-families
RB_SUB_FAMILIES = [
    "linear_equation",
    "syllogistic_logic",
    "sequence_pattern",
    "boolean_logic",
    "function_application",
]

# Hybrid sub-families
HYBRID_SUB_FAMILIES = [
    "two_hop_relational",
    "retrieve_transform",
    "retrieve_rule_apply",
]

# Checkpoint directory (relative to output_dir)
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "output"
