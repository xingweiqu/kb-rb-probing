# Atomic Capability Taxonomy

This document defines the capability taxonomy used by the probing pipeline. It supersedes the prior per-variant capability list in `probing_mvp/derive_labels.py`.

## Motivation

Earlier capability labels were named after the variants that produced them (`retrieval_disambiguation`, `decomposition_sensitive`, …). That coupling has two problems:

1. **Operations are confused with capabilities.** "Hint injection" is something we do to the prompt; "knowledge surfacing under hint" is the model property we are measuring.
2. **The set is not justified as atomic.** Why nine? Why not seven? A reviewer cannot tell whether the list is a complete decomposition of capability space or an arbitrary sample.

The new taxonomy decouples mechanism (what cognitive process the task draws on) from perturbation (what we do to the input), and adds Chain-of-Thought as an orthogonal probing condition.

## The 2×3 grid

|              | Knowledge                | Reasoning                | Composition                       |
|--------------|--------------------------|--------------------------|-----------------------------------|
| **Provide**  | KP — knowledge surfacing | RP — rule activation     | CP — gap-filling                  |
| **Block**    | KB — retrieval dependence| RB — rule dependence     | CB — bottleneck identification    |
| **Distract** | KD — retrieval robustness| RD — reasoning robustness| CD — cross-axis interference *    |

\* CD is currently unoperationalized; reserved for future hybrid-distract variants.

### Mechanism axis

- **Knowledge** — fact retrieval; expected to live in MLP layers (Geva 2021).
- **Reasoning** — rule application and multi-step composition; expected to involve attention circuits (Olsson 2022 induction heads).
- **Composition** — both retrieval and reasoning are required; failure of either causes failure.

### Perturbation axis

- **Provide** — inject information that should help the model. Capability is positive when the model was wrong without injection but correct with it.
- **Block** — remove information the model needs. Capability is positive when the model was correct originally but fails after blocking.
- **Distract** — inject misleading information. Capability is positive when the model was correct originally but fails when distracted.

The two axes are orthogonal by construction. A reviewer asking "why is this set atomic?" can be answered with "mechanism × perturbation gives a complete 2D coordinate system; we sample one capability per cell."

## Variant → capability mapping

The current 9 variants populate 8 of 9 cells:

| Variant                 | Cell        | Judgment                                           |
|-------------------------|-------------|----------------------------------------------------|
| KB-hint                 | KP          | ¬orig_correct ∧ variant_correct → KP=True          |
| KB-paraphrase           | KB          | orig_correct ∧ ¬variant_correct → KB=True          |
| KB-wrongclaim           | KD          | orig_correct ∧ ¬variant_correct → KD=True          |
| RB-scaffold             | RP          | ¬orig_correct ∧ variant_correct → RP=True          |
| RB-rule_removal         | RB          | orig_correct ∧ ¬variant_correct → RB=True          |
| RB-wrong_intermediate   | RD          | orig_correct ∧ ¬variant_correct → RD=True          |
| Hybrid-explicit_fact    | CP          | ¬orig_correct ∧ variant_correct → CP=True          |
| Hybrid-retrieval_blocked| CB          | orig_correct ∧ ¬variant_correct → CB=True          |
| Hybrid-both_blocked     | CB-control  | Lower-bound control; expected double-zero accuracy |

### Resolving the `explicit_fact ≈ hint` boundary concern

Both `hint` (KB) and `explicit_fact` (Hybrid) inject information and use the same Provide-axis judgment. In the old taxonomy this looked like a labeling collision. In the 2×3 grid it is a **prediction**: KP and CP differ along the mechanism axis but share the Provide axis, so they should produce different hidden-state geometry despite using identical operational templates. Hidden-state geometry analysis (Phase D) is the empirical test of this prediction.

## CoT axis (third dimension)

We add Chain-of-Thought as an orthogonal **probing condition**, not a capability.

- **no-CoT** — question fed to the model directly.
- **with-CoT** — question is prefixed with `"Let's think step by step.\n"` (Kojima 2022 zero-shot CoT) before being fed to the model.

CoT is applied at inference / probing time. The dataset itself is generated only once (CoT does not enter the data synthesis pipeline). Each (family, variant) item is then probed twice and yields two hidden states.

### Why CoT-on-Knowledge is not "weird"

A reviewer may ask why we even apply CoT to KB tasks, given Sprague et al. 2024 found CoT yields little benefit on pure retrieval. The answer is that we are not testing whether CoT helps — we are using CoT as a **differential probe**:

- If KB hidden-state geometry is unchanged by CoT, this confirms KB capabilities are retrieval-style (no intermediate computation to extend).
- If RB / Hybrid geometry is reorganized by CoT, this confirms reasoning capabilities involve composable intermediate states.
- The **differential response pattern** across mechanism axes is itself evidence that the taxonomy carves the model at real joints.

This reframes a potential weakness ("KB+CoT looks pointless") into the taxonomy's central validation experiment.

## Output schema

`probing_mvp/derive_labels.py` writes a JSON file shaped as:

```json
{
  "<family_id>": {
    "no_cot":   { "KP": true, "KD": false, "KB": null, "RP": ..., "CP": ..., "CB": ..., "CB-control": ... },
    "with_cot": { "KP": true, "KD": null,  "KB": false, ... }
  },
  ...
}
```

Each capability cell is `True` (capability evident), `False` (capability not evident), or `null` (judgment uninformative — typically because the original item was already correct/incorrect in a way that the variant cannot disambiguate under that axis).

## What this taxonomy lets the paper claim

1. **Taxonomy (theory)** — Capability space decomposes orthogonally into mechanism × perturbation; 6 atomic capabilities + 1 control + 1 CoT axis.
2. **Probing (correlation)** — Each capability is linearly decodable from hidden states; geometry shows expected structural relationships (e.g., CP between KP and RP).
3. **Causal alignment (mechanism)** — Layer-wise activation profiles of capability decoders match mechanism-specific ablation effects (MLP for K, attention for R), ruling out superficial encoding.
4. **CoT modulation** — The differential response of capability geometry to CoT is mechanism-specific (strong for R, weak for K), supporting taxonomy validity at a second axis.
