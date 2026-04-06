"""
Three approaches to distinguish Knowledge vs Reasoning in LLMs.
Run with base model (no --instruct) as default.

Usage:
  python probe.py --model_name /opt/tiger/Flame/Qwen3-8B-Base
  python probe.py --model_name /opt/tiger/Flame/Qwen3-8B --instruct
"""

import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Data ──────────────────────────────────────────────────────────────────────

# Method 1: paraphrase pairs (original, paraphrase)
# Knowledge: world facts — same info, different word order
# Reasoning: large-number arithmetic/logic — same computation, different phrasing
KNOWLEDGE_PAIRS = [
    ("The capital of France is Paris.", "Paris is the capital city of France."),
    ("Shakespeare wrote Hamlet.", "Hamlet was written by Shakespeare."),
    ("Water boils at 100 degrees Celsius.", "The boiling point of water is 100 degrees Celsius."),
    ("The Earth orbits the Sun.", "The Sun is orbited by the Earth."),
    ("The Great Wall of China is in Asia.", "Asia is home to the Great Wall of China."),
    ("Albert Einstein developed the theory of relativity.", "The theory of relativity was developed by Albert Einstein."),
    ("The human heart has four chambers.", "Four chambers are found in the human heart."),
    ("Oxygen has atomic number 8.", "The atomic number of oxygen is 8."),
    ("Napoleon was exiled to Saint Helena.", "Saint Helena was where Napoleon was exiled."),
    ("The Nile is the longest river in Africa.", "Africa's longest river is the Nile."),
    ("Photosynthesis produces oxygen.", "Oxygen is produced by photosynthesis."),
    ("The speed of light is approximately 300000 km/s.", "Approximately 300000 km/s is the speed of light."),
    ("The French Revolution began in 1789.", "1789 was the year the French Revolution began."),
    ("Humans have 46 chromosomes.", "46 chromosomes are found in human cells."),
    ("The Pacific Ocean is the largest ocean.", "The largest ocean on Earth is the Pacific Ocean."),
    ("Gold has the chemical symbol Au.", "The chemical symbol for gold is Au."),
    ("The moon orbits the Earth.", "The Earth is orbited by the moon."),
    ("Rome is the capital of Italy.", "Italy's capital city is Rome."),
    ("The mitochondria is the powerhouse of the cell.", "The powerhouse of the cell is the mitochondria."),
    ("Isaac Newton formulated the laws of motion.", "The laws of motion were formulated by Isaac Newton."),
]

REASONING_PAIRS = [
    ("73847 multiplied by 29 equals 2141563.", "29 times 73847 is 2141563."),
    ("The product of 8317 and 46 is 382582.", "8317 times 46 equals 382582."),
    ("987654 minus 398271 equals 589383.", "398271 subtracted from 987654 gives 589383."),
    ("The sum of 46829 and 73541 is 120370.", "46829 plus 73541 equals 120370."),
    ("A factory produces 2347 units per day. In 13 days it produces 30511 units.", "30511 units are produced in 13 days at 2347 units per day."),
    ("A rectangle with sides 127 and 83 has area 10541 square units.", "10541 is the area of a rectangle with sides 127 and 83."),
    ("If a car travels 347 km at 53 km/h, the journey takes about 6.5 hours.", "At 53 km/h, covering 347 km takes about 6.5 hours."),
    ("A tank holds 8450 liters. After removing 3127 liters, 5323 liters remain.", "5323 liters remain after removing 3127 from 8450 liters."),
    ("If 47x equals 3901, then x equals 83.", "x is 83 because 47 times 83 equals 3901."),
    ("If x squared equals 7569, then x equals 87.", "87 squared is 7569, so x equals 87."),
    ("All mammals are warm-blooded. Whales are mammals. Therefore whales are warm-blooded.", "Whales are warm-blooded because all mammals are warm-blooded and whales are mammals."),
    ("If A implies B and B implies C, then A implies C.", "The law of transitivity states that if A implies B and B implies C then A implies C."),
    ("63419 divided by 7 equals 9059 remainder 6.", "7 goes into 63419 exactly 9059 times with remainder 6."),
    ("The perimeter of a rectangle with sides 94 and 137 is 462.", "462 is the perimeter of a rectangle with sides 94 and 137."),
    ("If train A leaves at 9am at 80 km/h and train B at 10am at 100 km/h, they meet after 4 hours.", "The two trains meet 4 hours after train A departs."),
    ("2 to the power of 17 equals 131072.", "131072 is 2 raised to the 17th power."),
    ("The least common multiple of 12 and 18 is 36.", "36 is the LCM of 12 and 18."),
    ("If p and q are both true, then p AND q is true.", "p AND q is true when both p and q are true."),
    ("A sphere with radius 5 has volume approximately 523.6 cubic units.", "Approximately 523.6 cubic units is the volume of a sphere with radius 5."),
    ("The greatest common divisor of 84 and 56 is 28.", "28 is the GCD of 84 and 56."),
]

# Method 2: (question, answer) pairs — base model sees "Q: ... A: answer"
# Knowledge: facts the model almost certainly memorized
# Reasoning: computations requiring actual calculation
KNOWLEDGE_QA = [
    ("Q: What is the capital of France? A:", "Paris"),
    ("Q: Who wrote Hamlet? A:", "Shakespeare"),
    ("Q: What is the boiling point of water? A:", "100 degrees Celsius"),
    ("Q: What is the atomic number of oxygen? A:", "8"),
    ("Q: Where was Napoleon exiled? A:", "Saint Helena"),
    ("Q: What is the longest river in Africa? A:", "The Nile"),
    ("Q: How many chambers does the human heart have? A:", "Four"),
    ("Q: Who developed the theory of relativity? A:", "Albert Einstein"),
    ("Q: What does photosynthesis produce? A:", "Oxygen"),
    ("Q: What is the chemical symbol for gold? A:", "Au"),
    ("Q: What country is the Great Wall of China in? A:", "China"),
    ("Q: What is the largest ocean? A:", "The Pacific Ocean"),
    ("Q: How many chromosomes do humans have? A:", "46"),
    ("Q: What is the capital of Italy? A:", "Rome"),
    ("Q: What is the powerhouse of the cell? A:", "The mitochondria"),
    ("Q: Who formulated the laws of motion? A:", "Isaac Newton"),
    ("Q: What year did the French Revolution begin? A:", "1789"),
    ("Q: What orbits the Earth? A:", "The moon"),
    ("Q: What is the chemical formula for water? A:", "H2O"),
    ("Q: What planet is closest to the Sun? A:", "Mercury"),
    ("Q: What is the square root of 144? A:", "12"),
    ("Q: How many sides does a hexagon have? A:", "Six"),
    ("Q: What is the speed of sound in air? A:", "343 meters per second"),
    ("Q: Who painted the Mona Lisa? A:", "Leonardo da Vinci"),
    ("Q: What is the hardest natural substance? A:", "Diamond"),
    ("Q: How many bones are in the adult human body? A:", "206"),
    ("Q: What gas do plants absorb during photosynthesis? A:", "Carbon dioxide"),
    ("Q: What is the currency of Japan? A:", "The yen"),
    ("Q: What is the tallest mountain in the world? A:", "Mount Everest"),
    ("Q: What is the smallest planet in the solar system? A:", "Mercury"),
]

REASONING_QA = [
    ("Q: What is 73847 times 29? A:", "2141563"),
    ("Q: What is 8317 times 46? A:", "382582"),
    ("Q: What is 987654 minus 398271? A:", "589383"),
    ("Q: What is 46829 plus 73541? A:", "120370"),
    ("Q: A factory makes 2347 units per day. How many in 13 days? A:", "30511"),
    ("Q: What is the area of a rectangle with sides 127 and 83? A:", "10541"),
    ("Q: If 47x equals 3901, what is x? A:", "83"),
    ("Q: What is the square root of 7569? A:", "87"),
    ("Q: A car travels 347 km at 53 km/h. How many hours? A:", "6.5"),
    ("Q: A tank has 8450 liters. Remove 3127. How many remain? A:", "5323"),
    ("Q: What is 63419 divided by 7? A:", "9059"),
    ("Q: What is the perimeter of a rectangle with sides 94 and 137? A:", "462"),
    ("Q: What is 2 to the power of 17? A:", "131072"),
    ("Q: What is the LCM of 12 and 18? A:", "36"),
    ("Q: What is the GCD of 84 and 56? A:", "28"),
    ("Q: A sphere has radius 5. What is its volume to one decimal? A:", "523.6"),
    ("Q: What is 9371 times 83? A:", "777793"),
    ("Q: What is 234567 minus 98765? A:", "135802"),
    ("Q: If 13x equals 1547, what is x? A:", "119"),
    ("Q: What is the area of a triangle with base 94 and height 67? A:", "3149"),
    ("Q: What is 17 squared plus 13 squared? A:", "458"),
    ("Q: A train goes 120 km/h for 2.5 hours. How far? A:", "300"),
    ("Q: What is 15 percent of 4680? A:", "702"),
    ("Q: What is the sum of angles in a pentagon? A:", "540"),
    ("Q: If x plus 47 equals 193, what is x? A:", "146"),
    ("Q: What is 3 to the power of 8? A:", "6561"),
    ("Q: A rectangle has area 2491 and one side 47. What is the other side? A:", "53"),
    ("Q: What is 456 times 789? A:", "359784"),
    ("Q: What is the LCM of 15 and 25? A:", "75"),
    ("Q: If 7x minus 3 equals 179, what is x? A:", "26"),
]

# Method 3 uses KNOWLEDGE_QA / REASONING_QA with answer logprob as metric

# ── Symbolic data (排除记忆污染，用于论文实验) ────────────────────────────────
# KB: 随机符号映射的"事实"，模型不可能背过
# RB: 形式逻辑推理链，必须推导
SYMBOLIC_KNOWLEDGE_PAIRS = [
    ("∆ is the capital of ◇.", "◇'s capital is ∆."),
    ("The ⊕ of ⊗ is ⊙.", "⊙ is the ⊕ of ⊗."),
    ("Ψ belongs to the category of Φ.", "Φ contains Ψ."),
    ("The ℜ of ℵ equals ℶ.", "ℶ is the ℜ of ℵ."),
    ("⟁ was created by ⟂.", "⟂ created ⟁."),
    ("The property of ⊛ is ⊜.", "⊜ is a property of ⊛."),
    ("⋈ is located inside ⋉.", "⋉ contains ⋈."),
    ("The role of ⊞ is ⊟.", "⊟ is the role of ⊞."),
    ("⊠ is a type of ⊡.", "⊡ includes ⊠."),
    ("The value of ⊢ is ⊣.", "⊣ is the value of ⊢."),
    ("⊤ is associated with ⊥.", "⊥ is associated with ⊤."),
    ("The ⊦ of ⊧ is ⊨.", "⊨ is the ⊦ of ⊧."),
    ("⊩ belongs to ⊪.", "⊪ owns ⊩."),
    ("The ⊫ of ⊬ is ⊭.", "⊭ is the ⊫ of ⊬."),
    ("⊮ is part of ⊯.", "⊯ includes ⊮."),
    ("The ⊰ of ⊱ is ⊲.", "⊲ is the ⊰ of ⊱."),
    ("⊳ is related to ⊴.", "⊴ is related to ⊳."),
    ("The ⊵ of ⊶ is ⊷.", "⊷ is the ⊵ of ⊶."),
    ("⊸ is a member of ⊹.", "⊹ contains ⊸."),
    ("The ⊺ of ⊻ is ⊼.", "⊼ is the ⊺ of ⊻."),
]

SYMBOLIC_REASONING_PAIRS = [
    ("If ∆→◇ and ◇→□, then ∆→□.", "∆→□ follows from ∆→◇ and ◇→□."),
    ("If all ⊕ are ⊗ and all ⊗ are ⊙, then all ⊕ are ⊙.", "All ⊕ are ⊙ because all ⊕ are ⊗ and all ⊗ are ⊙."),
    ("If Ψ>Φ and Φ>Ω, then Ψ>Ω.", "Ψ>Ω follows from Ψ>Φ and Φ>Ω."),
    ("If ℜ=ℵ+ℶ and ℵ=ℷ, then ℜ=ℷ+ℶ.", "ℜ=ℷ+ℶ by substituting ℵ=ℷ into ℜ=ℵ+ℶ."),
    ("If ⟁ implies ⟂ and ⟂ implies ⟃, then ⟁ implies ⟃.", "⟁ implies ⟃ by transitivity."),
    ("If ⊛ is not ⊜ and ⊜ is not ⊝, then ⊛ may or may not be ⊝.", "The relation between ⊛ and ⊝ is undetermined."),
    ("If ⋈⊂⋉ and ⋉⊂⋊, then ⋈⊂⋊.", "⋈⊂⋊ follows from ⋈⊂⋉ and ⋉⊂⋊."),
    ("If ⊞=2×⊟ and ⊟=3×⊠, then ⊞=6×⊠.", "⊞=6×⊠ by substituting ⊟=3×⊠."),
    ("If ⊡ is true and ⊢ is false, then ⊡ AND ⊢ is false.", "⊡ AND ⊢ is false because ⊢ is false."),
    ("If ⊣ OR ⊤ is true and ⊣ is false, then ⊤ is true.", "⊤ must be true given ⊣ OR ⊤ and ⊣ is false."),
    ("If ⊥=⊦+⊧ and ⊦=⊨-⊩, then ⊥=⊨-⊩+⊧.", "⊥=⊨-⊩+⊧ by substituting ⊦=⊨-⊩."),
    ("If all ⊪ are ⊫ and no ⊬ are ⊫, then no ⊬ are ⊪.", "No ⊬ are ⊪ by contrapositive reasoning."),
    ("If ⊭>⊮ and ⊮>⊯, and ⊯>⊰, then ⊭>⊰.", "⊭>⊰ by chaining three inequalities."),
    ("If ⊱=⊲×⊳ and ⊲=⊴/⊵, then ⊱=⊴×⊳/⊵.", "⊱=⊴×⊳/⊵ by substituting ⊲=⊴/⊵."),
    ("If ⊶ implies ⊷ and NOT ⊷, then NOT ⊶.", "NOT ⊶ follows by modus tollens."),
    ("If ⊸∈⊹ and ⊹∈⊺, then ⊸∈⊺ if ∈ is transitive.", "⊸∈⊺ by transitivity of ∈."),
    ("If ⊻=⊼+⊽ and ⊼=⊾-⊿, then ⊻=⊾-⊿+⊽.", "⊻=⊾-⊿+⊽ by substitution."),
    ("If ⋀ is necessary for ⋁ and ⋁ is necessary for ⋂, then ⋀ is necessary for ⋂.", "⋀ is necessary for ⋂ by transitivity."),
    ("If ⋃=⋄×⋅ and ⋅=⋆+⋇, then ⋃=⋄×(⋆+⋇).", "⋃=⋄×(⋆+⋇) by substituting ⋅=⋆+⋇."),
    ("If ⋈ is sufficient for ⋉ and ⋉ is sufficient for ⋊, then ⋈ is sufficient for ⋊.", "⋈ is sufficient for ⋊ by transitivity."),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_template(tokenizer, texts, is_instruct):
    if not is_instruct:
        return texts
    formatted = []
    for t in texts:
        msg = [{"role": "user", "content": t}]
        formatted.append(tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return formatted


def get_last_token_hidden(model, tokenizer, texts, device, is_instruct=False):
    """[B, n_layers+1, hidden] — hidden states at last non-pad token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatted = apply_template(tokenizer, texts, is_instruct)
    inputs = tokenizer(formatted, return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    seq_lens = inputs["attention_mask"].sum(1) - 1
    hidden = torch.stack(out.hidden_states)   # [L, B, T, H]
    idx = seq_lens[None, :, None, None].expand(hidden.shape[0], -1, 1, hidden.shape[-1])
    last = hidden.gather(2, idx).squeeze(2).float()  # [L, B, H]
    return last.permute(1, 0, 2).cpu()               # [B, L, H]


def get_answer_logprob(model, tokenizer, q_text, answer, device):
    """Mean log-prob of answer tokens given q_text prefix."""
    full_text = q_text + answer
    q_ids = tokenizer(q_text, return_tensors="pt")["input_ids"][0]
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    q_len = len(q_ids)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    ans_ids = inputs["input_ids"][0][q_len:]
    if len(ans_ids) == 0:
        return 0.0
    return log_probs[q_len-1:q_len-1+len(ans_ids)].gather(
        1, ans_ids.unsqueeze(1)).mean().item()


# ── Method 1: Paraphrase Invariance ──────────────────────────────────────────

def method1_paraphrase(model, tokenizer, device, is_instruct=False):
    print("\n=== Method 1: Paraphrase Invariance ===")
    results = {}
    for label, pairs in [("knowledge", KNOWLEDGE_PAIRS), ("reasoning", REASONING_PAIRS)]:
        h_orig = get_last_token_hidden(model, tokenizer, [p[0] for p in pairs], device, is_instruct)
        h_para = get_last_token_hidden(model, tokenizer, [p[1] for p in pairs], device, is_instruct)
        sim = nn.CosineSimilarity(dim=-1)(h_orig, h_para).mean(0).numpy()
        results[label] = 1 - sim
        print(f"  {label}: mean_dist={results[label].mean():.4f}")
    return results


# ── Method 2: Information-theoretic (4 variants) ─────────────────────────────

def method2_information(model, tokenizer, device, is_instruct=False):
    """
    Four variants to distinguish KB vs RB with high separability:

    A. Hint delta:        logprob(with correct answer hint) - logprob(baseline)
                          KB: small/negative (already confident)
                          RB: large positive (hint helps a lot)

    B. Counterfactual:    logprob(with WRONG answer injected) - logprob(baseline)
                          KB: large negative (model resists wrong answer)
                          RB: small/positive (model accepts context, no strong prior)

    C. CoT sensitivity:   logprob(with step-by-step reasoning prefix) - logprob(baseline)
                          KB: small (extra steps don't help)
                          RB: large positive (chain-of-thought helps)

    D. Symbol substitution: logprob after replacing key tokens with unseen symbols
                          KB: large negative (memory destroyed)
                          RB: small change (reasoning structure preserved)
    """
    print("\n=== Method 2: Information-theoretic (A/B/C/D) ===")

    # Wrong answers for counterfactual injection (B)
    KNOWLEDGE_WRONG = [
        ("Q: What is the capital of France? A:", "Berlin"),
        ("Q: Who wrote Hamlet? A:", "Dickens"),
        ("Q: What is the boiling point of water? A:", "50 degrees Celsius"),
        ("Q: What is the atomic number of oxygen? A:", "12"),
        ("Q: Where was Napoleon exiled? A:", "Corsica"),
        ("Q: What is the longest river in Africa? A:", "The Congo"),
        ("Q: How many chambers does the human heart have? A:", "Two"),
        ("Q: Who developed the theory of relativity? A:", "Newton"),
        ("Q: What does photosynthesis produce? A:", "Carbon dioxide"),
        ("Q: What is the chemical symbol for gold? A:", "Ag"),
        ("Q: What country is the Great Wall of China in? A:", "Japan"),
        ("Q: What is the largest ocean? A:", "The Atlantic Ocean"),
        ("Q: How many chromosomes do humans have? A:", "23"),
        ("Q: What is the capital of Italy? A:", "Milan"),
        ("Q: What is the powerhouse of the cell? A:", "The nucleus"),
        ("Q: Who formulated the laws of motion? A:", "Galileo"),
        ("Q: What year did the French Revolution begin? A:", "1812"),
        ("Q: What orbits the Earth? A:", "The Sun"),
        ("Q: What is the chemical formula for water? A:", "H2O2"),
        ("Q: What planet is closest to the Sun? A:", "Venus"),
        ("Q: What is the square root of 144? A:", "14"),
        ("Q: How many sides does a hexagon have? A:", "Eight"),
        ("Q: What is the speed of sound in air? A:", "1000 meters per second"),
        ("Q: Who painted the Mona Lisa? A:", "Michelangelo"),
        ("Q: What is the hardest natural substance? A:", "Quartz"),
        ("Q: How many bones are in the adult human body? A:", "300"),
        ("Q: What gas do plants absorb during photosynthesis? A:", "Oxygen"),
        ("Q: What is the currency of Japan? A:", "The yuan"),
        ("Q: What is the tallest mountain in the world? A:", "K2"),
        ("Q: What is the smallest planet in the solar system? A:", "Mars"),
    ]
    REASONING_WRONG = [
        ("Q: What is 73847 times 29? A:", "9999999"),
        ("Q: What is 8317 times 46? A:", "111111"),
        ("Q: What is 987654 minus 398271? A:", "700000"),
        ("Q: What is 46829 plus 73541? A:", "99999"),
        ("Q: A factory makes 2347 units per day. How many in 13 days? A:", "99999"),
        ("Q: What is the area of a rectangle with sides 127 and 83? A:", "5000"),
        ("Q: If 47x equals 3901, what is x? A:", "100"),
        ("Q: What is the square root of 7569? A:", "100"),
        ("Q: A car travels 347 km at 53 km/h. How many hours? A:", "10"),
        ("Q: A tank has 8450 liters. Remove 3127. How many remain? A:", "9999"),
        ("Q: What is 63419 divided by 7? A:", "10000"),
        ("Q: What is the perimeter of a rectangle with sides 94 and 137? A:", "300"),
        ("Q: What is 2 to the power of 17? A:", "65536"),
        ("Q: What is the LCM of 12 and 18? A:", "100"),
        ("Q: What is the GCD of 84 and 56? A:", "14"),
        ("Q: A sphere has radius 5. What is its volume to one decimal? A:", "100.0"),
        ("Q: What is 9371 times 83? A:", "999999"),
        ("Q: What is 234567 minus 98765? A:", "200000"),
        ("Q: If 13x equals 1547, what is x? A:", "200"),
        ("Q: What is the area of a triangle with base 94 and height 67? A:", "9999"),
        ("Q: What is 17 squared plus 13 squared? A:", "500"),
        ("Q: A train goes 120 km/h for 2.5 hours. How far? A:", "400"),
        ("Q: What is 15 percent of 4680? A:", "1000"),
        ("Q: What is the sum of angles in a pentagon? A:", "360"),
        ("Q: If x plus 47 equals 193, what is x? A:", "200"),
        ("Q: What is 3 to the power of 8? A:", "10000"),
        ("Q: A rectangle has area 2491 and one side 47. What is the other side? A:", "100"),
        ("Q: What is 456 times 789? A:", "500000"),
        ("Q: What is the LCM of 15 and 25? A:", "100"),
        ("Q: If 7x minus 3 equals 179, what is x? A:", "30"),
    ]

    results = {
        "A_hint":    {"knowledge": [], "reasoning": []},
        "B_counter": {"knowledge": [], "reasoning": []},
        "C_cot":     {"knowledge": [], "reasoning": []},
        "D_symbol":  {"knowledge": [], "reasoning": []},
    }

    def _delta(q, a_correct, a_wrong, qa_list_idx, label):
        lp_base = get_answer_logprob(model, tokenizer, q, a_correct, device)

        # A: hint delta
        hint_q = q[:-2] + f"\n\n(Note: the answer is {a_correct})\n\nA:"
        lp_hint = get_answer_logprob(model, tokenizer, hint_q, a_correct, device)
        results["A_hint"][label].append(lp_hint - lp_base)

        # B: counterfactual (wrong answer injected)
        counter_q = q[:-2] + f"\n\n(Note: the answer is {a_wrong})\n\nA:"
        lp_counter = get_answer_logprob(model, tokenizer, counter_q, a_correct, device)
        results["B_counter"][label].append(lp_counter - lp_base)

        # C: chain-of-thought prefix
        cot_q = q[:-2] + "\n\nLet's think step by step.\n\nA:"
        lp_cot = get_answer_logprob(model, tokenizer, cot_q, a_correct, device)
        results["C_cot"][label].append(lp_cot - lp_base)

        # D: symbol substitution — replace answer tokens with placeholder
        # Proxy: measure logprob drop when question is prefixed with misleading symbol context
        sym_q = f"In this context, all specific names and numbers are replaced by symbols. {q}"
        lp_sym = get_answer_logprob(model, tokenizer, sym_q, a_correct, device)
        results["D_symbol"][label].append(lp_sym - lp_base)

    for i, ((q, a), (_, a_wrong)) in enumerate(zip(KNOWLEDGE_QA, KNOWLEDGE_WRONG)):
        _delta(q, a, a_wrong, i, "knowledge")
    for i, ((q, a), (_, a_wrong)) in enumerate(zip(REASONING_QA, REASONING_WRONG)):
        _delta(q, a, a_wrong, i, "reasoning")

    # Convert to numpy and print summary
    for variant in results:
        for label in results[variant]:
            results[variant][label] = np.array(results[variant][label])
        k = results[variant]["knowledge"].mean()
        r = results[variant]["reasoning"].mean()
        print(f"  {variant}: knowledge={k:.3f}  reasoning={r:.3f}  gap={r-k:.3f}")

    return results


# ── Method 3: MLP vs Attention Ablation (logprob metric) ─────────────────────

def method3_ablation(model, tokenizer, device, is_instruct=False):
    """
    Ablate each layer's MLP or Attention output (zero it out).
    Measure drop in answer logprob (not L2 distance).
    Knowledge: logprob drops more when MLP ablated
    Reasoning: logprob drops more when Attention ablated
    """
    print("\n=== Method 3: MLP vs Attention Ablation (logprob) ===")

    layers = model.model.layers
    n_layers = len(layers)
    nk, nr = len(KNOWLEDGE_QA), len(REASONING_QA)

    # baseline logprobs
    k_base = np.array([get_answer_logprob(model, tokenizer, q, a, device) for q, a in KNOWLEDGE_QA])
    r_base = np.array([get_answer_logprob(model, tokenizer, q, a, device) for q, a in REASONING_QA])

    k_mlp  = np.zeros((nk, n_layers))
    r_mlp  = np.zeros((nr, n_layers))
    k_attn = np.zeros((nk, n_layers))
    r_attn = np.zeros((nr, n_layers))

    def make_zero_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return hook

    for l in range(n_layers):
        print(f"  ablating layer {l}/{n_layers-1}...", end="\r")
        for component, k_arr, r_arr in [("mlp", k_mlp, r_mlp), ("attn", k_attn, r_attn)]:
            target = layers[l].mlp if component == "mlp" else layers[l].self_attn
            h = target.register_forward_hook(make_zero_hook())

            for i, (q, a) in enumerate(KNOWLEDGE_QA):
                k_arr[i, l] = get_answer_logprob(model, tokenizer, q, a, device)
            for i, (q, a) in enumerate(REASONING_QA):
                r_arr[i, l] = get_answer_logprob(model, tokenizer, q, a, device)

            h.remove()

    # impact = baseline - ablated (positive = ablation hurt)
    res = {
        "k_mlp":  (k_base[:, None] - k_mlp).mean(0),
        "r_mlp":  (r_base[:, None] - r_mlp).mean(0),
        "k_attn": (k_base[:, None] - k_attn).mean(0),
        "r_attn": (r_base[:, None] - r_attn).mean(0),
    }
    print()
    print(f"  knowledge MLP  drop (mean): {res['k_mlp'].mean():.4f}")
    print(f"  reasoning MLP  drop (mean): {res['r_mlp'].mean():.4f}")
    print(f"  knowledge Attn drop (mean): {res['k_attn'].mean():.4f}")
    print(f"  reasoning Attn drop (mean): {res['r_attn'].mean():.4f}")
    return res


# ── Plot (3 separate figures) ─────────────────────────────────────────────────

def plot_all(m1, m2, m3):
    n_layers = len(m1["knowledge"])

    # Figure 1: Paraphrase Invariance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m1["knowledge"], marker="o", color="steelblue", label="knowledge")
    ax.plot(m1["reasoning"], marker="s", color="tomato",    label="reasoning")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Paraphrase Distance (1 − cosine sim)")
    ax.set_title("Method 1: Paraphrase Invariance\nKnowledge stable early; Reasoning unstable longer")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig1_paraphrase.png", dpi=150)
    plt.close()
    print("Saved fig1_paraphrase.png")

    # Figure 2: Information-theoretic
    fig, ax = plt.subplots(figsize=(7, 5))
    k_d, r_d = m2["knowledge"], m2["reasoning"]
    positions = [1, 2]
    bp = ax.boxplot([k_d, r_d], positions=positions, patch_artist=True,
                    tick_labels=["knowledge", "reasoning"])
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("tomato")
    # overlay individual points
    ax.scatter(np.ones(len(k_d)) * 1 + np.random.uniform(-0.05, 0.05, len(k_d)), k_d,
               color="steelblue", alpha=0.6, zorder=3, s=20)
    ax.scatter(np.ones(len(r_d)) * 2 + np.random.uniform(-0.05, 0.05, len(r_d)), r_d,
               color="tomato", alpha=0.6, zorder=3, s=20)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Δ log-prob (hint − no hint)")
    ax.set_title(f"Method 2: Information-theoretic\nknowledge mean={k_d.mean():.2f}  reasoning mean={r_d.mean():.2f}")
    plt.tight_layout()
    plt.savefig("fig2_information.png", dpi=150)
    plt.close()
    print("Saved fig2_information.png")

    # Figure 3: Ablation logprob drop
    n_layers = len(m3["k_mlp"])  # transformer blocks only
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(n_layers)

    ax = axes[0]
    ax.plot(x, m3["k_mlp"], color="steelblue", label="knowledge")
    ax.plot(x, m3["r_mlp"], color="tomato",    label="reasoning")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Layer ablated"); ax.set_ylabel("Logprob drop (baseline − ablated)")
    ax.set_title("Method 3a: MLP Ablation\n(higher = MLP at this layer matters more)")
    ax.legend()

    ax = axes[1]
    ax.plot(x, m3["k_attn"], color="steelblue", label="knowledge")
    ax.plot(x, m3["r_attn"], color="tomato",    label="reasoning")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Layer ablated"); ax.set_ylabel("Logprob drop (baseline − ablated)")
    ax.set_title("Method 3b: Attention Ablation\n(higher = Attention at this layer matters more)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("fig3_ablation.png", dpi=150)
    plt.close()
    print("Saved fig3_ablation.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_checkpoint_evolution(ckpt_results):
    """
    ckpt_results: list of {"name": str, "knowledge": np.array, "reasoning": np.array}
    Produces 3 subplots:
      1. Knowledge paraphrase distance per layer across checkpoints
      2. Reasoning paraphrase distance per layer across checkpoints
      3. Boundary layer (argmax distance) and mean distance gap across checkpoints
    """
    n = len(ckpt_results)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    names = [r["name"] for r in ckpt_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: knowledge curves
    ax = axes[0]
    for r, c in zip(ckpt_results, colors):
        ax.plot(r["knowledge"], color=c, label=r["name"])
    ax.set_xlabel("Layer"); ax.set_ylabel("Paraphrase Distance")
    ax.set_title("Knowledge: paraphrase distance per layer")
    ax.legend(fontsize=7)

    # Subplot 2: reasoning curves
    ax = axes[1]
    for r, c in zip(ckpt_results, colors):
        ax.plot(r["reasoning"], color=c, label=r["name"])
    ax.set_xlabel("Layer"); ax.set_ylabel("Paraphrase Distance")
    ax.set_title("Reasoning: paraphrase distance per layer")
    ax.legend(fontsize=7)

    # Subplot 3: boundary layer + mean gap across checkpoints
    ax = axes[2]
    k_boundary = [int(np.argmax(r["knowledge"])) for r in ckpt_results]
    r_boundary = [int(np.argmax(r["reasoning"])) for r in ckpt_results]
    gap = [r["reasoning"].mean() - r["knowledge"].mean() for r in ckpt_results]
    x = np.arange(n)
    ax.plot(x, k_boundary, marker="o", color="steelblue", label="knowledge boundary layer")
    ax.plot(x, r_boundary, marker="s", color="tomato",    label="reasoning boundary layer")
    ax2 = ax.twinx()
    ax2.bar(x, gap, alpha=0.3, color="gray", label="mean dist gap (R−K)")
    ax2.set_ylabel("Mean distance gap")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Boundary layer (argmax)")
    ax.set_title("Boundary layer & gap across checkpoints")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig("fig_checkpoint_evolution.png", dpi=150)
    plt.close()
    print("Saved fig_checkpoint_evolution.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",     default="/opt/tiger/Flame/Qwen3-8B-Base")
    parser.add_argument("--instruct_model", default="/opt/tiger/Flame/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    # Checkpoint evolution mode: pass multiple checkpoint dirs
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="List of base-model checkpoint paths for Method 1 evolution analysis")
    # Pythia mode: load checkpoints by revision from HuggingFace
    parser.add_argument("--pythia_model", default=None,
                        help="Pythia model name, e.g. EleutherAI/pythia-1b")
    parser.add_argument("--pythia_steps", nargs="+", type=int, default=None,
                        help="Training steps to load, e.g. 1 512 1000 4000 16000 64000 143000")
    parser.add_argument("--symbolic", action="store_true",
                        help="Use symbolic data (SYMBOLIC_KNOWLEDGE_PAIRS / SYMBOLIC_REASONING_PAIRS) instead of natural language")
    args = parser.parse_args()

    if args.pythia_model and args.pythia_steps:
        # ── Pythia checkpoint evolution mode ──────────────────────────────────
        from transformers import GPTNeoXForCausalLM
        ckpt_results = []
        out_ckpt = {}
        pairs_k = SYMBOLIC_KNOWLEDGE_PAIRS if args.symbolic else KNOWLEDGE_PAIRS
        pairs_r = SYMBOLIC_REASONING_PAIRS if args.symbolic else REASONING_PAIRS
        data_tag = "symbolic" if args.symbolic else "natural"

        for step in args.pythia_steps:
            revision = f"step{step}"
            name = f"step{step}"
            print(f"\nLoading {args.pythia_model} @ {revision}...")
            tok = AutoTokenizer.from_pretrained(args.pythia_model, revision=revision)
            mdl = GPTNeoXForCausalLM.from_pretrained(
                args.pythia_model, revision=revision, dtype=torch.bfloat16).to(args.device)
            mdl.eval()

            # Run method1 with custom pairs
            h_orig = get_last_token_hidden(mdl, tok, [p[0] for p in pairs_k], args.device)
            h_para = get_last_token_hidden(mdl, tok, [p[1] for p in pairs_k], args.device)
            k_dist = (1 - nn.CosineSimilarity(dim=-1)(h_orig, h_para).mean(0)).numpy()

            h_orig = get_last_token_hidden(mdl, tok, [p[0] for p in pairs_r], args.device)
            h_para = get_last_token_hidden(mdl, tok, [p[1] for p in pairs_r], args.device)
            r_dist = (1 - nn.CosineSimilarity(dim=-1)(h_orig, h_para).mean(0)).numpy()

            del mdl; torch.cuda.empty_cache()
            print(f"  knowledge mean_dist={k_dist.mean():.4f}  reasoning mean_dist={r_dist.mean():.4f}")

            ckpt_results.append({"name": name, "knowledge": k_dist, "reasoning": r_dist})
            out_ckpt[name] = {
                "knowledge_per_layer": k_dist.tolist(),
                "reasoning_per_layer": r_dist.tolist(),
            }

        plot_checkpoint_evolution(ckpt_results)
        out_file = f"pythia_evolution_{data_tag}.json"
        with open(out_file, "w") as f:
            json.dump(out_ckpt, f, indent=2)
        print(f"\nSaved {out_file}")
        print(json.dumps(out_ckpt, indent=2))

    elif args.checkpoints:
        # ── Checkpoint evolution mode ──────────────────────────────────────────
        ckpt_results = []
        out_ckpt = {}
        pairs_k = SYMBOLIC_KNOWLEDGE_PAIRS if args.symbolic else KNOWLEDGE_PAIRS
        pairs_r = SYMBOLIC_REASONING_PAIRS if args.symbolic else REASONING_PAIRS
        for ckpt_path in args.checkpoints:
            name = ckpt_path.rstrip("/").split("/")[-1]
            print(f"\nLoading checkpoint {name} from {ckpt_path}...")
            tok = AutoTokenizer.from_pretrained(ckpt_path)
            mdl = AutoModelForCausalLM.from_pretrained(
                ckpt_path, dtype=torch.bfloat16).to(args.device)
            mdl.eval()

            h_orig = get_last_token_hidden(mdl, tok, [p[0] for p in pairs_k], args.device)
            h_para = get_last_token_hidden(mdl, tok, [p[1] for p in pairs_k], args.device)
            k_dist = (1 - nn.CosineSimilarity(dim=-1)(h_orig, h_para).mean(0)).numpy()

            h_orig = get_last_token_hidden(mdl, tok, [p[0] for p in pairs_r], args.device)
            h_para = get_last_token_hidden(mdl, tok, [p[1] for p in pairs_r], args.device)
            r_dist = (1 - nn.CosineSimilarity(dim=-1)(h_orig, h_para).mean(0)).numpy()

            del mdl; torch.cuda.empty_cache()
            ckpt_results.append({"name": name, "knowledge": k_dist, "reasoning": r_dist})
            out_ckpt[name] = {
                "knowledge_per_layer": k_dist.tolist(),
                "reasoning_per_layer": r_dist.tolist(),
            }

        plot_checkpoint_evolution(ckpt_results)

        with open("checkpoint_evolution.json", "w") as f:
            json.dump(out_ckpt, f, indent=2)
        print("\nSaved checkpoint_evolution.json")
        print(json.dumps(out_ckpt, indent=2))

    else:
        # ── Single-model mode (original behavior) ─────────────────────────────
        print(f"Loading base model {args.base_model}...")
        tokenizer_base = AutoTokenizer.from_pretrained(args.base_model)
        model_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16).to(args.device)
        model_base.eval()
        m1 = method1_paraphrase(model_base, tokenizer_base, args.device, is_instruct=False)
        del model_base
        torch.cuda.empty_cache()

        print(f"\nLoading instruct model {args.instruct_model}...")
        tokenizer_inst = AutoTokenizer.from_pretrained(args.instruct_model)
        model_inst = AutoModelForCausalLM.from_pretrained(
            args.instruct_model, dtype=torch.bfloat16).to(args.device)
        model_inst.eval()
        m2 = method2_information(model_inst, tokenizer_inst, args.device, is_instruct=True)
        m3 = method3_ablation(model_inst, tokenizer_inst, args.device, is_instruct=True)
        del model_inst
        torch.cuda.empty_cache()

        plot_all(m1, m2, m3)

        out = {
            "m1": {"knowledge_per_layer": m1["knowledge"].tolist(),
                   "reasoning_per_layer": m1["reasoning"].tolist()},
            "m2": {"knowledge_deltas": m2["knowledge"].tolist(),
                   "reasoning_deltas": m2["reasoning"].tolist()},
            "m3": {"k_mlp":  m3["k_mlp"].tolist(),  "r_mlp":  m3["r_mlp"].tolist(),
                   "k_attn": m3["k_attn"].tolist(), "r_attn": m3["r_attn"].tolist()},
        }
        with open("probe_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nSaved probe_results.json")
        print(json.dumps(out, indent=2))
