# Hidden Input Sanity Check

本检查仅基于现有数据文件与 `run_first_pass_probing.py` 的 hidden-state 缓存逻辑；不重跑模型、不重新提取 hidden states。

## Code path (input construction)

- `run_first_pass_probing.py:666` `cache_hidden_states(...)`：对非 MCQ，`final_p = prompt`，`pre_p = build_pre_answer_prompt(prompt)`。
- `run_first_pass_probing.py:525` `extract_last_token_hidden(...)`：对 `batch_texts` 直接调用 `tokenizer(batch_texts, add_special_tokens=False, truncation=True, max_length=512, padding=True)`。

## Samples (actual input_text sent into tokenizer)

每条样例给出：`family_id`、`variant`、`input_text(final_input)`、`input_text(pre_answer)`，并检查 `gold_answer` 是否出现在 input_text 中。

### Sample 1
- family_id: `probe_nat_hybrid_025`
- variant: `original`
- gold_answer: `No`
- gold_in_final_input: `False`
- gold_in_pre_answer_input: `False`
```text
FINAL_INPUT:
Can a 17‑year‑old legally vote in most countries?

PRE_ANSWER:
Can a 17‑year‑old legally vote in most countries?
Answer:
```

### Sample 2
- family_id: `probe_nat_hybrid_025`
- variant: `wrongclaim_bare`
- gold_answer: `No`
- gold_in_final_input: `False`
- gold_in_pre_answer_input: `False`
```text
FINAL_INPUT:
Some sources claim the answer is $3.00. Can a 17‑year‑old legally vote in most countries?

PRE_ANSWER:
Some sources claim the answer is $3.00. Can a 17‑year‑old legally vote in most countries?
Answer:
```

### Sample 3
- family_id: `probe_nat_hybrid_025`
- variant: `premise_removal`
- gold_answer: `No`
- gold_in_final_input: `False`
- gold_in_pre_answer_input: `False`
```text
FINAL_INPUT:
Answer the question without relying on any added premises: Can a 17‑year‑old legally vote in most countries?

PRE_ANSWER:
Answer the question without relying on any added premises: Can a 17‑year‑old legally vote in most countries?
Answer:
```

### Sample 4
- family_id: `probe_nat_hybrid_026`
- variant: `wrongclaim_bare`
- gold_answer: `No`
- gold_in_final_input: `False`
- gold_in_pre_answer_input: `False`
```text
FINAL_INPUT:
Some sources claim the answer is 15 hours. A patient’s recorded body temperature is 95°F. Based on the medical definition of fever, does this patient have a fever?

PRE_ANSWER:
Some sources claim the answer is 15 hours. A patient’s recorded body temperature is 95°F. Based on the medical definition of fever, does this patient have a fever?
Answer:
```

### Sample 5
- family_id: `probe_nat_hybrid_026`
- variant: `original`
- gold_answer: `No`
- gold_in_final_input: `False`
- gold_in_pre_answer_input: `False`
```text
FINAL_INPUT:
A patient’s recorded body temperature is 95°F. Based on the medical definition of fever, does this patient have a fever?

PRE_ANSWER:
A patient’s recorded body temperature is 95°F. Based on the medical definition of fever, does this patient have a fever?
Answer:
```

