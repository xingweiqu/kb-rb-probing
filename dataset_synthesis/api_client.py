"""OpenAI-compatible API client wrapper with retry and JSON parsing.

This repo targets local vLLM servers exposing the OpenAI API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import asyncio
from itertools import cycle


def _resolve_vllm_endpoints() -> list[str]:
    """Resolve vLLM endpoints from env or defaults.

    Supports:
    - `VLLM_ENDPOINTS` as a comma-separated list of base URLs
      (e.g., "http://127.0.0.1:9101,http://127.0.0.1:9102").
    - Defaults to localhost ports 9101-9107.
    """
    raw = os.environ.get("VLLM_ENDPOINTS", "").strip()
    if raw:
        endpoints = [x.strip() for x in raw.split(",") if x.strip()]
        return endpoints
    return [f"http://127.0.0.1:{p}" for p in range(9101, 9108)]

from .configs.defaults import MAX_RETRIES, MAX_TOKENS, MODEL, REQUEST_DELAY, RETRY_DELAYS

logger = logging.getLogger(__name__)


class APIError(Exception):
    pass


class APIClient:
    def __init__(
        self,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = MAX_RETRIES,
        retry_delays: list[float] | None = None,
        request_delay: float = REQUEST_DELAY,
        mock: bool = False,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delays = retry_delays or RETRY_DELAYS
        self.request_delay = request_delay
        self.mock = mock

        # vLLM (OpenAI-compatible) endpoints.
        self.endpoints = _resolve_vllm_endpoints()
        self._rr_endpoints = cycle(self.endpoints)
        self._rr_lock = asyncio.Lock()

        self._async_client = None

    async def _get_async_http_client(self):
        # Lazy-init so importing the module doesn't require httpx.
        if self._async_client is None:
            import httpx  # type: ignore

            limits = httpx.Limits(
                max_keepalive_connections=64,
                max_connections=256,
                keepalive_expiry=30.0,
            )
            timeout = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)
            self._async_client = httpx.AsyncClient(
                trust_env=False,  # bypass sys proxy for localhost
                limits=limits,
                timeout=timeout,
            )
        return self._async_client

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    async def _next_endpoint(self) -> str:
        async with self._rr_lock:
            return next(self._rr_endpoints)

    @staticmethod
    def _extract_openai_content(payload: dict[str, Any]) -> str:
        # vLLM matches OpenAI responses: choices[0].message.content
        try:
            choice0 = payload.get("choices", [{}])[0]
            message = choice0.get("message", {})
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            # Some configurations may return reasoning separately.
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning
        except Exception:
            pass
        raise APIError("Malformed OpenAI response: missing choices[0].message.content")

    async def call_api_async(
        self,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return assistant content.

        Args:
            response_format: OpenAI-compatible `response_format`.
              Use `{ "type": "json_object" }` for strict JSON objects.
              Use `None` for free-form (needed for JSON arrays).
        """
        if self.mock:
            return json.dumps(self._mock_call_api_json(system, user), ensure_ascii=False)

        client = await self._get_async_http_client()
        endpoint = await self._next_endpoint()
        url = endpoint.rstrip("/") + "/v1/chat/completions"

        # Keep prompts explicit about JSON-only output (parsing is strict downstream).
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": float(os.environ.get("VLLM_TEMPERATURE", "0.0")),
            "top_p": float(os.environ.get("VLLM_TOP_P", "0.95")),
        }

        # Encourage strict JSON output when prompts demand it.
        if response_format is None:
            pass
        elif response_format:
            payload["response_format"] = response_format
        elif os.environ.get("VLLM_RESPONSE_FORMAT", "1") not in ("0", "false", "False"):
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = self._extract_openai_content(data)
                return text
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    # Retry on a different endpoint.
                    endpoint = await self._next_endpoint()
                    url = endpoint.rstrip("/") + "/v1/chat/completions"
        raise APIError(f"All {self.max_retries + 1} attempts failed") from last_error

    def call_api(self, system: str, user: str) -> str:
        """Sync wrapper around `call_api_async` (for legacy call sites)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_api_async(system, user))
        raise RuntimeError("call_api() cannot be used inside an event loop; use call_api_async().")

    def call_api_json(self, system: str, user: str) -> Any:
        """Call the API and parse the response as JSON.

        Retries on JSON parse failures and API errors.
        """
        if self.mock:
            return self._mock_call_api_json(system, user)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "call_api_json() cannot be used inside an event loop; use call_api_json_async()."
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.call_api(system, user)
                parsed = self._parse_json(raw)
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                return parsed
            except (APIError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "API JSON call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        e,
                    )
                    time.sleep(delay)
        raise APIError(f"All {self.max_retries + 1} attempts failed") from last_error

    async def call_api_json_async(
        self,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        if self.mock:
            return self._mock_call_api_json(system, user)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = await self.call_api_async(system, user, response_format=response_format)
                parsed = self._parse_json(raw)
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                return parsed
            except (APIError, json.JSONDecodeError) as e:
                last_error = e
                # If the model produced almost-JSON, ask it to repair deterministically.
                if isinstance(e, json.JSONDecodeError):
                    try:
                        repair_system = (
                            "You are a strict JSON repair tool. "
                            "Convert the given text into VALID JSON that matches the expected schema. "
                            "Return ONLY JSON, no markdown, no commentary. "
                            "Use double quotes for all keys and string values."
                        )
                        repair_user = (
                            "The following text was intended to be JSON but is invalid. "
                            "Rewrite it as valid JSON (preserving all keys/values you can).\n\n"
                            "TEXT:\n" + raw
                        )
                        repaired_raw = await self.call_api_async(
                            repair_system,
                            repair_user,
                            response_format={"type": "json_object"},
                        )
                        repaired = self._parse_json(repaired_raw)
                        if self.request_delay > 0:
                            await asyncio.sleep(self.request_delay)
                        return repaired
                    except Exception:
                        pass
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "API JSON call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
        raise APIError(f"All {self.max_retries + 1} attempts failed") from last_error

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1
            if lines[0].strip().startswith("```"):
                start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Best-effort recovery for occasional vLLM JSON formatting glitches.
            recovered = APIClient._recover_json_substring(text)
            if recovered is not None:
                try:
                    return json.loads(recovered)
                except json.JSONDecodeError:
                    # Try removing trailing commas.
                    recovered2 = re.sub(r",\s*([}\]])", r"\1", recovered)
                    return json.loads(recovered2)
            raise

    @staticmethod
    def _recover_json_substring(text: str) -> str | None:
        """Extract the most likely JSON object/array substring from a response."""
        # Prefer object if present.
        obj_start = text.find("{")
        arr_start = text.find("[")
        if obj_start == -1 and arr_start == -1:
            return None

        def _find_balanced(start_idx: int, open_ch: str, close_ch: str) -> str | None:
            depth = 0
            in_str = False
            esc = False
            for i in range(start_idx, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = False
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        continue
                    if ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            return text[start_idx : i + 1]
            return None

        candidates: list[str] = []
        if obj_start != -1:
            got = _find_balanced(obj_start, "{", "}")
            if got:
                candidates.append(got)
        if arr_start != -1:
            got = _find_balanced(arr_start, "[", "]")
            if got:
                candidates.append(got)

        if not candidates:
            return None
        # Choose the longest candidate.
        return max(candidates, key=len)

    # ------------------------------
    # Mock / offline synthesis mode
    # ------------------------------

    @staticmethod
    def _extract_count(user: str, default: int = 1) -> int:
        m = re.search(r"Generate\s+(\d+)", user)
        if not m:
            return default
        try:
            return int(m.group(1))
        except Exception:
            return default

    @staticmethod
    def _extract_json_object_from_text(text: str) -> dict[str, Any]:
        start = text.find("{")
        if start < 0:
            return {}
        blob = text[start:]
        try:
            return json.loads(blob)
        except Exception:
            # Best-effort: trim after last closing brace
            end = blob.rfind("}")
            if end >= 0:
                try:
                    return json.loads(blob[: end + 1])
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _extract_bulleted_field(user: str, field: str) -> str:
        prefix = f"- {field}: "
        for line in user.splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
        return ""

    def _mock_kb_structures(self, count: int) -> list[dict[str, Any]]:
        examples = [
            ("country_capital", "Germany", "Berlin", "capital_of", "The capital of Germany is Berlin."),
            ("country_capital", "Japan", "Tokyo", "capital_of", "The capital of Japan is Tokyo."),
            ("work_author", "Pride and Prejudice", "Jane Austen", "written_by", "Pride and Prejudice was written by Jane Austen."),
            ("element_symbol", "Oxygen", "O", "has_symbol", "The chemical symbol for Oxygen is O."),
            ("entity_attribute", "Earth", "blue", "typical_color", "Earth is commonly described as blue (the 'Blue Planet')."),
        ]
        out: list[dict[str, Any]] = []
        for i in range(count):
            sub_family, q_ent, ans, rel, fact = examples[i % len(examples)]
            out.append(
                {
                    "sub_family": sub_family,
                    "type": "single_hop_binding",
                    "nodes": [
                        {"id": "n1", "label": q_ent, "role": "query_entity"},
                        {"id": "n2", "label": ans, "role": "answer"},
                    ],
                    "edges": [{"source": "n1", "target": "n2", "relation": rel}],
                    "support_chain": [f"n1 --{rel}--> n2"],
                    "gold_derivation": "direct_lookup",
                    "gold_answer": ans,
                    "support_facts": [fact],
                    "required_steps": 1,
                }
            )
        return out

    def _mock_rb_structures(self, count: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(count):
            if i % 3 == 0:
                # linear equation
                out.append(
                    {
                        "sub_family": "linear_equation",
                        "type": "algebraic_equation",
                        "variables": [
                            {"name": "x", "role": "unknown"},
                            {"name": "3", "role": "constant"},
                            {"name": "10", "role": "constant"},
                        ],
                        "rules": ["x + 3 = 10"],
                        "derivation_steps": ["Subtract 3 from both sides", "x = 7"],
                        "gold_derivation": "algebraic_manipulation",
                        "gold_answer": "7",
                        "support_facts": ["x + 3 = 10"],
                        "required_steps": 2,
                    }
                )
            elif i % 3 == 1:
                # boolean logic
                out.append(
                    {
                        "sub_family": "boolean_logic",
                        "type": "boolean_expression",
                        "variables": [
                            {"name": "A", "role": "parameter"},
                            {"name": "B", "role": "parameter"},
                        ],
                        "rules": ["A = True", "B = False", "Evaluate: A AND (NOT B)"],
                        "derivation_steps": ["NOT B is True", "True AND True is True"],
                        "gold_derivation": "truth_table",
                        "gold_answer": "True",
                        "support_facts": ["A = True", "B = False", "Expression: A AND (NOT B)"],
                        "required_steps": 2,
                    }
                )
            else:
                # sequence pattern
                out.append(
                    {
                        "sub_family": "sequence_pattern",
                        "type": "sequence_rule",
                        "variables": [{"name": "n", "role": "parameter"}],
                        "rules": ["Sequence: 2, 4, 6, 8, ?", "Rule: add 2"],
                        "derivation_steps": ["Each term increases by 2", "Next term is 10"],
                        "gold_derivation": "pattern_extrapolation",
                        "gold_answer": "10",
                        "support_facts": ["2, 4, 6, 8, ?", "add 2 each step"],
                        "required_steps": 2,
                    }
                )
        return out

    def _mock_hybrid_structures(self, count: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(count):
            # simple two-hop relational
            out.append(
                {
                    "sub_family": "two_hop_relational",
                    "type": "two_hop_composition",
                    "nodes": [
                        {"id": "n1", "label": "Paris", "role": "query_entity"},
                        {"id": "n2", "label": "France", "role": "intermediate"},
                        {"id": "n3", "label": "French", "role": "answer"},
                    ],
                    "edges": [
                        {"source": "n1", "target": "n2", "relation": "capital_of"},
                        {"source": "n2", "target": "n3", "relation": "official_language"},
                    ],
                    "rules": [],
                    "support_chain": ["Paris --capital_of--> France", "France --official_language--> French"],
                    "gold_derivation": "multi_hop_lookup",
                    "gold_answer": "French",
                    "support_facts": [
                        "Paris is the capital of France.",
                        "The official language of France is French.",
                    ],
                    "required_steps": 2,
                }
            )
        return out

    def _mock_base_item(self, structure: dict[str, Any]) -> dict[str, Any]:
        gold = structure.get("gold_answer", "")
        sub_family = structure.get("sub_family", "")

        # Graph-based KB/Hybrid
        if structure.get("nodes") and structure.get("edges"):
            nodes = structure.get("nodes", [])
            edges = structure.get("edges", [])
            q_ent = next((n.get("label") for n in nodes if n.get("role") == "query_entity"), "")
            rel = edges[0].get("relation", "") if edges else ""
            if sub_family == "country_capital" or rel == "capital_of":
                base_q = f"What is the capital of {q_ent}?"
            elif sub_family == "work_author" or rel == "written_by":
                base_q = f"Who wrote {q_ent}?"
            elif sub_family == "element_symbol" or rel == "has_symbol":
                base_q = f"What is the chemical symbol for {q_ent}?"
            elif sub_family == "two_hop_relational":
                base_q = "What language is spoken in the country whose capital is Paris?"
            else:
                base_q = f"Given the relation {rel}, what is associated with {q_ent}?"

            facts = structure.get("support_facts", [])
            if not facts:
                facts = structure.get("support_facts", [])
            reasoning = ["Use the provided relation/facts."]
            if gold:
                reasoning.append("Retrieve the required fact(s) to get the answer.")
            return {
                "base_question": base_q,
                "gold_answer": gold,
                "gold_reasoning_chain": reasoning,
                "support_facts": facts or [],
            }

        # Rule-based RB
        rules = structure.get("rules", [])
        base_q = " ".join(rules) + " What is the answer?" if rules else "Apply the rules and give the answer."
        return {
            "base_question": base_q,
            "gold_answer": gold,
            "gold_reasoning_chain": structure.get("derivation_steps", []) or ["Apply the given rules."],
            "support_facts": structure.get("support_facts", []) or rules,
        }

    def _mock_variants(self, user: str) -> dict[str, Any]:
        base_q = self._extract_bulleted_field(user, "base_question")
        gold = self._extract_bulleted_field(user, "gold_answer")
        support_facts_raw = self._extract_bulleted_field(user, "support_facts")
        try:
            support_facts = json.loads(support_facts_raw) if support_facts_raw else []
        except Exception:
            support_facts = []

        reasoning_raw = self._extract_bulleted_field(user, "gold_reasoning_chain")
        try:
            reasoning = json.loads(reasoning_raw) if reasoning_raw else []
        except Exception:
            reasoning = []

        def q(text: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"question": text, "metadata": meta or {}}

        # Simple scaffolds
        steps = [
            "Step 1: Identify the relevant premise/rule.",
            "Step 2: Apply it to the query.",
            "Step 3: State the final answer.",
        ]

        wrong = "a plausible but incorrect alternative"
        if gold and gold.isdigit():
            wrong = str(int(gold) + 1)
        elif gold:
            wrong = gold + " (wrong)"

        variants: dict[str, Any] = {
            "original": q(base_q),
            "hint": q(base_q + " Hint: focus on the key relation/rule."),
            "premise": q(" ".join(support_facts[:1]) + " " + base_q if support_facts else base_q, {"injected_premise": support_facts[:1]}),
            "premise_removal": q("Answer the question: " + re.sub(r"\bof\s+[^?]+\?", "?", base_q), {"removed": True}),
            "highlight": q("Focus on the key evidence needed. " + base_q, {"highlight": "key evidence"}),
            "full_support_bundle": q((" ".join(support_facts) + " " + base_q).strip(), {"support_bundle": support_facts}),
            "scaffold_1": q("First identify the key premise. Then: " + base_q, {"steps": steps[:1]}),
            "scaffold_2": q("Follow these steps: 1) " + steps[0] + " 2) " + steps[1] + " Now: " + base_q, {"steps": steps[:2]}),
            "scaffold_3": q("Follow these steps: " + " ".join(steps[:3]) + " Now: " + base_q, {"steps": steps[:3]}),
            "scaffold_shuffled": q(
                "Steps (shuffled): " + " ".join([steps[1], steps[0], steps[2]]) + " Now: " + base_q,
                {"original_order": steps[:3], "shuffled_order": [steps[1], steps[0], steps[2]]},
            ),
            "cot_full": q(
                base_q
                + " Let's think step by step. "
                + " ".join([f"Step {i+1}: {s}" for i, s in enumerate(reasoning or steps[:2])])
                + " Therefore, the answer is?",
                {"cot": "full"},
            ),
            "cot_partial": q(
                base_q
                + " Let's think step by step. "
                + " ".join([f"Step {i+1}: {s}" for i, s in enumerate((reasoning or steps[:2])[:1])])
                + " Step 2: ... Therefore, the answer is?",
                {"included_steps": (reasoning or steps[:2])[:1], "omitted_steps": (reasoning or steps[:2])[1:]},
            ),
            "cot_shuffled": q(
                base_q
                + " Let's think step by step (shuffled). "
                + " ".join([f"Step {i+1}: {s}" for i, s in enumerate([steps[1], steps[0]])])
                + " Therefore, the answer is?",
                {"original_order": steps[:2], "shuffled_order": [steps[1], steps[0]]},
            ),
            "wrongclaim_bare": q(
                f"Some sources say the answer is {wrong}. {base_q}",
                {"wrong_claim": wrong},
            ),
            "wrongclaim_confident": q(
                f"Obviously, the answer is {wrong}. {base_q}",
                {"wrong_claim": wrong, "confidence_wrapper": "Obviously"},
            ),
            "wrongclaim_attributed": q(
                f"According to experts, the answer is {wrong}. {base_q}",
                {"wrong_claim": wrong, "attribution": "experts"},
            ),
            "competing_claims": q(
                f"Some say the answer is {gold}, others say it is {wrong}. {base_q}",
                {"correct_claim": gold, "wrong_claim": wrong},
            ),
            "paraphrase": q("Rephrased: " + base_q),
            "terminology_swap": q("Using technical terminology: " + base_q, {"swap_map": {"answer": "solution"}}),
            "substitution": q("With harmless substitutions: " + base_q, {"substitution_map": {}}),
        }
        return variants

    def _mock_mcq(self, user: str) -> dict[str, Any]:
        gold = self._extract_bulleted_field(user, "gold_answer")
        # If gold is a symbol, craft other symbols; otherwise simple string distractors.
        symbol_pool = ["∆", "◇", "⊕", "Ω", "★", "⊗", "▽", "◆"]
        if gold in symbol_pool:
            distractors = [s for s in symbol_pool if s != gold][:3]
        else:
            distractors = [f"{gold}_alt{i}" for i in range(1, 4)]
        options = [gold] + distractors
        return {
            "options": options,
            "correct_index": 0,
            "option_metadata": [
                {"role": "gold", "source": "mock"},
                {"role": "same_type", "source": "mock"},
                {"role": "structurally_related", "source": "mock"},
                {"role": "wrongclaim_aligned", "source": "mock"},
            ],
        }

    def _mock_call_api_json(self, system: str, user: str) -> Any:
        # Stage 1: structures
        if "Generate underlying structures for knowledge-based (KB)" in system:
            return self._mock_kb_structures(self._extract_count(user, default=1))
        if "Generate underlying structures for rule-based (RB)" in system:
            return self._mock_rb_structures(self._extract_count(user, default=1))
        if "Generate underlying structures for Hybrid item families" in system:
            return self._mock_hybrid_structures(self._extract_count(user, default=1))

        # Stage 3: variants
        if "generate all 19 controlled variants" in system.lower() or "generate all 19 variants" in user.lower():
            return self._mock_variants(user)

        # Stage 5: MCQ
        if "Generate MCQ" in system or "multiple choice question" in system.lower():
            return self._mock_mcq(user)

        # Stage 2: base items
        # (Make this check more specific so it doesn't catch MCQ prompts.)
        if "base_question" in system and "gold_answer" in system and "Return a JSON object" in system:
            structure = self._extract_json_object_from_text(user)
            return self._mock_base_item(structure)

        # Fallback
        return {}
