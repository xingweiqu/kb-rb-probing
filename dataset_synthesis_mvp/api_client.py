"""OpenAI-compatible async API client for the MVP pipeline.

Targets the ridgerzhu proxy (or any OpenAI-compatible endpoint) configured via
`config.ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` plus `MODEL`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]
REQUEST_DELAY = 0.0


class APIError(Exception):
    pass


class APIClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        max_tokens: int = 6144,
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
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._async_client = None

    async def _get_async_http_client(self):
        if self._async_client is None:
            import httpx  # type: ignore

            limits = httpx.Limits(
                max_keepalive_connections=64,
                max_connections=256,
                keepalive_expiry=30.0,
            )
            timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
            self._async_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return self._async_client

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    @staticmethod
    def _extract_openai_content(payload: dict[str, Any]) -> str:
        try:
            choice0 = payload.get("choices", [{}])[0]
            message = choice0.get("message", {})
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning
        except Exception:
            pass
        raise APIError("Malformed response: missing choices[0].message.content")

    async def call_api_async(
        self,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if self.mock:
            return "{}"

        client = await self._get_async_http_client()
        url = self.base_url + "/v1/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user or "."})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": float(os.environ.get("MVP_TEMPERATURE", "0.0")),
            "top_p": float(os.environ.get("MVP_TOP_P", "0.95")),
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return self._extract_openai_content(data)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
        raise APIError(f"All {self.max_retries + 1} attempts failed") from last_error

    async def call_api_json_async(
        self,
        system: str,
        user: str,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        last_raw: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                last_raw = await self.call_api_async(system, user, response_format=response_format)
                parsed = self._parse_json(last_raw)
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                return parsed
            except (APIError, json.JSONDecodeError) as e:
                last_error = e
                if isinstance(e, json.JSONDecodeError) and last_raw:
                    try:
                        repair_system = (
                            "You are a strict JSON repair tool. "
                            "Convert the given text into VALID JSON. "
                            "Return ONLY JSON, no markdown, no commentary."
                        )
                        repair_user = (
                            "The following text was intended to be JSON but is invalid. "
                            "Rewrite it as valid JSON.\n\nTEXT:\n" + last_raw
                        )
                        repaired_raw = await self.call_api_async(
                            repair_system, repair_user,
                            response_format={"type": "json_object"},
                        )
                        return self._parse_json(repaired_raw)
                    except Exception:
                        pass
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "API JSON call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
        raise APIError(f"All {self.max_retries + 1} attempts failed") from last_error

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
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
            recovered = APIClient._recover_json_substring(text)
            if recovered is not None:
                try:
                    return json.loads(recovered)
                except json.JSONDecodeError:
                    recovered2 = re.sub(r",\s*([}\]])", r"\1", recovered)
                    return json.loads(recovered2)
            raise

    @staticmethod
    def _recover_json_substring(text: str) -> str | None:
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
        return max(candidates, key=len) if candidates else None
