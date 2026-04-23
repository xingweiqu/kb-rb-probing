"""Anthropic API client wrapper with retry and JSON parsing.

Replace the `call_api` implementation with your actual API call logic.
The rest of the pipeline only calls `call_api` — swap it out freely.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

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
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delays = retry_delays or RETRY_DELAYS
        self.request_delay = request_delay

    def call_api(self, system: str, user: str) -> str:
        """Send a message to the API and return the text response.

        ============================================================
        TODO: Replace this with your actual Anthropic API call.
        ============================================================

        Expected behavior:
          - Send `system` as the system prompt and `user` as the user message
          - Return the assistant's text response as a string
          - Raise APIError on failure after retries are exhausted

        Example implementation with the Anthropic SDK:

            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        """
        raise NotImplementedError(
            "Replace APIClient.call_api with your actual API implementation. "
            "See the docstring for the expected interface."
        )

    def call_api_json(self, system: str, user: str) -> Any:
        """Call the API and parse the response as JSON.

        Retries on JSON parse failures and API errors.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.call_api(system, user)
                parsed = self._parse_json(raw)
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                return parsed
            except (APIError, json.JSONDecodeError, NotImplementedError) as e:
                last_error = e
                if isinstance(e, NotImplementedError):
                    raise
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        e,
                    )
                    time.sleep(delay)
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
        return json.loads(text)
