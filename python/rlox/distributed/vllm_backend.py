"""vLLM-compatible backend for text generation and log-prob scoring."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any


class VllmBackend:
    """Client for a vLLM-compatible HTTP server.

    Sends requests to the OpenAI-compatible ``/v1/completions`` endpoint
    and a custom ``/v1/log_probs`` endpoint.

    Uses stdlib urllib to avoid requiring the ``requests`` package.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        """Send a JSON POST request and return the parsed response."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 10,
        n: int = 1,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate completions for a list of prompts.

        Returns a list of dicts with keys "text" and "log_probs".
        """
        results = []
        for prompt in prompts:
            payload = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "n": n,
                "logprobs": 1,
                **kwargs,
            }
            data = self._post("/v1/completions", payload)

            for choice in data.get("choices", []):
                log_probs = None
                if "logprobs" in choice and choice["logprobs"]:
                    log_probs = choice["logprobs"].get("token_logprobs", [])
                results.append({
                    "text": choice.get("text", ""),
                    "log_probs": log_probs,
                })

        return results

    def log_probs(
        self,
        input_ids: list[list[int]],
    ) -> list[list[float]]:
        """Score input_ids and return per-token log probabilities.

        Calls a custom ``/v1/log_probs`` endpoint.
        """
        data = self._post("/v1/log_probs", {"input_ids": input_ids})
        return data.get("log_probs", [])
