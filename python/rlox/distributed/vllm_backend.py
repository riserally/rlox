"""Inference backends for LLM generation: vLLM, TGI, and SGLang.

Each backend implements a common interface for text generation and
log-probability scoring via HTTP APIs.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any


class _BaseBackend:
    """Base class for HTTP-based inference backends."""

    def __init__(self, base_url: str) -> None:
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
        raise NotImplementedError

    def log_probs(
        self,
        input_ids: list[list[int]],
    ) -> list[list[float]]:
        raise NotImplementedError


class VllmBackend(_BaseBackend):
    """Client for a vLLM-compatible HTTP server.

    Sends requests to the OpenAI-compatible ``/v1/completions`` endpoint
    and a custom ``/v1/log_probs`` endpoint.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__(base_url)

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
                results.append(
                    {
                        "text": choice.get("text", ""),
                        "log_probs": log_probs,
                    }
                )

        return results

    def log_probs(
        self,
        input_ids: list[list[int]],
    ) -> list[list[float]]:
        """Score input_ids and return per-token log probabilities."""
        data = self._post("/v1/log_probs", {"input_ids": input_ids})
        return data.get("log_probs", [])


class TgiBackend(_BaseBackend):
    """Client for HuggingFace Text Generation Inference (TGI).

    Uses the ``/generate`` and ``/generate_stream`` endpoints.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        super().__init__(base_url)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 10,
        n: int = 1,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        results = []
        for prompt in prompts:
            for _ in range(n):
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_new_tokens,
                        "details": True,
                        **kwargs,
                    },
                }
                data = self._post("/generate", payload)
                text = data.get("generated_text", "")
                tokens = data.get("details", {}).get("tokens", [])
                token_logprobs = [t.get("logprob", 0.0) for t in tokens]
                results.append({"text": text, "log_probs": token_logprobs})
        return results

    def log_probs(
        self,
        input_ids: list[list[int]],
    ) -> list[list[float]]:
        """TGI doesn't natively support scoring — use generate with 0 tokens."""
        results = []
        for ids in input_ids:
            # Decode IDs to text (placeholder — real impl needs tokenizer)
            text = " ".join(str(i) for i in ids)
            payload = {
                "inputs": text,
                "parameters": {"max_new_tokens": 0, "details": True},
            }
            data = self._post("/generate", payload)
            tokens = data.get("details", {}).get("prefill", [])
            results.append([t.get("logprob", 0.0) for t in tokens])
        return results


class SglangBackend(_BaseBackend):
    """Client for SGLang inference server.

    Uses the ``/generate`` endpoint with SGLang's native format.
    """

    def __init__(self, base_url: str = "http://localhost:30000"):
        super().__init__(base_url)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 10,
        n: int = 1,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        results = []
        for prompt in prompts:
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "n": n,
                    **kwargs,
                },
            }
            data = self._post("/generate", payload)
            for output in data.get("text", [data.get("text", "")]):
                if isinstance(output, str):
                    results.append({"text": output, "log_probs": None})
                else:
                    results.append({"text": str(output), "log_probs": None})
        return results

    def log_probs(
        self,
        input_ids: list[list[int]],
    ) -> list[list[float]]:
        """Score input_ids via SGLang endpoint."""
        data = self._post("/log_probs", {"input_ids": input_ids})
        return data.get("log_probs", [])


def create_backend(
    backend: str = "vllm",
    url: str | None = None,
) -> _BaseBackend:
    """Factory for inference backends.

    Parameters
    ----------
    backend : str
        One of "vllm", "tgi", "sglang".
    url : str, optional
        Server URL. Defaults vary by backend.
    """
    backends = {
        "vllm": (VllmBackend, "http://localhost:8000"),
        "tgi": (TgiBackend, "http://localhost:8080"),
        "sglang": (SglangBackend, "http://localhost:30000"),
    }
    if backend not in backends:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: {list(backends.keys())}"
        )
    cls, default_url = backends[backend]
    return cls(base_url=url or default_url)
