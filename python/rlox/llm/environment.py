"""LLM Environment: wraps a vLLM backend for RL interaction."""

from __future__ import annotations

from typing import Any

from rlox.distributed.vllm_backend import VllmBackend


class LLMEnvironment:
    """High-level LLM environment for text generation in RL loops.

    Wraps a VllmBackend to provide a simple generate() interface.
    """

    def __init__(self, backend: str = "vllm", url: str = "http://localhost:8000"):
        self.backend_name = backend
        self.url = url

        if backend == "vllm":
            self._backend = VllmBackend(base_url=url)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        max_new_tokens: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate completions for prompts using the configured backend."""
        return self._backend.generate(
            prompts, max_new_tokens=max_new_tokens, n=n, **kwargs
        )
