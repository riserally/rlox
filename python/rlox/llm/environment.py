"""LLM Environment: wraps inference backends for RL interaction."""

from __future__ import annotations

from typing import Any

from rlox.distributed.vllm_backend import create_backend, _BaseBackend


class LLMEnvironment:
    """High-level LLM environment for text generation in RL loops.

    Wraps a vLLM/TGI/SGLang backend to provide a unified generate() interface.

    Parameters
    ----------
    backend : str
        One of "vllm", "tgi", "sglang".
    url : str
        Server URL. Defaults vary by backend.
    """

    def __init__(
        self,
        backend: str = "vllm",
        url: str | None = None,
    ):
        self.backend_name = backend
        self._backend: _BaseBackend = create_backend(backend, url)

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

    def log_probs(
        self,
        input_ids: list[list[int]],
    ) -> list[list[float]]:
        """Score input_ids and return per-token log probabilities."""
        return self._backend.log_probs(input_ids)
