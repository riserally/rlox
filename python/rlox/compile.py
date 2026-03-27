"""torch.compile utilities for rlox algorithms.

Apply ``torch.compile`` to the neural networks inside any rlox algorithm or
trainer, enabling potential speedups via graph-level optimizations.

Usage::

    from rlox.compile import compile_policy
    from rlox.algorithms.ppo import PPO

    ppo = PPO(env_id="CartPole-v1")
    compile_policy(ppo)  # compiles ppo.policy in-place
"""

from __future__ import annotations

import warnings
from typing import Any


def _has_forward(module: Any) -> bool:
    """Check if a module defines its own forward() (not just nn.Module base)."""
    import torch.nn as nn

    if not isinstance(module, nn.Module):
        return False
    # Check if forward is overridden (not the base Module.forward)
    return type(module).forward is not nn.Module.forward


def _try_compile(module: Any, name: str, **kwargs: Any) -> Any:
    """Attempt to compile a module; return original if it fails."""
    import torch

    if not _has_forward(module):
        warnings.warn(
            f"Skipping torch.compile for {name}: no forward() method. "
            f"The Rust data pipeline already provides the main speedup.",
            stacklevel=3,
        )
        return module

    try:
        return torch.compile(module, **kwargs)
    except Exception as e:
        warnings.warn(
            f"torch.compile failed for {name}: {e}. Using uncompiled model.",
            stacklevel=3,
        )
        return module


def compile_policy(
    algo: Any,
    backend: str = "inductor",
    mode: str = "default",
) -> Any:
    """Apply ``torch.compile`` to an algorithm's networks.

    Works with any rlox algorithm (PPO, SAC, DQN, TD3, etc.) or trainer
    wrapping one (via ``trainer.algo``).

    Networks that don't define ``forward()`` (like rlox's policy classes
    which use ``get_action_and_logprob``) are skipped with a warning.
    Networks with standard ``forward()`` (QNetwork, SquashedGaussianPolicy,
    etc.) are compiled normally.

    Parameters
    ----------
    algo
        Algorithm or trainer instance.
    backend : str
        ``torch.compile`` backend (default ``"inductor"``).
    mode : str
        Compilation mode: ``"default"``, ``"reduce-overhead"``, or
        ``"max-autotune"``.

    Returns
    -------
    The algorithm/trainer with compiled networks (modified in-place).
    """
    compile_kwargs = {"backend": backend, "mode": mode}

    # Unwrap trainer to get the underlying algorithm
    inner = getattr(algo, "algo", algo)

    # PPO / A2C: compile individual policy methods if forward() is not overridden
    if hasattr(inner, "policy"):
        policy = inner.policy
        if _has_forward(policy):
            inner.policy = _try_compile(policy, "policy", **compile_kwargs)
        else:
            # Compile the hot methods used during training
            import torch
            for method_name in ("get_action_and_logprob", "get_value", "get_logprob_and_entropy"):
                if hasattr(policy, method_name):
                    try:
                        original = getattr(policy, method_name)
                        compiled = torch.compile(original, **compile_kwargs)
                        setattr(policy, method_name, compiled)
                    except Exception as e:
                        warnings.warn(
                            f"torch.compile failed for policy.{method_name}: {e}",
                            stacklevel=2,
                        )

    # SAC / TD3: actor + twin critics
    if hasattr(inner, "actor"):
        inner.actor = _try_compile(inner.actor, "actor", **compile_kwargs)
    if hasattr(inner, "critic1"):
        inner.critic1 = _try_compile(inner.critic1, "critic1", **compile_kwargs)
    if hasattr(inner, "critic2"):
        inner.critic2 = _try_compile(inner.critic2, "critic2", **compile_kwargs)

    # DQN: q_network
    if hasattr(inner, "q_network"):
        inner.q_network = _try_compile(inner.q_network, "q_network", **compile_kwargs)

    return algo
