"""Plugin ecosystem for rlox -- registries and auto-discovery.

Provides decorators to register custom environments, buffers, and reward
shapers, plus entry-point based plugin discovery.  Registration decorators
validate that classes implement the required interface (Part 4).

Helper functions :func:`get_buffer` and :func:`get_reward_shaper` create
instances by name from the registries, with fuzzy "did you mean?" suggestions
on typos (Part 8).

Example
-------
>>> from rlox.plugins import register_env
>>>
>>> @register_env("my-custom-env-v0")
... class MyCustomEnv:
...     def step(self, action): ...
...     def reset(self): ...
"""

from __future__ import annotations

import difflib
import importlib.metadata
from typing import Any


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, type] = {}
BUFFER_REGISTRY: dict[str, type] = {}
REWARD_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Fuzzy suggestion helper (Part 8)
# ---------------------------------------------------------------------------

def _suggest_similar(name: str, registry: dict[str, Any]) -> str:
    """Return a 'Did you mean?' hint for *name* against *registry* keys."""
    matches = difflib.get_close_matches(name, registry.keys(), n=3, cutoff=0.6)
    if matches:
        return f" Did you mean: {', '.join(matches)}?"
    return ""


# ---------------------------------------------------------------------------
# Registration decorators (with validation -- Part 4)
# ---------------------------------------------------------------------------

def register_env(name: str):
    """Class decorator that registers a custom environment under *name*.

    Validates that the class has ``step`` and ``reset`` methods.
    """
    def decorator(cls: type) -> type:
        for method in ("step", "reset"):
            if not hasattr(cls, method):
                raise TypeError(
                    f"Environment {cls.__name__} missing required method: {method}"
                )
        ENV_REGISTRY[name] = cls
        return cls
    return decorator


def register_buffer(name: str):
    """Class decorator that registers a custom buffer type under *name*.

    Validates that the class has ``push``, ``sample``, and ``__len__`` methods.
    """
    def decorator(cls: type) -> type:
        for method in ("push", "sample", "__len__"):
            if not hasattr(cls, method):
                raise TypeError(
                    f"Buffer {cls.__name__} missing required method: {method}"
                )
        BUFFER_REGISTRY[name] = cls
        return cls
    return decorator


def register_reward(name: str):
    """Class decorator that registers a reward shaper under *name*.

    Validates that the class has a ``shape`` or ``__call__`` method.
    """
    def decorator(cls: type) -> type:
        has_shape = "shape" in cls.__dict__
        has_call = "__call__" in cls.__dict__
        if not has_shape and not has_call:
            raise TypeError(
                f"Reward shaper {cls.__name__} missing required method: "
                f"shape or __call__"
            )
        REWARD_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Helper functions (Parts 2, 3)
# ---------------------------------------------------------------------------

def get_buffer(name: str, capacity: int, obs_dim: int, act_dim: int) -> Any:
    """Create a buffer by registry name.

    Checks built-in buffers first (``replay``, ``prioritized``), then
    falls back to :data:`BUFFER_REGISTRY`.

    Parameters
    ----------
    name : str
        Buffer type name.
    capacity : int
        Maximum number of transitions.
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Action dimensionality.

    Returns
    -------
    Buffer instance.

    Raises
    ------
    ValueError
        If *name* is not found in built-in or registered buffers.
    """
    import rlox as _rlox

    builtin: dict[str, type] = {
        "replay": _rlox.ReplayBuffer,
        "prioritized": _rlox.PrioritizedReplayBuffer,
    }
    all_buffers = {**builtin, **BUFFER_REGISTRY}
    if name not in all_buffers:
        hint = _suggest_similar(name, all_buffers)
        raise ValueError(
            f"Unknown buffer: {name!r}. Available: {sorted(all_buffers)}.{hint}"
        )
    return all_buffers[name](capacity, obs_dim, act_dim)


def get_reward_shaper(name: str, **kwargs: Any) -> Any:
    """Create a reward shaper by registry name.

    Checks built-in shapers first (``potential``, ``goal_distance``), then
    falls back to :data:`REWARD_REGISTRY`.

    Parameters
    ----------
    name : str
        Reward shaper name.
    **kwargs
        Forwarded to the shaper constructor.

    Returns
    -------
    Reward shaper instance.

    Raises
    ------
    ValueError
        If *name* is not found in built-in or registered shapers.
    """
    from rlox.reward_shaping import PotentialShaping, GoalDistanceShaping

    builtin: dict[str, type] = {
        "potential": PotentialShaping,
        "goal_distance": GoalDistanceShaping,
    }
    all_rewards = {**builtin, **REWARD_REGISTRY}
    if name not in all_rewards:
        hint = _suggest_similar(name, all_rewards)
        raise ValueError(
            f"Unknown reward shaper: {name!r}. Available: {sorted(all_rewards)}.{hint}"
        )
    return all_rewards[name](**kwargs)


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------

def discover_plugins(namespace: str = "rlox_plugins") -> None:
    """Auto-discover plugins installed as entry points.

    Each entry point in the given *namespace* group is loaded, which
    triggers any ``@register_*`` decorators defined at module level.
    """
    for ep in importlib.metadata.entry_points(group=namespace):
        ep.load()


def list_registered() -> dict[str, list[str]]:
    """Return all registered algorithms, environments, buffers, and rewards.

    Returns
    -------
    dict mapping category name to a sorted list of registered names.
    """
    from rlox.trainer import ALGORITHM_REGISTRY

    return {
        "algorithms": sorted(ALGORITHM_REGISTRY.keys()),
        "environments": sorted(ENV_REGISTRY.keys()),
        "buffers": sorted(BUFFER_REGISTRY.keys()),
        "rewards": sorted(REWARD_REGISTRY.keys()),
    }
