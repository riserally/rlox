"""Plugin ecosystem for rlox -- registries and auto-discovery.

Provides decorators to register custom environments, buffers, and reward
shapers, plus entry-point based plugin discovery.

Example
-------
>>> from rlox.plugins import register_env
>>>
>>> @register_env("my-custom-env-v0")
... class MyCustomEnv:
...     pass
"""

from __future__ import annotations

import importlib.metadata


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, type] = {}
BUFFER_REGISTRY: dict[str, type] = {}
REWARD_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------

def register_env(name: str):
    """Class decorator that registers a custom environment under *name*."""
    def decorator(cls: type) -> type:
        ENV_REGISTRY[name] = cls
        return cls
    return decorator


def register_buffer(name: str):
    """Class decorator that registers a custom buffer type under *name*."""
    def decorator(cls: type) -> type:
        BUFFER_REGISTRY[name] = cls
        return cls
    return decorator


def register_reward(name: str):
    """Class decorator that registers a reward shaper under *name*."""
    def decorator(cls: type) -> type:
        REWARD_REGISTRY[name] = cls
        return cls
    return decorator


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
