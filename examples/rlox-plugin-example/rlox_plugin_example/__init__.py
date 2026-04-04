"""Example rlox plugin package.

Demonstrates how to register a custom environment and a custom buffer
wrapper using the rlox plugin system.

Install this package (``pip install -e .``) and rlox will auto-discover
the registered components via the ``rlox_plugins`` entry point group.

Usage
-----
After installation::

    from rlox.plugins import list_registered

    reg = list_registered()
    print(reg["environments"])  # [..., "NoisyCartPole-v0"]
    print(reg["buffers"])       # [..., "logging-replay"]
"""

from __future__ import annotations

from rlox_plugin_example.envs import NoisyCartPole  # noqa: F401
from rlox_plugin_example.buffers import LoggingReplayBuffer  # noqa: F401


def register_all() -> None:
    """Entry point hook -- importing this module triggers the decorators."""
