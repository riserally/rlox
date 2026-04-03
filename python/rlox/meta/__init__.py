"""Meta-learning algorithms.

Provides outer-loop meta-learning that leverages Rust-accelerated
weight operations for efficient parameter averaging and updates.
"""

from rlox.meta.reptile import Reptile

__all__ = ["Reptile"]
