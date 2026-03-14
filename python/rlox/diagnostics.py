"""Training diagnostics: detect common failure modes and issue warnings."""

from __future__ import annotations

import warnings
from typing import Any

from rlox.callbacks import Callback


class TrainingDiagnostics(Callback):
    """Detect and warn about common training pathologies.

    - Entropy collapse (entropy < 10% of initial)
    - KL spike (approx_kl > 10x target_kl)
    - Gradient explosion (grad_norm > 100x baseline)
    - Value function divergence (explained_var < -1)
    """

    def __init__(self, target_kl: float = 0.01):
        super().__init__()
        self.target_kl = target_kl
        self._initial_entropy: float | None = None
        self._baseline_grad_norm: float | None = None

    def on_step(self, **kwargs: Any) -> bool:
        entropy = kwargs.get("entropy")
        approx_kl = kwargs.get("approx_kl")
        grad_norm = kwargs.get("grad_norm")
        explained_var = kwargs.get("explained_var")

        if entropy is not None:
            if self._initial_entropy is None:
                self._initial_entropy = entropy
            elif self._initial_entropy > 0 and entropy < 0.1 * self._initial_entropy:
                warnings.warn(
                    f"Entropy collapse detected: {entropy:.4f} < 10% of "
                    f"initial {self._initial_entropy:.4f}",
                    stacklevel=2,
                )

        if approx_kl is not None and approx_kl > 10 * self.target_kl:
            warnings.warn(
                f"KL spike detected: approx_kl={approx_kl:.4f} > "
                f"10x target_kl={self.target_kl:.4f}",
                stacklevel=2,
            )

        if grad_norm is not None:
            if self._baseline_grad_norm is None:
                self._baseline_grad_norm = grad_norm
            elif (
                self._baseline_grad_norm > 0
                and grad_norm > 100 * self._baseline_grad_norm
            ):
                warnings.warn(
                    f"Gradient explosion detected: grad_norm={grad_norm:.4f} > "
                    f"100x baseline={self._baseline_grad_norm:.4f}",
                    stacklevel=2,
                )

        if explained_var is not None and explained_var < -1.0:
            warnings.warn(
                f"Value function divergence: explained_var={explained_var:.4f} < -1.0",
                stacklevel=2,
            )

        return True
