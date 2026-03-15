"""Training diagnostics: detect common failure modes and issue warnings."""

from __future__ import annotations

import warnings
from typing import Any

from rlox.callbacks import Callback


class TrainingDiagnostics(Callback):
    """Auto-detect common training failures.

    Monitors training batches for pathological behavior and accumulates
    warnings in ``self.warnings``. Also emits Python warnings for
    immediate visibility.

    Detections
    ----------
    - Entropy collapse: entropy drops below 10% of initial
    - KL spike: approx_kl > 10x target_kl
    - Gradient explosion: grad_norm > 100x max_grad_norm
    - Value function divergence: explained_var < -1
    """

    def __init__(
        self,
        target_kl: float | None = None,
        max_grad_norm: float = 0.5,
    ) -> None:
        super().__init__()
        self.initial_entropy: float | None = None
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.warnings: list[str] = []

    def on_train_batch(self, **kwargs: Any) -> None:
        """Check training metrics for pathologies after each SGD update."""
        entropy = kwargs.get("entropy")
        approx_kl = kwargs.get("approx_kl")
        grad_norm = kwargs.get("grad_norm")
        explained_var = kwargs.get("explained_var")

        if entropy is not None:
            if self.initial_entropy is None:
                self.initial_entropy = entropy
            elif self.initial_entropy > 0 and entropy < 0.1 * self.initial_entropy:
                msg = (
                    f"Entropy collapse detected: {entropy:.4f} < 10% of "
                    f"initial {self.initial_entropy:.4f}"
                )
                self.warnings.append(msg)
                warnings.warn(msg, stacklevel=2)

        if (
            approx_kl is not None
            and self.target_kl is not None
            and approx_kl > 10 * self.target_kl
        ):
            msg = (
                f"KL spike detected: approx_kl={approx_kl:.4f} > "
                f"10x target_kl={self.target_kl:.4f}"
            )
            self.warnings.append(msg)
            warnings.warn(msg, stacklevel=2)

        if grad_norm is not None and grad_norm > 100 * self.max_grad_norm:
            msg = (
                f"Gradient explosion detected: grad_norm={grad_norm:.4f} > "
                f"100x max_grad_norm={self.max_grad_norm:.4f}"
            )
            self.warnings.append(msg)
            warnings.warn(msg, stacklevel=2)

        if explained_var is not None and explained_var < -1.0:
            msg = (
                f"Value function divergence: explained_var={explained_var:.4f} < -1.0"
            )
            self.warnings.append(msg)
            warnings.warn(msg, stacklevel=2)

    def on_step(self, **kwargs: Any) -> bool:
        """Legacy hook — delegates to on_train_batch for backward compat."""
        entropy = kwargs.get("entropy")
        approx_kl = kwargs.get("approx_kl")
        grad_norm = kwargs.get("grad_norm")
        explained_var = kwargs.get("explained_var")

        # Only forward if there are diagnostic-relevant kwargs
        if any(v is not None for v in (entropy, approx_kl, grad_norm, explained_var)):
            self.on_train_batch(**kwargs)

        return True
