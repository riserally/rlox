"""RemoteEnvPool: gRPC-based distributed environment pool.

Presents the same interface as :class:`~rlox.gym_vec_env.GymVecEnv` and
:class:`~rlox.VecEnv` so it can be used directly with
:class:`~rlox.collectors.RolloutCollector`.

This is a stub implementation — actual gRPC communication will be wired
up once the Rust gRPC worker server is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class _WorkerInfo:
    """Metadata for a single remote worker."""

    address: str
    envs: int


class RemoteEnvPool:
    """Pool of remote environment workers connected via gRPC.

    Presents the same interface as GymVecEnv / VecEnv so it can be used
    directly with RolloutCollector.

    Example::

        pool = RemoteEnvPool(
            workers=["gpu-node-1:50051", "gpu-node-2:50051"],
            envs_per_worker=64,
        )
        collector = RolloutCollector(env=pool, n_envs=128)

    Parameters
    ----------
    workers : list[str]
        List of ``host:port`` addresses for gRPC environment workers.
    envs_per_worker : int
        Number of environments each worker manages (default 64).
    obs_shape : tuple[int, ...], optional
        Observation shape per environment. Required for space metadata;
        if not provided, ``observation_space`` raises until the first
        reset populates it.
    n_actions : int, optional
        Number of discrete actions (for ``action_space`` metadata).
    """

    def __init__(
        self,
        workers: list[str],
        envs_per_worker: int = 64,
        obs_shape: tuple[int, ...] | None = None,
        n_actions: int | None = None,
    ) -> None:
        if not workers:
            raise ValueError("workers list must not be empty")
        if envs_per_worker < 1:
            raise ValueError(f"envs_per_worker must be >= 1, got {envs_per_worker}")

        self._workers = [
            _WorkerInfo(address=addr, envs=envs_per_worker) for addr in workers
        ]
        self._envs_per_worker = envs_per_worker
        self._total_envs = len(workers) * envs_per_worker
        self._obs_shape = obs_shape
        self._n_actions = n_actions
        self._connected = False

    def _require_connection(self) -> None:
        """Raise if the gRPC server is not running."""
        raise ConnectionError(
            "gRPC server not running — RemoteEnvPool requires active "
            "worker processes. Start workers with `rlox-worker` before "
            "calling step/reset."
        )

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step all remote environments.

        Returns
        -------
        dict with keys ``obs``, ``rewards``, ``terminated``, ``truncated``,
        ``terminal_obs`` — matching the ``VecEnv`` / ``GymVecEnv`` contract.

        Raises
        ------
        ConnectionError
            If the gRPC worker processes are not running.
        """
        self._require_connection()
        # Unreachable — _require_connection always raises for now
        return {}  # pragma: no cover

    def reset_all(self, seed: int | None = None) -> np.ndarray:
        """Reset all remote environments.

        Returns
        -------
        np.ndarray of shape ``(n_envs, obs_dim)`` with dtype float64.

        Raises
        ------
        ConnectionError
            If the gRPC worker processes are not running.
        """
        self._require_connection()
        # Unreachable — _require_connection always raises for now
        return np.empty(0)  # pragma: no cover

    def num_envs(self) -> int:
        """Return the total number of environments across all workers."""
        return self._total_envs

    @property
    def action_space(self) -> Any:
        """Return a placeholder action space descriptor.

        When the gRPC connection is live this will query the workers.
        For now returns a dict with available metadata.
        """
        return {"type": "discrete", "n": self._n_actions} if self._n_actions else None

    @property
    def observation_space(self) -> Any:
        """Return a placeholder observation space descriptor.

        When the gRPC connection is live this will query the workers.
        For now returns a dict with available metadata.
        """
        return {"shape": self._obs_shape} if self._obs_shape else None

    @property
    def worker_addresses(self) -> list[str]:
        """Return the list of worker addresses."""
        return [w.address for w in self._workers]

    def __repr__(self) -> str:
        return (
            f"RemoteEnvPool(workers={len(self._workers)}, "
            f"envs_per_worker={self._envs_per_worker}, "
            f"total_envs={self._total_envs})"
        )
