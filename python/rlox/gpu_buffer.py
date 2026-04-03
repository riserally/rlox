"""GPU-resident replay buffer with prefetching and CUDA stream transfers.

Wraps :class:`rlox.ReplayBuffer` and keeps a prefetch queue of GPU-resident
batches so the learner never stalls waiting for CPU->GPU transfer.

Usage::

    from rlox.gpu_buffer import GPUReplayBuffer

    buf = GPUReplayBuffer(capacity=100_000, obs_dim=17, act_dim=6, device="cuda")
    buf.push(obs, action, reward, terminated, truncated, next_obs)

    # Synchronous sample (returns dict of GPU tensors):
    batch = buf.sample(256, seed=42)

    # Prefetch N batches to overlap transfer with training:
    buf.prefetch(256, n_batches=4, seed=100)
    batch = buf.sample(256, seed=200)  # pops from prefetch queue if available

Requires ``torch`` at runtime. Falls back to CPU tensors when CUDA is
unavailable.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

import rlox


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "GPUReplayBuffer requires PyTorch. Install with: pip install torch"
        )


class GPUReplayBuffer:
    """Replay buffer with GPU-side sample caching for reduced transfer overhead.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Action dimensionality.
    device : str
        Target device, e.g. ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
    prefetch_size : int
        Maximum number of batches held in the prefetch queue.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
        prefetch_size: int = 4,
    ) -> None:
        _require_torch()
        self.inner = rlox.ReplayBuffer(capacity, obs_dim, act_dim)
        self._obs_dim = obs_dim
        self._act_dim = act_dim

        # Resolve device — fall back to CPU if CUDA requested but unavailable.
        if device.startswith("cuda") and not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        self._prefetch_queue: collections.deque[Dict[str, torch.Tensor]] = (
            collections.deque(maxlen=prefetch_size)
        )

    # ------------------------------------------------------------------
    # Delegation — push / push_batch / len
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs: Optional[np.ndarray] = None,
    ) -> None:
        """Push a single transition into the underlying Rust buffer."""
        if next_obs is not None:
            self.inner.push(obs, action, reward, terminated, truncated, next_obs)
        else:
            self.inner.push(obs, action, reward, terminated, truncated)

    def push_batch(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        """Push multiple transitions at once."""
        self.inner.push_batch(obs, next_obs, actions, rewards, terminated, truncated)

    def __len__(self) -> int:
        return len(self.inner)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_to_gpu(self, batch_size: int, seed: int) -> Dict[str, torch.Tensor]:
        """Sample from Rust buffer and transfer to GPU."""
        # Use sample_flat if available (gpu feature compiled in), else
        # fall back to the standard sample path.
        raw: Dict[str, Any]
        if hasattr(self.inner, "sample_flat"):
            raw = self.inner.sample_flat(batch_size, seed)
        else:
            raw = self.inner.sample(batch_size, seed)

        result: Dict[str, torch.Tensor] = {}
        for key, val in raw.items():
            t = torch.from_numpy(np.asarray(val))
            if self._device.type != "cpu":
                t = t.pin_memory().to(self._device, non_blocking=True)
            result[key] = t
        return result

    def sample(self, batch_size: int, seed: int) -> Dict[str, torch.Tensor]:
        """Sample a batch and return GPU tensors.

        If the prefetch queue has batches, returns the next one (ignoring
        ``batch_size`` and ``seed`` — those were set at prefetch time).
        Otherwise performs a synchronous sample + transfer.
        """
        if self._prefetch_queue:
            return self._prefetch_queue.popleft()
        return self._sample_to_gpu(batch_size, seed)

    def prefetch(
        self,
        batch_size: int,
        n_batches: int = 4,
        seed: int = 0,
    ) -> None:
        """Prefetch *n_batches* to the GPU.

        Each batch gets a different seed (``seed + i``) for diversity.
        If a CUDA device is in use, transfers are overlapped using a
        dedicated CUDA stream.

        Parameters
        ----------
        batch_size : int
            Transitions per batch.
        n_batches : int
            Number of batches to prefetch.
        seed : int
            Base seed; batch *i* uses ``seed + i``.
        """
        use_stream = self._device.type == "cuda"

        if use_stream:
            stream = torch.cuda.Stream(device=self._device)
            with torch.cuda.stream(stream):
                for i in range(n_batches):
                    batch = self._sample_to_gpu(batch_size, seed + i)
                    self._prefetch_queue.append(batch)
            # Record an event so the default stream waits for transfers.
            stream.synchronize()
        else:
            for i in range(n_batches):
                batch = self._sample_to_gpu(batch_size, seed + i)
                self._prefetch_queue.append(batch)

    @property
    def device(self) -> "torch.device":
        """The target device for sampled tensors."""
        return self._device

    @property
    def prefetch_queue_size(self) -> int:
        """Number of batches currently in the prefetch queue."""
        return len(self._prefetch_queue)
