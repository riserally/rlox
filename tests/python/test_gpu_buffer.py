"""Tests for the GPUReplayBuffer Python wrapper.

All tests use device='cpu' so they run on any machine without a GPU.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlox.gpu_buffer import GPUReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACT_DIM = 2
CAPACITY = 200


def _make_buffer(device: str = "cpu", prefetch_size: int = 4) -> GPUReplayBuffer:
    return GPUReplayBuffer(
        capacity=CAPACITY,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        device=device,
        prefetch_size=prefetch_size,
    )


def _fill_buffer(buf: GPUReplayBuffer, n: int = 100) -> None:
    for i in range(n):
        obs = np.full(OBS_DIM, float(i), dtype=np.float32)
        next_obs = np.full(OBS_DIM, float(i + 1), dtype=np.float32)
        action = np.array([0.1 * i, -0.1 * i], dtype=np.float32)
        buf.push(obs, action, float(i), i % 5 == 0, i % 7 == 0, next_obs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGPUBufferConstructs:
    """Construction and basic properties."""

    def test_constructs_cpu(self):
        buf = _make_buffer(device="cpu")
        assert buf.device == torch.device("cpu")
        assert len(buf) == 0

    def test_constructs_falls_back_to_cpu_when_no_cuda(self):
        """If CUDA is not available, requesting 'cuda' falls back to CPU."""
        buf = _make_buffer(device="cuda")
        if not torch.cuda.is_available():
            assert buf.device == torch.device("cpu")
        else:
            assert buf.device.type == "cuda"

    def test_len_increases_after_push(self):
        buf = _make_buffer()
        assert len(buf) == 0
        _fill_buffer(buf, n=10)
        assert len(buf) == 10


class TestGPUBufferPushSample:
    """Push data and verify sample returns correct tensors."""

    def test_sample_returns_tensors(self):
        buf = _make_buffer()
        _fill_buffer(buf, n=50)
        batch = buf.sample(16, seed=42)

        assert isinstance(batch, dict)
        assert "obs" in batch
        assert "next_obs" in batch
        assert "actions" in batch
        assert "rewards" in batch

        assert isinstance(batch["obs"], torch.Tensor)
        assert batch["obs"].shape == (16, OBS_DIM)
        assert batch["next_obs"].shape == (16, OBS_DIM)
        assert batch["rewards"].shape == (16,)

    def test_sample_actions_shape(self):
        buf = _make_buffer()
        _fill_buffer(buf, n=50)
        batch = buf.sample(8, seed=0)
        # act_dim=2 -> actions should be 2D
        assert batch["actions"].shape == (8, ACT_DIM)

    def test_sample_dtype_is_float(self):
        buf = _make_buffer()
        _fill_buffer(buf, n=50)
        batch = buf.sample(8, seed=0)
        for key in ("obs", "next_obs", "rewards"):
            assert batch[key].dtype == torch.float32, f"{key} not float32"

    def test_sample_device_is_correct(self):
        buf = _make_buffer(device="cpu")
        _fill_buffer(buf, n=50)
        batch = buf.sample(8, seed=0)
        for key, tensor in batch.items():
            assert tensor.device == torch.device("cpu"), f"{key} on wrong device"


class TestGPUBufferPrefetch:
    """Prefetch fills the queue and sample pops from it."""

    def test_prefetch_fills_queue(self):
        buf = _make_buffer(device="cpu", prefetch_size=8)
        _fill_buffer(buf, n=50)

        assert buf.prefetch_queue_size == 0
        buf.prefetch(batch_size=8, n_batches=4, seed=0)
        assert buf.prefetch_queue_size == 4

    def test_sample_pops_from_prefetch(self):
        buf = _make_buffer(device="cpu", prefetch_size=8)
        _fill_buffer(buf, n=50)

        buf.prefetch(batch_size=8, n_batches=3, seed=0)
        assert buf.prefetch_queue_size == 3

        batch = buf.sample(8, seed=999)  # seed ignored, pops from queue
        assert isinstance(batch, dict)
        assert buf.prefetch_queue_size == 2

    def test_prefetch_respects_maxlen(self):
        buf = _make_buffer(device="cpu", prefetch_size=2)
        _fill_buffer(buf, n=50)

        buf.prefetch(batch_size=8, n_batches=5, seed=0)
        # deque maxlen=2, so only last 2 survive
        assert buf.prefetch_queue_size == 2

    def test_sample_falls_through_when_queue_empty(self):
        buf = _make_buffer(device="cpu")
        _fill_buffer(buf, n=50)

        assert buf.prefetch_queue_size == 0
        batch = buf.sample(8, seed=42)
        assert isinstance(batch, dict)
        assert "obs" in batch


class TestGPUBufferFallsBackToCPU:
    """The buffer works seamlessly without CUDA."""

    def test_works_without_cuda(self):
        buf = _make_buffer(device="cpu")
        _fill_buffer(buf, n=30)

        batch = buf.sample(8, seed=42)
        assert batch["obs"].device == torch.device("cpu")
        assert batch["obs"].shape == (8, OBS_DIM)

    def test_prefetch_works_on_cpu(self):
        buf = _make_buffer(device="cpu")
        _fill_buffer(buf, n=30)

        buf.prefetch(batch_size=4, n_batches=2, seed=0)
        batch = buf.sample(4, seed=0)
        assert batch["obs"].device == torch.device("cpu")
