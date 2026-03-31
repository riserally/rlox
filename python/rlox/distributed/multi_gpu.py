"""Multi-GPU training composition using PyTorch DDP/FSDP.

rlox does NOT reinvent distributed gradient sync -- it composes with
PyTorch's existing ``DistributedDataParallel`` and ``FullyShardedDataParallel``.
Each GPU runs its own set of rlox environments (on CPU); only the model
forward/backward runs on GPU.

Example
-------
::

    # Launch with: torchrun --nproc_per_node=4 train.py
    from rlox.distributed.multi_gpu import MultiGPUTrainer
    trainer = MultiGPUTrainer(PPOTrainer, env="HalfCheetah-v4")
    trainer.train(1_000_000)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Type

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"ddp", "fsdp"}


class MultiGPUTrainer:
    """Wraps any rlox trainer for multi-GPU training via PyTorch DDP or FSDP.

    Parameters
    ----------
    trainer_cls : type
        Trainer class (e.g. PPOTrainer, SACTrainer).
    env : str
        Gymnasium environment ID.
    config : dict, optional
        Config overrides.
    backend : str
        PyTorch distributed backend ("nccl" for GPU, "gloo" for CPU).
    strategy : str
        Parallelism strategy: ``"ddp"`` or ``"fsdp"`` (default ``"ddp"``).
    """

    def __init__(
        self,
        trainer_cls: Type,
        env: str,
        config: dict[str, Any] | None = None,
        backend: str = "nccl",
        strategy: str = "ddp",
        **kwargs: Any,
    ):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_VALID_STRATEGIES}, got {strategy!r}"
            )

        self.strategy = strategy

        if not dist.is_initialized():
            dist.init_process_group(backend)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

        # Each GPU gets its own trainer with a different seed
        cfg = config or {}
        seed = cfg.pop("seed", 42) + self.rank * 1000
        self.trainer = trainer_cls(env=env, config=cfg, seed=seed, **kwargs)

        # Wrap trainable networks
        inner = getattr(self.trainer, "algo", self.trainer)

        if strategy == "ddp":
            self._wrap_ddp(inner)
        elif strategy == "fsdp":
            self._wrap_fsdp(inner)

        # Move target networks to device WITHOUT wrapping
        for attr in (
            "critic1_target",
            "critic2_target",
            "actor_target",
            "target_network",
        ):
            net = getattr(inner, attr, None)
            if net is not None:
                setattr(inner, attr, net.to(self.device))

    def _wrap_ddp(self, inner: Any) -> None:
        """Wrap networks with DistributedDataParallel."""
        from torch.nn.parallel import DistributedDataParallel as DDP

        # On-policy: single policy network
        if hasattr(inner, "policy"):
            inner.policy = DDP(inner.policy.to(self.device), device_ids=[self.rank])

        # Off-policy (SAC/TD3): actor + twin critics
        for attr in ("actor", "critic1", "critic2", "q_network"):
            net = getattr(inner, attr, None)
            if net is not None and isinstance(net, torch.nn.Module):
                setattr(
                    inner, attr, DDP(net.to(self.device), device_ids=[self.rank])
                )

    def _wrap_fsdp(self, inner: Any) -> None:
        """Wrap networks with FullyShardedDataParallel."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # On-policy: single policy network
        if hasattr(inner, "policy"):
            inner.policy = FSDP(inner.policy.to(self.device))

        # Off-policy: actor + critics
        for attr in ("actor", "critic1", "critic2", "q_network"):
            net = getattr(inner, attr, None)
            if net is not None and isinstance(net, torch.nn.Module):
                setattr(inner, attr, FSDP(net.to(self.device)))

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run distributed training.

        Only rank 0 returns full metrics; other ranks return reduced metrics.
        """
        result = self.trainer.train(total_timesteps=total_timesteps)

        # Reduce metrics across ranks
        if dist.is_initialized() and self.world_size > 1:
            result = reduce_metrics(
                {k: torch.tensor(v, dtype=torch.float32) for k, v in result.items()}
            )
            result = {k: v.item() for k, v in result.items()}

        if not is_main_rank():
            logger.debug("Rank %d: training complete (metrics reduced to rank 0)", self.rank)

        return result


# ---------------------------------------------------------------------------
# Rank-aware helpers
# ---------------------------------------------------------------------------


def is_main_rank() -> bool:
    """Return True if this process is rank 0 or distributed is not active.

    Use this to guard logging, checkpointing, and evaluation that should
    only run once.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_metrics(
    metrics: dict[str, torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> dict[str, torch.Tensor]:
    """All-reduce a dict of scalar tensors and average across ranks.

    Parameters
    ----------
    metrics : dict[str, Tensor]
        Scalar tensors to reduce.
    op : ReduceOp
        Reduction operation (default: SUM, then divide by world_size).

    Returns
    -------
    dict[str, Tensor] with averaged values.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    reduced = {}
    for key, tensor in metrics.items():
        t = tensor.clone().detach()
        dist.all_reduce(t, op=op)
        reduced[key] = t / world_size
    return reduced


# ---------------------------------------------------------------------------
# Elastic training
# ---------------------------------------------------------------------------


def launch_elastic(
    trainer_fn: Callable[[], None],
    min_nodes: int = 1,
    max_nodes: int = 4,
    nproc_per_node: int = 1,
) -> None:
    """Launch fault-tolerant elastic training using ``torch.distributed.run``.

    This is a convenience wrapper around ``torch.distributed.launcher.api``
    for launching rlox trainers with elastic scaling support.

    Parameters
    ----------
    trainer_fn : callable
        Zero-argument function that creates and runs a trainer.
        Will be invoked once per worker process.
    min_nodes : int
        Minimum number of nodes to start training (default 1).
    max_nodes : int
        Maximum number of nodes for elastic scaling (default 4).
    nproc_per_node : int
        Number of worker processes per node (typically num GPUs).

    Raises
    ------
    ValueError
        If ``min_nodes > max_nodes``.
    RuntimeError
        If the elastic launcher is not available.
    """
    if min_nodes > max_nodes:
        raise ValueError(
            f"min_nodes ({min_nodes}) must be <= max_nodes ({max_nodes})"
        )

    if nproc_per_node < 1:
        raise ValueError(
            f"nproc_per_node must be >= 1, got {nproc_per_node}"
        )

    try:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch
    except ImportError as exc:
        raise RuntimeError(
            "Elastic training requires PyTorch >= 1.10 with "
            "torch.distributed.launcher support."
        ) from exc

    config = LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id="rlox_elastic",
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:29400",
        max_restarts=3,
        start_method="spawn",
    )

    elastic_launch(config, trainer_fn)()
