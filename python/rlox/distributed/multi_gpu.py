"""Multi-GPU training composition using PyTorch DDP/FSDP.

rlox does NOT reinvent distributed gradient sync — it composes with
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

from typing import Any, Type


class MultiGPUTrainer:
    """Wraps any rlox trainer for multi-GPU training via PyTorch DDP.

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
    """

    def __init__(
        self,
        trainer_cls: Type,
        env: str,
        config: dict[str, Any] | None = None,
        backend: str = "nccl",
        **kwargs: Any,
    ):
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

        # Each GPU gets its own trainer with a different seed
        cfg = config or {}
        seed = cfg.pop("seed", 42) + self.rank * 1000
        self.trainer = trainer_cls(env=env, config=cfg, seed=seed, **kwargs)

        # Wrap the policy with DDP
        if hasattr(self.trainer, "algo") and hasattr(self.trainer.algo, "policy"):
            policy = self.trainer.algo.policy.to(self.device)
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.trainer.algo.policy = DDP(policy, device_ids=[self.rank])
        elif hasattr(self.trainer, "policy"):
            policy = self.trainer.policy.to(self.device)
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.trainer.policy = DDP(policy, device_ids=[self.rank])

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run distributed training."""
        return self.trainer.train(total_timesteps=total_timesteps)
