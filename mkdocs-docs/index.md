# rlox Python API

Rust-accelerated reinforcement learning. The Polars architecture pattern applied to RL: Rust data plane for environments, buffers, and advantage computation; Python control plane for training logic, policies, and neural networks.

## Quick Start

```python
import rlox
from rlox.trainers import PPOTrainer

trainer = PPOTrainer(env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

## Modules

| Module | Description |
|--------|-------------|
| [`rlox` (core)](api/core.md) | Rust primitives via PyO3: environments, buffers, GAE, LLM ops |
| [`rlox.algorithms`](api/algorithms.md) | Algorithm implementations: PPO, SAC, DQN, TD3, A2C, GRPO, DPO, and more |
| [`rlox.trainers`](api/trainers.md) | High-level trainer API with sensible defaults |
| [`rlox.policies` / `rlox.networks`](api/policies.md) | Policy networks and shared architectures |
| [`rlox.callbacks`](api/callbacks.md) | Callback system for monitoring, evaluation, and checkpointing |
| [`rlox.logging`](api/logging.md) | Logging backends: W&B, TensorBoard |
| [`rlox.evaluation`](api/evaluation.md) | Statistical evaluation utilities (IQM, performance profiles, bootstrap CI) |
| [`rlox.llm`](api/llm.md) | LLM environment, reward model serving |
| [`rlox.distributed`](api/distributed.md) | Distributed training and inference backends |
| [`rlox.config`](api/config.md) | Configuration dataclasses with validation and serialization |
