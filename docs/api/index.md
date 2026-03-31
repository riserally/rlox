# API Reference

Auto-generated from source code. For the Rust API, see the [cargo doc reference](/rlox/rust/rlox_core/).

## Python Modules

| Module | Description |
|--------|-------------|
| [Algorithms](algorithms.md) | PPO, SAC, DQN, TD3, A2C, MAPPO, DreamerV3, IMPALA, offline RL (TD3+BC, IQL, CQL, BC), LLM (GRPO, DPO), HybridPPO |
| [Trainers](trainers.md) | High-level trainers for all algorithms (PPOTrainer, SACTrainer, MAPPOTrainer, etc.) |
| [Config](config.md) | TrainingConfig, PPOConfig, SACConfig, DQNConfig -- configuration dataclasses and YAML loading |
| [Buffers & Primitives](buffers.md) | ReplayBuffer, PrioritizedReplayBuffer, OfflineDatasetBuffer, CandleCollector |
| [Callbacks & Logging](callbacks.md) | EvalCallback, CheckpointCallback, ProgressBarCallback, loggers |
| [Policies](policies.md) | DiscretePolicy, ContinuousPolicy, SquashedGaussianPolicy |
| [Distributed](distributed.md) | Multi-GPU training, gRPC actor workers, elastic scaling |
| [Dashboard](dashboard.md) | MetricsCollector, TerminalDashboard, HTMLReport -- training metrics visualisation |

## Rust Crates

| Crate | Description |
|-------|-------------|
| [`rlox-core`](/rlox/rust/rlox_core/) | VecEnv, replay/prioritized/offline buffers, GAE, V-trace, KL, GRPO, async pipeline |
| [`rlox-nn`](/rlox/rust/rlox_nn/) | Backend-agnostic NN traits (ActorCritic, QFunction, etc.) |
| [`rlox-candle`](/rlox/rust/rlox_candle/) | Candle backend: CandleActorCritic, SharedPolicy, hybrid collection |
