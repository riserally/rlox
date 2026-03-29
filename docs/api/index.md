# API Reference

Auto-generated from source code. For the Rust API, see the [cargo doc reference](/rlox/rust/rlox_core/).

## Python Modules

| Module | Description |
|--------|-------------|
| [Algorithms](algorithms.md) | PPO, SAC, DQN, TD3, offline RL (TD3+BC, IQL, CQL, BC), LLM (GRPO, DPO), HybridPPO |
| [Buffers & Primitives](buffers.md) | ReplayBuffer, PrioritizedReplayBuffer, OfflineDatasetBuffer, CandleCollector |
| [Callbacks & Logging](callbacks.md) | EvalCallback, CheckpointCallback, ProgressBarCallback, loggers |
| [Policies](policies.md) | DiscretePolicy, ContinuousPolicy, SquashedGaussianPolicy |

## Rust Crates

| Crate | Description |
|-------|-------------|
| [`rlox-core`](/rlox/rust/rlox_core/) | VecEnv, replay/prioritized/offline buffers, GAE, V-trace, KL, GRPO, async pipeline |
| [`rlox-nn`](/rlox/rust/rlox_nn/) | Backend-agnostic NN traits (ActorCritic, QFunction, etc.) |
| [`rlox-candle`](/rlox/rust/rlox_candle/) | Candle backend: CandleActorCritic, SharedPolicy, hybrid collection |
