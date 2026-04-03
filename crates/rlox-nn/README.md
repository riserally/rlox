# rlox-nn

Backend-agnostic neural network traits for reinforcement learning.

## Key Types

### Traits

- `ActorCritic` -- combined actor-critic policy (PPO, A2C)
- `QFunction` -- Q-value function (DQN, SAC)
- `StochasticPolicy` -- stochastic policy with log-probability (SAC)
- `DeterministicPolicy` -- deterministic policy (TD3)
- `EntropyTuner` -- automatic entropy coefficient tuning

### Data Types

- `TensorData` -- flat `Vec<f32>` with shape metadata, safe for FFI across backends
- `ActionOutput` -- structured policy output (action, log_prob, value)
- `EvalOutput` -- evaluation output (action only)
- `PPOStepConfig`, `DQNStepConfig` -- algorithm hyperparameter structs

### Distributions

- `distributions` module -- probability distribution utilities for policy outputs

## Usage

```rust
use rlox_nn::{TensorData, ActorCritic, ActionOutput};

// TensorData wraps flat vectors with shape info
let obs = TensorData::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4]);

// Implement ActorCritic for your backend:
// impl ActorCritic for MyNetwork {
//     fn act(&self, obs: &TensorData) -> Result<ActionOutput, NNError> { ... }
//     fn evaluate(&self, obs: &TensorData) -> Result<EvalOutput, NNError> { ... }
//     fn ppo_step(&mut self, config: &PPOStepConfig) -> Result<StepInfo, NNError> { ... }
// }
```

## Part of rlox

This crate defines the NN interface for [rlox](https://github.com/riserally/rlox). Concrete implementations live in `rlox-candle` (Candle backend) and `rlox-burn` (Burn backend).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE).
