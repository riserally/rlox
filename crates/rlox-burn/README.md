# rlox-burn

[Burn](https://burn.dev/) backend for rlox neural network inference and training.

Implements the `rlox-nn` traits (`ActorCritic`, `QFunction`, `StochasticPolicy`, `DeterministicPolicy`) using Burn with `Autodiff<NdArray>` and optional `wgpu` backends.

## Key Components

- `actor_critic` -- Burn-based actor-critic implementation
- `dqn` -- DQN network implementation
- `stochastic` -- stochastic policy (SAC)
- `deterministic` -- deterministic policy (TD3)
- `continuous_q` -- continuous action Q-function
- `entropy` -- entropy coefficient tuning
- `mlp` -- configurable MLP builder
- `convert` -- tensor conversion utilities between Burn and `TensorData`

## Usage

```rust
use rlox_burn::actor_critic::BurnActorCritic;
use rlox_nn::{ActorCritic, TensorData};

// Build an actor-critic network with Burn backend
// let network = BurnActorCritic::new(obs_dim, act_dim, hidden_sizes, device);
// let obs = TensorData::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4]);
// let output = network.act(&obs).unwrap();
```

## Part of rlox

This crate is the Burn NN backend for [rlox](https://github.com/wojciechkpl/rlox). The Candle alternative is in `rlox-candle`. See the main project for Python bindings and algorithms.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE).
