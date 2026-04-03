# rlox-candle

[Candle](https://github.com/huggingface/candle) backend for rlox neural network inference and training.

Implements the `rlox-nn` traits using HuggingFace's Candle framework. This is the recommended pure-Rust backend, providing CPU inference at 180K+ SPS on CartPole with zero Python overhead.

## Key Components

- `CandleActorCritic` -- discrete actor-critic with PPO step support
- `collector` -- `SharedPolicy` and `make_candle_callbacks()` for hybrid Rust-side data collection
- `dqn` -- DQN network implementation
- `stochastic` -- stochastic policy (SAC)
- `deterministic` -- deterministic policy (TD3)
- `continuous_q` -- continuous action Q-function
- `entropy` -- entropy coefficient tuning
- `mlp` -- configurable MLP builder with Tanh/ReLU activation

## Usage

```rust
use rlox_candle::actor_critic::CandleActorCritic;
use rlox_candle::mlp::MlpConfig;
use rlox_nn::{ActorCritic, TensorData};

// Build a Candle actor-critic
// let config = MlpConfig { hidden_sizes: vec![64, 64], activation: "relu" };
// let network = CandleActorCritic::new(4, 2, &config, &Device::Cpu).unwrap();
// let obs = TensorData::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4]);
// let output = network.act(&obs).unwrap();

// Hybrid collection (Rust env stepping + Candle inference)
// let collector = CandleCollector::new(network, vec_env);
// let batch = collector.collect(2048).unwrap();
```

## Part of rlox

This crate is the Candle NN backend for [rlox](https://github.com/riserally/rlox). See the main project for Python bindings, algorithms, and full documentation.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE).
