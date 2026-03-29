//! rlox-candle: Candle backend for pure-Rust neural network inference and training.
//!
//! Implements [`rlox_nn::ActorCritic`] and other NN traits using
//! [Candle](https://github.com/huggingface/candle) (HuggingFace's Rust ML framework).
//!
//! Key components:
//! - [`actor_critic::CandleActorCritic`]: Discrete actor-critic with PPO step.
//! - [`collector`]: `SharedPolicy` + `make_candle_callbacks()` for hybrid
//!   Rust collection (180K+ SPS on CartPole, zero Python overhead).
//! - [`dqn`], [`stochastic`], [`deterministic`]: SAC/TD3/DQN network implementations.
//! - [`mlp`]: Configurable MLP builder with Tanh/ReLU activation.

pub mod actor_critic;
pub mod collector;
pub mod continuous_q;
pub mod convert;
pub mod deterministic;
pub mod dqn;
pub mod entropy;
pub mod mlp;
pub mod stochastic;
mod training;
