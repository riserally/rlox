//! rlox-burn: Burn backend for neural network inference and training.
//!
//! Implements the [`rlox_nn`] traits (`ActorCritic`, `QFunction`,
//! `StochasticPolicy`, `DeterministicPolicy`, `EntropyTuner`) using
//! [Burn](https://burn.dev/) with `Autodiff<NdArray>` and optional `wgpu` backends.
//!
//! Key modules:
//! - [`actor_critic`]: Combined actor-critic networks (PPO, A2C).
//! - [`dqn`]: DQN Q-network implementation.
//! - [`stochastic`]: Stochastic policy for SAC.
//! - [`deterministic`]: Deterministic policy for TD3.
//! - [`mlp`]: Configurable MLP builder with activation selection.

pub mod actor_critic;
pub mod continuous_q;
pub mod convert;
pub mod deterministic;
pub mod dqn;
pub mod entropy;
pub mod mlp;
pub mod stochastic;
mod training;
