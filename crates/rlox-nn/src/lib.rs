//! rlox-nn: Backend-agnostic neural network traits for reinforcement learning.
//!
//! Defines the [`ActorCritic`], [`QFunction`], [`StochasticPolicy`],
//! [`DeterministicPolicy`], and [`EntropyTuner`] traits that any NN backend
//! (Candle, Burn, or custom) can implement.
//!
//! Key types:
//! - [`TensorData`]: Flat `Vec<f32>` with shape metadata, safe for FFI.
//! - [`ActionOutput`], [`EvalOutput`]: Structured policy outputs.
//! - [`PPOStepConfig`], [`DQNStepConfig`]: Algorithm hyperparameters.

pub mod distributions;
pub mod error;
pub mod tensor_data;
pub mod traits;

pub use error::NNError;
pub use tensor_data::TensorData;
pub use traits::*;
