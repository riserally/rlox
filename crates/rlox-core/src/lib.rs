//! rlox-core: Rust-accelerated reinforcement learning primitives.
//!
//! This crate provides the high-performance data plane for rlox:
//!
//! - **Buffers** ([`buffer`]): Ring buffer, prioritized replay (SumTree),
//!   memory-mapped replay, offline dataset buffer, columnar storage.
//! - **Environments** ([`env`]): Native CartPole, vectorized stepping (Rayon),
//!   Gymnasium wrapper via PyO3.
//! - **Training** ([`training`]): GAE (single + batched), V-trace, expectile loss.
//! - **LLM ops** ([`llm`]): Token-level KL divergence, GRPO group advantages,
//!   sequence packing, DPO pairs — all with f32/f64 variants and Rayon parallelism.
//! - **Pipeline** ([`pipeline`]): Async rollout collector with crossbeam channels,
//!   backpressure, and flat `RolloutBatch` format.
//!
//! All operations release the GIL when called from Python via PyO3.

pub mod buffer;
pub mod env;
pub mod error;
pub mod llm;
pub mod pipeline;
pub mod seed;
pub mod training;
