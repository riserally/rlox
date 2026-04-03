//! rlox-python: PyO3 bindings exposing rlox Rust primitives to Python.
//!
//! Builds the `_rlox_core` native extension module that powers the `rlox`
//! Python package. All compute-heavy operations release the GIL.
//!
//! Exposed functionality:
//! - **Environments**: `CartPole`, `VecEnv`, `GymEnv`
//! - **Buffers**: `ReplayBuffer`, `PrioritizedReplayBuffer`, `MmapReplayBuffer`,
//!   `SequenceReplayBuffer`, `HERBuffer`, `OfflineDatasetBuffer`, and more.
//! - **Training ops**: `compute_gae`, `compute_vtrace`, `shape_rewards_pbrs`,
//!   `reptile_update`, `polyak_update`, `RunningStats`.
//! - **LLM ops**: `compute_group_advantages`, `compute_token_kl`, `pack_sequences`.
//! - **NN**: `ActorCritic` (Candle), `CandleCollector` for hybrid Rust collection.
//! - **Pipeline**: Async rollout collector with `Pipeline` and `RolloutBatch`.

mod buffer;
mod env;
mod llm;
mod nn;
mod training;

use pyo3::prelude::*;

use buffer::{
    PyEpisodeTracker, PyExperienceTable, PyHERBuffer, PyMmapReplayBuffer,
    PyOfflineDatasetBuffer, PyPrioritizedReplayBuffer, PyReplayBuffer,
    PySequenceReplayBuffer, PyVarLenStore,
};
use env::{PyCartPole, PyGymEnv, PyVecEnv};
use llm::PyDPOPair;
use nn::{PyActorCritic, PyCandleCollector};
use training::{PyPipeline, PyRolloutBatch, PyRunningStats, PyRunningStatsVec};

#[pymodule]
fn _rlox_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCartPole>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyGymEnv>()?;
    m.add_class::<PyExperienceTable>()?;
    m.add_class::<PyReplayBuffer>()?;
    m.add_class::<PyPrioritizedReplayBuffer>()?;
    m.add_class::<PyMmapReplayBuffer>()?;
    m.add_class::<PyOfflineDatasetBuffer>()?;
    m.add_class::<PyVarLenStore>()?;
    m.add_class::<PyEpisodeTracker>()?;
    m.add_class::<PySequenceReplayBuffer>()?;
    m.add_class::<PyHERBuffer>()?;
    m.add_class::<PyDPOPair>()?;
    m.add_class::<PyRunningStats>()?;
    m.add_class::<PyRunningStatsVec>()?;
    m.add_class::<PyActorCritic>()?;
    m.add_class::<PyCandleCollector>()?;
    m.add_class::<PyRolloutBatch>()?;
    m.add_class::<PyPipeline>()?;
    m.add_function(wrap_pyfunction!(training::compute_gae, m)?)?;
    m.add_function(wrap_pyfunction!(training::compute_gae_batched, m)?)?;
    m.add_function(wrap_pyfunction!(training::compute_gae_batched_f32, m)?)?;
    m.add_function(wrap_pyfunction!(training::compute_vtrace, m)?)?;
    m.add_function(wrap_pyfunction!(training::pack_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(training::random_shift_batch, m)?)?;
    m.add_function(wrap_pyfunction!(training::shape_rewards_pbrs, m)?)?;
    m.add_function(wrap_pyfunction!(
        training::compute_goal_distance_potentials,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(training::reptile_update, m)?)?;
    m.add_function(wrap_pyfunction!(training::polyak_update, m)?)?;
    m.add_function(wrap_pyfunction!(training::average_weight_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(buffer::py_sample_mixed, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_group_advantages, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_token_kl, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_batch_group_advantages, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_token_kl_schulman, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_batch_token_kl, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_batch_token_kl_schulman, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_token_kl_f32, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_token_kl_schulman_f32, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_batch_token_kl_f32, m)?)?;
    m.add_function(wrap_pyfunction!(
        llm::compute_batch_token_kl_schulman_f32,
        m
    )?)?;
    Ok(())
}
