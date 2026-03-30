mod buffer;
mod env;
mod llm;
mod nn;
mod training;

use pyo3::prelude::*;

use buffer::{
    PyExperienceTable, PyMmapReplayBuffer, PyOfflineDatasetBuffer, PyPrioritizedReplayBuffer,
    PyReplayBuffer, PyVarLenStore,
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
