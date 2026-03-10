mod buffer;
mod env;
mod llm;
mod training;

use pyo3::prelude::*;

use buffer::{PyExperienceTable, PyReplayBuffer, PyVarLenStore};
use env::{PyCartPole, PyGymEnv, PyVecEnv};
use llm::PyDPOPair;

#[pymodule]
fn _rlox_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCartPole>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyGymEnv>()?;
    m.add_class::<PyExperienceTable>()?;
    m.add_class::<PyReplayBuffer>()?;
    m.add_class::<PyVarLenStore>()?;
    m.add_class::<PyDPOPair>()?;
    m.add_function(wrap_pyfunction!(training::compute_gae, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_group_advantages, m)?)?;
    m.add_function(wrap_pyfunction!(llm::compute_token_kl, m)?)?;
    Ok(())
}
