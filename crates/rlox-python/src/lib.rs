mod buffer;
mod env;
mod training;

use pyo3::prelude::*;

use buffer::{PyExperienceTable, PyReplayBuffer, PyVarLenStore};
use env::{PyCartPole, PyGymEnv, PyVecEnv};

#[pymodule]
fn _rlox_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCartPole>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyGymEnv>()?;
    m.add_class::<PyExperienceTable>()?;
    m.add_class::<PyReplayBuffer>()?;
    m.add_class::<PyVarLenStore>()?;
    m.add_function(wrap_pyfunction!(training::compute_gae, m)?)?;
    Ok(())
}
