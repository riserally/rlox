mod env;

use pyo3::prelude::*;

use env::{PyCartPole, PyGymEnv, PyVecEnv};

#[pymodule]
fn _rlox_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCartPole>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyGymEnv>()?;
    Ok(())
}
