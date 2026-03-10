use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use rlox_core::training::gae;

/// Compute Generalized Advantage Estimation (GAE).
///
/// Args:
///     rewards: 1-D f64 array of rewards
///     values: 1-D f64 array of value estimates
///     dones: 1-D array (bool or f64; 0.0/False = not done, 1.0/True = done)
///     last_value: bootstrap value for the last step
///     gamma: discount factor
///     lam: GAE lambda parameter
///
/// Returns:
///     (advantages, returns) as a tuple of two numpy f64 arrays
#[pyfunction]
#[pyo3(signature = (rewards, values, dones, last_value, gamma, lam))]
pub fn compute_gae<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    dones: &Bound<'py, pyo3::types::PyAny>,
    last_value: f64,
    gamma: f64,
    lam: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let rewards_slice = rewards.as_slice()?;
    let values_slice = values.as_slice()?;

    // Accept dones as either f64 array or bool array
    let dones_vec: Vec<f64> = if let Ok(arr) = dones.extract::<PyReadonlyArray1<'py, f64>>() {
        arr.as_slice()?.to_vec()
    } else if let Ok(arr) = dones.extract::<PyReadonlyArray1<'py, bool>>() {
        arr.as_slice()?.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
    } else {
        // Try casting via numpy .astype(float)
        let np_arr: &Bound<'py, pyo3::types::PyAny> = dones;
        let float_arr = np_arr
            .call_method1("astype", ("float64",))
            .map_err(|_| PyTypeError::new_err("dones must be a numpy array of float64 or bool"))?;
        let readonly: PyReadonlyArray1<'py, f64> = float_arr.extract()?;
        readonly.as_slice()?.to_vec()
    };

    let (advantages, returns) =
        gae::compute_gae(rewards_slice, values_slice, &dones_vec, last_value, gamma, lam);

    Ok((
        PyArray1::from_vec(py, advantages),
        PyArray1::from_vec(py, returns),
    ))
}
