use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use rlox_core::llm::ops;

/// Compute GRPO group advantages: (reward - mean) / std.
/// Returns zeros if std < 1e-8.
#[pyfunction]
pub fn compute_group_advantages<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = rewards.as_slice()?;
    let result = ops::compute_group_advantages(slice);
    Ok(PyArray1::from_vec(py, result))
}

/// Compute token-level KL divergence: sum(exp(log_p) * (log_p - log_q)).
#[pyfunction]
pub fn compute_token_kl(
    log_probs_policy: PyReadonlyArray1<'_, f64>,
    log_probs_ref: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    Ok(ops::compute_token_kl(policy_slice, ref_slice))
}

/// A DPO preference pair.
#[pyclass(name = "DPOPair")]
pub struct PyDPOPair {
    inner: ops::DPOPair,
}

#[pymethods]
impl PyDPOPair {
    #[new]
    #[pyo3(signature = (prompt_tokens, chosen_tokens, rejected_tokens))]
    fn new(
        prompt_tokens: PyReadonlyArray1<'_, u32>,
        chosen_tokens: PyReadonlyArray1<'_, u32>,
        rejected_tokens: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ops::DPOPair::new(
                prompt_tokens.as_slice()?.to_vec(),
                chosen_tokens.as_slice()?.to_vec(),
                rejected_tokens.as_slice()?.to_vec(),
            ),
        })
    }

    fn chosen_len(&self) -> usize {
        self.inner.chosen_len()
    }

    fn rejected_len(&self) -> usize {
        self.inner.rejected_len()
    }
}
