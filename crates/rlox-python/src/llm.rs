use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use rlox_core::llm::ops;

// ---------------------------------------------------------------------------
// f64 functions (backward-compatible)
// ---------------------------------------------------------------------------

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
    ops::compute_token_kl(policy_slice, ref_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Compute batched GRPO group advantages for all groups in a single call.
/// `rewards` is a flat array of shape (n_prompts * group_size,).
#[pyfunction]
pub fn compute_batch_group_advantages<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    group_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = rewards.as_slice()?;
    let result = ops::compute_batch_group_advantages(slice, group_size)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Compute token-level KL divergence using Schulman (2020) estimator.
#[pyfunction]
pub fn compute_token_kl_schulman(
    log_probs_policy: PyReadonlyArray1<'_, f64>,
    log_probs_ref: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    ops::compute_token_kl_schulman(policy_slice, ref_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Batched token-level KL divergence: process all sequences in a single call.
///
/// `log_probs_policy` and `log_probs_ref` are flat arrays of shape (batch * seq_len,).
/// Returns an array of shape (batch,) with per-sequence KL values.
#[pyfunction]
pub fn compute_batch_token_kl<'py>(
    py: Python<'py>,
    log_probs_policy: PyReadonlyArray1<'py, f64>,
    log_probs_ref: PyReadonlyArray1<'py, f64>,
    seq_len: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    let result = ops::compute_batch_token_kl(policy_slice, ref_slice, seq_len)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Batched token-level KL divergence using the Schulman (2020) estimator.
///
/// `log_probs_policy` and `log_probs_ref` are flat arrays of shape (batch * seq_len,).
/// Returns an array of shape (batch,) with per-sequence KL values.
#[pyfunction]
pub fn compute_batch_token_kl_schulman<'py>(
    py: Python<'py>,
    log_probs_policy: PyReadonlyArray1<'py, f64>,
    log_probs_ref: PyReadonlyArray1<'py, f64>,
    seq_len: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    let result = ops::compute_batch_token_kl_schulman(policy_slice, ref_slice, seq_len)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

// ---------------------------------------------------------------------------
// f32 functions
// ---------------------------------------------------------------------------

/// Compute token-level KL divergence (f32): sum(exp(log_p) * (log_p - log_q)).
#[pyfunction]
pub fn compute_token_kl_f32(
    log_probs_policy: PyReadonlyArray1<'_, f32>,
    log_probs_ref: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    ops::f32_ops::compute_token_kl(policy_slice, ref_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Compute token-level KL divergence using Schulman estimator (f32).
#[pyfunction]
pub fn compute_token_kl_schulman_f32(
    log_probs_policy: PyReadonlyArray1<'_, f32>,
    log_probs_ref: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    ops::f32_ops::compute_token_kl_schulman(policy_slice, ref_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Batched token-level KL divergence (f32).
#[pyfunction]
pub fn compute_batch_token_kl_f32<'py>(
    py: Python<'py>,
    log_probs_policy: PyReadonlyArray1<'py, f32>,
    log_probs_ref: PyReadonlyArray1<'py, f32>,
    seq_len: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    let result = ops::f32_ops::compute_batch_token_kl(policy_slice, ref_slice, seq_len)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

/// Batched token-level KL divergence using Schulman estimator (f32).
#[pyfunction]
pub fn compute_batch_token_kl_schulman_f32<'py>(
    py: Python<'py>,
    log_probs_policy: PyReadonlyArray1<'py, f32>,
    log_probs_ref: PyReadonlyArray1<'py, f32>,
    seq_len: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let policy_slice = log_probs_policy.as_slice()?;
    let ref_slice = log_probs_ref.as_slice()?;
    let result = ops::f32_ops::compute_batch_token_kl_schulman(policy_slice, ref_slice, seq_len)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, result))
}

// ---------------------------------------------------------------------------
// DPOPair
// ---------------------------------------------------------------------------

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
