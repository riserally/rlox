use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rlox_core::pipeline::channel::{Pipeline, RolloutBatch};
use rlox_core::training::gae;
use rlox_core::training::normalization::RunningStats;
use rlox_core::training::packing;
use rlox_core::training::vtrace;

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

/// Python-facing RunningStats (Welford's algorithm).
#[pyclass(name = "RunningStats")]
pub struct PyRunningStats {
    inner: RunningStats,
}

#[pymethods]
impl PyRunningStats {
    #[new]
    fn new() -> Self {
        Self {
            inner: RunningStats::new(),
        }
    }

    fn update(&mut self, value: f64) {
        self.inner.update(value);
    }

    fn batch_update(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.inner.batch_update(values.as_slice()?);
        Ok(())
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn var(&self) -> f64 {
        self.inner.var()
    }

    fn std(&self) -> f64 {
        self.inner.std()
    }

    fn normalize(&self, value: f64) -> f64 {
        self.inner.normalize(value)
    }

    fn count(&self) -> u64 {
        self.inner.count()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Pack variable-length sequences into fixed-size bins (first-fit-decreasing).
///
/// Args:
///     sequences: list of 1-D uint32 numpy arrays
///     max_length: maximum bin length
///
/// Returns:
///     list of dicts, each with keys: input_ids, attention_mask, position_ids, sequence_starts
#[pyfunction]
#[pyo3(signature = (sequences, max_length))]
pub fn pack_sequences<'py>(
    py: Python<'py>,
    sequences: Vec<PyReadonlyArray1<'py, u32>>,
    max_length: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let vecs: Vec<Vec<u32>> = sequences
        .iter()
        .map(|arr| arr.as_slice().map(|s| s.to_vec()))
        .collect::<Result<_, _>>()?;
    let slices: Vec<&[u32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let packed = packing::pack_sequences(&slices, max_length)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut result = Vec::with_capacity(packed.len());
    for batch in packed {
        let dict = PyDict::new(py);
        dict.set_item("input_ids", PyArray1::from_vec(py, batch.input_ids))?;
        dict.set_item(
            "attention_mask",
            PyArray1::from_vec(py, batch.attention_mask),
        )?;
        dict.set_item("position_ids", PyArray1::from_vec(py, batch.position_ids))?;
        dict.set_item(
            "sequence_starts",
            PyArray1::from_vec(py, batch.sequence_starts.into_iter().map(|s| s as u64).collect()),
        )?;
        result.push(dict);
    }
    Ok(result)
}

/// Compute V-trace targets and policy gradient advantages (Espeholt et al. 2018).
///
/// Args:
///     log_rhos: 1-D f32 array of log importance ratios log(pi/mu)
///     rewards: 1-D f32 array of rewards
///     values: 1-D f32 array of value estimates
///     bootstrap_value: bootstrap value for the last step
///     gamma: discount factor
///     rho_bar: clipping threshold for importance weights (default 1.0)
///     c_bar: trace cutting threshold (default 1.0)
///
/// Returns:
///     (vs, pg_advantages) as a tuple of two numpy f32 arrays
#[pyfunction]
#[pyo3(signature = (log_rhos, rewards, values, bootstrap_value, gamma, rho_bar=1.0, c_bar=1.0))]
pub fn compute_vtrace<'py>(
    py: Python<'py>,
    log_rhos: PyReadonlyArray1<'py, f32>,
    rewards: PyReadonlyArray1<'py, f32>,
    values: PyReadonlyArray1<'py, f32>,
    bootstrap_value: f32,
    gamma: f32,
    rho_bar: f32,
    c_bar: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let (vs, pg_advantages) = vtrace::compute_vtrace(
        log_rhos.as_slice()?,
        rewards.as_slice()?,
        values.as_slice()?,
        bootstrap_value,
        gamma,
        rho_bar,
        c_bar,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray1::from_vec(py, vs),
        PyArray1::from_vec(py, pg_advantages),
    ))
}

/// Python-facing RolloutBatch — a flat batch of rollout data.
///
/// All arrays are 1-D numpy arrays. Shape metadata (obs_dim, act_dim, n_steps,
/// n_envs) is stored as scalar attributes so the Python side can reshape.
#[pyclass(name = "RolloutBatch")]
pub struct PyRolloutBatch {
    inner: RolloutBatch,
}

#[pymethods]
impl PyRolloutBatch {
    /// Create a new RolloutBatch from flat numpy arrays and shape metadata.
    #[new]
    #[pyo3(signature = (observations, actions, rewards, dones, advantages, returns, obs_dim, act_dim, n_steps, n_envs))]
    fn new(
        observations: PyReadonlyArray1<'_, f32>,
        actions: PyReadonlyArray1<'_, f32>,
        rewards: PyReadonlyArray1<'_, f64>,
        dones: PyReadonlyArray1<'_, f64>,
        advantages: PyReadonlyArray1<'_, f64>,
        returns: PyReadonlyArray1<'_, f64>,
        obs_dim: usize,
        act_dim: usize,
        n_steps: usize,
        n_envs: usize,
    ) -> PyResult<Self> {
        let obs_slice = observations.as_slice()?;
        let act_slice = actions.as_slice()?;
        let rew_slice = rewards.as_slice()?;
        let don_slice = dones.as_slice()?;
        let adv_slice = advantages.as_slice()?;
        let ret_slice = returns.as_slice()?;

        let expected_obs = n_steps * n_envs * obs_dim;
        let expected_act = n_steps * n_envs * act_dim;
        let expected_flat = n_steps * n_envs;

        if obs_slice.len() != expected_obs {
            return Err(PyValueError::new_err(format!(
                "observations length {} != n_steps*n_envs*obs_dim={}",
                obs_slice.len(),
                expected_obs
            )));
        }
        if act_slice.len() != expected_act {
            return Err(PyValueError::new_err(format!(
                "actions length {} != n_steps*n_envs*act_dim={}",
                act_slice.len(),
                expected_act
            )));
        }
        if rew_slice.len() != expected_flat {
            return Err(PyValueError::new_err(format!(
                "rewards length {} != n_steps*n_envs={}",
                rew_slice.len(),
                expected_flat
            )));
        }

        Ok(Self {
            inner: RolloutBatch {
                observations: obs_slice.to_vec(),
                actions: act_slice.to_vec(),
                rewards: rew_slice.to_vec(),
                dones: don_slice.to_vec(),
                advantages: adv_slice.to_vec(),
                returns: ret_slice.to_vec(),
                obs_dim,
                act_dim,
                n_steps,
                n_envs,
            },
        })
    }

    #[getter]
    fn observations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.observations)
    }

    #[getter]
    fn actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.actions)
    }

    #[getter]
    fn rewards<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.rewards)
    }

    #[getter]
    fn dones<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.dones)
    }

    #[getter]
    fn advantages<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.advantages)
    }

    #[getter]
    fn returns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.returns)
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim
    }

    #[getter]
    fn act_dim(&self) -> usize {
        self.inner.act_dim
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.inner.n_envs
    }
}

/// Bounded experience pipeline for decoupled collection and training.
///
/// Wraps a crossbeam bounded channel. The collector side calls `send()`,
/// the learner side calls `recv()` or `try_recv()`.
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    inner: Pipeline,
}

#[pymethods]
impl PyPipeline {
    /// Create a new pipeline with the given buffer capacity.
    #[new]
    #[pyo3(signature = (capacity=4))]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be >= 1"));
        }
        Ok(Self {
            inner: Pipeline::new(capacity),
        })
    }

    /// Send a batch into the pipeline (blocks if full).
    fn send(&self, batch: &PyRolloutBatch) -> PyResult<()> {
        self.inner
            .send(batch.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Try to receive a batch without blocking. Returns None if empty.
    fn try_recv(&self) -> Option<PyRolloutBatch> {
        self.inner.try_recv().map(|b| PyRolloutBatch { inner: b })
    }

    /// Receive a batch, blocking until one is available.
    fn recv(&self) -> PyResult<PyRolloutBatch> {
        self.inner
            .recv()
            .map(|b| PyRolloutBatch { inner: b })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Number of batches currently buffered in the channel.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the channel is currently empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
