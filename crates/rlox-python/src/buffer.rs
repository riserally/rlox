use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rlox_core::buffer::columnar::ExperienceTable;
use rlox_core::buffer::extra_columns::ColumnHandle;
use rlox_core::buffer::mmap::MmapReplayBuffer;
use rlox_core::buffer::priority::PrioritizedReplayBuffer;
use rlox_core::buffer::ringbuf::ReplayBuffer;
use rlox_core::buffer::varlen::VarLenStore;
use rlox_core::buffer::ExperienceRecord;

// ---------------------------------------------------------------------------
// BatchDictBuilder — shared helper for constructing Python sample dicts
// ---------------------------------------------------------------------------

/// Builder for constructing the Python dict returned by `sample()`.
///
/// Centralises the numpy array construction that was previously duplicated
/// across `PyReplayBuffer::sample` and `PyPrioritizedReplayBuffer::sample`.
struct BatchDictBuilder<'py> {
    dict: Bound<'py, PyDict>,
    py: Python<'py>,
}

impl<'py> BatchDictBuilder<'py> {
    fn new(py: Python<'py>) -> Self {
        Self {
            dict: PyDict::new(py),
            py,
        }
    }

    /// Add a 2D f32 array (observations, next_obs).
    fn add_2d(&self, key: &str, data: Vec<f32>, rows: usize, cols: usize) -> PyResult<()> {
        let arr = PyArray1::from_vec(self.py, data);
        let arr_2d = arr
            .reshape([rows, cols])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.dict.set_item(key, arr_2d)
    }

    /// Add a 1D f32 array (rewards, 1D actions).
    fn add_1d_f32(&self, key: &str, data: Vec<f32>) -> PyResult<()> {
        self.dict.set_item(key, PyArray1::from_vec(self.py, data))
    }

    /// Add actions — 2D if act_dim > 1, 1D otherwise.
    fn add_actions(&self, data: Vec<f32>, batch_size: usize, act_dim: usize) -> PyResult<()> {
        let arr = PyArray1::from_vec(self.py, data);
        if act_dim > 1 {
            let arr_2d = arr
                .reshape([batch_size, act_dim])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.dict.set_item("actions", arr_2d)
        } else {
            self.dict.set_item("actions", arr)
        }
    }

    /// Add a 1D bool-as-u8 array (terminated, truncated).
    fn add_bool(&self, key: &str, data: &[bool]) -> PyResult<()> {
        let u8s: Vec<u8> = data.iter().map(|&b| b as u8).collect();
        self.dict
            .set_item(key, PyArray1::from_slice(self.py, &u8s))
    }

    /// Add extra columns from the sampled batch.
    fn add_extra_columns(
        &self,
        extra: &[(String, Vec<f32>)],
        batch_size: usize,
    ) -> PyResult<()> {
        for (name, data) in extra {
            let dim = data.len() / batch_size;
            let arr = PyArray1::from_slice(self.py, data);
            if dim > 1 {
                let arr_2d = arr
                    .reshape([batch_size, dim])
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                self.dict.set_item(name.as_str(), arr_2d)?;
            } else {
                self.dict.set_item(name.as_str(), arr)?;
            }
        }
        Ok(())
    }

    fn build(self) -> Bound<'py, PyDict> {
        self.dict
    }
}

// ---------- ExperienceTable ----------

#[pyclass(name = "ExperienceTable")]
pub struct PyExperienceTable {
    inner: ExperienceTable,
}

#[pymethods]
impl PyExperienceTable {
    #[new]
    fn new(obs_dim: usize, act_dim: usize) -> Self {
        Self {
            inner: ExperienceTable::new(obs_dim, act_dim),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Push a transition. ``next_obs`` is optional for on-policy use.
    #[pyo3(signature = (obs, action, reward, terminated, truncated, next_obs=None))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: PyReadonlyArray1<f32>,
        reward: f32,
        terminated: bool,
        truncated: bool,
        next_obs: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<()> {
        let obs_slice = obs.as_slice()?;
        let action_slice = action.as_slice()?;
        let zeros;
        let next_obs_slice = match &next_obs {
            Some(n) => n.as_slice()?,
            None => {
                zeros = vec![0.0f32; obs_slice.len()];
                &zeros
            }
        };
        self.inner
            .push_slices(obs_slice, next_obs_slice, action_slice, reward, terminated, truncated)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return observations as a numpy array with shape (n, obs_dim).
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let raw = self.inner.observations_raw();
        let n = self.inner.len();
        let obs_dim = self.inner.obs_dim();
        let array_1d = PyArray1::from_slice(py, raw);
        let array_2d = array_1d
            .reshape([n, obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(array_2d)
    }

    /// Return rewards as a numpy array with shape (n,).
    fn rewards<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.rewards_raw())
    }

    fn clear(&mut self) {
        self.inner.clear();
    }
}

// ---------- ReplayBuffer ----------

#[pyclass(name = "ReplayBuffer")]
pub struct PyReplayBuffer {
    inner: ReplayBuffer,
    /// Column handles stored by registration order, for PyO3 integer-based access.
    column_handles: Vec<ColumnHandle>,
}

#[pymethods]
impl PyReplayBuffer {
    #[new]
    fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            inner: ReplayBuffer::new(capacity, obs_dim, act_dim),
            column_handles: Vec::new(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Register an extra f32 column. Returns an integer handle.
    ///
    /// Example::
    ///
    ///     lp = buf.register_column("log_prob", 1)
    ///     buf.push(obs, action, reward, term, trunc, next_obs)
    ///     buf.push_extra(lp, np.array([0.5], dtype=np.float32))
    fn register_column(&mut self, name: &str, dim: usize) -> usize {
        let handle = self.inner.register_column(name, dim);
        self.column_handles.push(handle);
        handle.index()
    }

    /// Push extra column data for the most recently pushed transition.
    fn push_extra(&mut self, handle: usize, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let h = self
            .column_handles
            .get(handle)
            .ok_or_else(|| PyRuntimeError::new_err(format!("invalid column handle: {handle}")))?;
        self.inner
            .push_extra(*h, values.as_slice()?)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Push a transition. ``next_obs`` defaults to zeros if omitted.
    #[pyo3(signature = (obs, action, reward, terminated, truncated, next_obs=None))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: PyReadonlyArray1<f32>,
        reward: f32,
        terminated: bool,
        truncated: bool,
        next_obs: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<()> {
        let obs_slice = obs.as_slice()?;
        let action_slice = action.as_slice()?;
        let zeros;
        let next_obs_slice = match &next_obs {
            Some(n) => n.as_slice()?,
            None => {
                zeros = vec![0.0f32; obs_slice.len()];
                &zeros
            }
        };
        self.inner
            .push_slices(obs_slice, next_obs_slice, action_slice, reward, terminated, truncated)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Sample a batch. Returns a dict with numpy arrays.
    ///
    /// Extra columns (if registered) appear as additional keys in the dict.
    #[pyo3(signature = (batch_size, seed))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = py.allow_threads(|| {
            self.inner.sample(batch_size, seed)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let builder = BatchDictBuilder::new(py);
        builder.add_2d("obs", batch.observations, batch.batch_size, batch.obs_dim)?;
        builder.add_2d("next_obs", batch.next_observations, batch.batch_size, batch.obs_dim)?;
        builder.add_actions(batch.actions, batch.batch_size, batch.act_dim)?;
        builder.add_1d_f32("rewards", batch.rewards)?;
        builder.add_bool("terminated", &batch.terminated)?;
        builder.add_bool("truncated", &batch.truncated)?;
        if !batch.extra.is_empty() {
            builder.add_extra_columns(&batch.extra, batch.batch_size)?;
        }
        Ok(builder.build())
    }
}

// ---------- PrioritizedReplayBuffer ----------

#[pyclass(name = "PrioritizedReplayBuffer")]
pub struct PyPrioritizedReplayBuffer {
    inner: PrioritizedReplayBuffer,
}

#[pymethods]
impl PyPrioritizedReplayBuffer {
    #[new]
    #[pyo3(signature = (capacity, obs_dim, act_dim, alpha=0.6, beta=0.4))]
    fn new(capacity: usize, obs_dim: usize, act_dim: usize, alpha: f64, beta: f64) -> Self {
        Self {
            inner: PrioritizedReplayBuffer::new(capacity, obs_dim, act_dim, alpha, beta),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (obs, action, reward, terminated, truncated, next_obs=None, priority=1.0))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: PyReadonlyArray1<f32>,
        reward: f32,
        terminated: bool,
        truncated: bool,
        next_obs: Option<PyReadonlyArray1<f32>>,
        priority: f64,
    ) -> PyResult<()> {
        let obs_slice = obs.as_slice()?;
        let action_slice = action.as_slice()?;
        let zeros;
        let next_obs_slice = match &next_obs {
            Some(n) => n.as_slice()?,
            None => {
                zeros = vec![0.0f32; obs_slice.len()];
                &zeros
            }
        };
        self.inner
            .push_slices(obs_slice, next_obs_slice, action_slice, reward, terminated, truncated, priority)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Sample a batch. Returns a dict with numpy arrays + weights + indices.
    #[pyo3(signature = (batch_size, seed=42))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = py.allow_threads(|| {
            self.inner.sample(batch_size, seed)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let builder = BatchDictBuilder::new(py);
        builder.add_2d("obs", batch.observations, batch.batch_size, batch.obs_dim)?;
        builder.add_2d("next_obs", batch.next_observations, batch.batch_size, batch.obs_dim)?;
        builder.add_actions(batch.actions, batch.batch_size, batch.act_dim)?;
        builder.add_1d_f32("rewards", batch.rewards)?;
        builder.add_bool("terminated", &batch.terminated)?;
        builder.add_bool("truncated", &batch.truncated)?;

        // Prioritized-specific fields
        let weights: Vec<f32> = batch.weights.iter().map(|&w| w as f32).collect();
        builder.add_1d_f32("weights", weights)?;
        let indices: Vec<u64> = batch.indices.iter().map(|&i| i as u64).collect();
        builder.dict.set_item("indices", PyArray1::from_vec(py, indices))?;

        Ok(builder.build())
    }

    /// Update priorities for previously sampled indices.
    #[pyo3(signature = (indices, priorities))]
    fn update_priorities(
        &mut self,
        indices: PyReadonlyArray1<u64>,
        priorities: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let idx_slice = indices.as_slice()?;
        let pri_slice = priorities.as_slice()?;
        let idx_usize: Vec<usize> = idx_slice.iter().map(|&i| i as usize).collect();
        self.inner
            .update_priorities(&idx_usize, pri_slice)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn set_beta(&mut self, beta: f64) {
        self.inner.set_beta(beta);
    }
}

// ---------------------------------------------------------------------------
// MmapReplayBuffer — spills to disk for Atari-scale observations
// ---------------------------------------------------------------------------

#[pyclass(name = "MmapReplayBuffer")]
pub struct PyMmapReplayBuffer {
    inner: MmapReplayBuffer,
}

#[pymethods]
impl PyMmapReplayBuffer {
    #[new]
    #[pyo3(signature = (hot_capacity, total_capacity, obs_dim, act_dim, cold_path))]
    fn new(
        hot_capacity: usize,
        total_capacity: usize,
        obs_dim: usize,
        act_dim: usize,
        cold_path: &str,
    ) -> PyResult<Self> {
        let inner = MmapReplayBuffer::new(
            hot_capacity,
            total_capacity,
            obs_dim,
            act_dim,
            std::path::PathBuf::from(cold_path),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (obs, action, reward, terminated, truncated, next_obs=None))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: PyReadonlyArray1<f32>,
        reward: f32,
        terminated: bool,
        truncated: bool,
        next_obs: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<()> {
        let obs_vec = obs.as_slice()?.to_vec();
        let next_obs_vec = match next_obs {
            Some(n) => n.as_slice()?.to_vec(),
            None => vec![0.0; obs_vec.len()],
        };
        let record = ExperienceRecord {
            obs: obs_vec,
            next_obs: next_obs_vec,
            action: action.as_slice()?.to_vec(),
            reward,
            terminated,
            truncated,
        };
        self.inner
            .push(record)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = py.allow_threads(|| {
            self.inner.sample(batch_size, seed)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let builder = BatchDictBuilder::new(py);
        builder.add_2d("obs", batch.observations, batch.batch_size, batch.obs_dim)?;
        builder.add_2d("next_obs", batch.next_observations, batch.batch_size, batch.obs_dim)?;
        builder.add_actions(batch.actions, batch.batch_size, batch.act_dim)?;
        builder.add_1d_f32("rewards", batch.rewards)?;
        builder.add_bool("terminated", &batch.terminated)?;
        builder.add_bool("truncated", &batch.truncated)?;
        Ok(builder.build())
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner
            .close()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

// ---------- VarLenStore ----------

#[pyclass(name = "VarLenStore")]
pub struct PyVarLenStore {
    inner: VarLenStore,
}

#[pymethods]
impl PyVarLenStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: VarLenStore::new(),
        }
    }

    fn push(&mut self, seq: PyReadonlyArray1<u32>) -> PyResult<()> {
        self.inner.push(seq.as_slice()?);
        Ok(())
    }

    fn num_sequences(&self) -> usize {
        self.inner.num_sequences()
    }

    fn total_elements(&self) -> usize {
        self.inner.total_elements()
    }

    fn get<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if index >= self.inner.num_sequences() {
            return Err(PyRuntimeError::new_err("index out of range"));
        }
        Ok(PyArray1::from_slice(py, self.inner.get(index)))
    }
}
