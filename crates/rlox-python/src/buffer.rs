use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rlox_core::buffer::columnar::ExperienceTable;
use rlox_core::buffer::extra_columns::ColumnHandle;
use rlox_core::buffer::mmap::MmapReplayBuffer;
use rlox_core::buffer::offline::OfflineDatasetBuffer;
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
        self.dict.set_item(key, PyArray1::from_slice(self.py, &u8s))
    }

    /// Add extra columns from the sampled batch.
    fn add_extra_columns(&self, extra: &[(String, Vec<f32>)], batch_size: usize) -> PyResult<()> {
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
            .push_slices(
                obs_slice,
                next_obs_slice,
                action_slice,
                reward,
                terminated,
                truncated,
            )
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
            .push_slices(
                obs_slice,
                next_obs_slice,
                action_slice,
                reward,
                terminated,
                truncated,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Push multiple transitions at once from flat numpy arrays.
    ///
    /// Useful for vectorized environment collection where multiple
    /// transitions are available simultaneously.
    #[pyo3(signature = (obs, next_obs, actions, rewards, terminated, truncated))]
    fn push_batch(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        next_obs: PyReadonlyArray1<f32>,
        actions: PyReadonlyArray1<f32>,
        rewards: PyReadonlyArray1<f32>,
        terminated: PyReadonlyArray1<u8>,
        truncated: PyReadonlyArray1<u8>,
    ) -> PyResult<()> {
        let term_bool: Vec<bool> = terminated.as_slice()?.iter().map(|&v| v != 0).collect();
        let trunc_bool: Vec<bool> = truncated.as_slice()?.iter().map(|&v| v != 0).collect();
        self.inner
            .push_batch(
                obs.as_slice()?,
                next_obs.as_slice()?,
                actions.as_slice()?,
                rewards.as_slice()?,
                &term_bool,
                &trunc_bool,
            )
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
        let batch = py
            .allow_threads(|| self.inner.sample(batch_size, seed))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let builder = BatchDictBuilder::new(py);
        builder.add_2d("obs", batch.observations, batch.batch_size, batch.obs_dim)?;
        builder.add_2d(
            "next_obs",
            batch.next_observations,
            batch.batch_size,
            batch.obs_dim,
        )?;
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
            .push_slices(
                obs_slice,
                next_obs_slice,
                action_slice,
                reward,
                terminated,
                truncated,
                priority,
            )
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
        let batch = py
            .allow_threads(|| self.inner.sample(batch_size, seed))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let builder = BatchDictBuilder::new(py);
        builder.add_2d("obs", batch.observations, batch.batch_size, batch.obs_dim)?;
        builder.add_2d(
            "next_obs",
            batch.next_observations,
            batch.batch_size,
            batch.obs_dim,
        )?;
        builder.add_actions(batch.actions, batch.batch_size, batch.act_dim)?;
        builder.add_1d_f32("rewards", batch.rewards)?;
        builder.add_bool("terminated", &batch.terminated)?;
        builder.add_bool("truncated", &batch.truncated)?;

        // Prioritized-specific fields
        let weights: Vec<f32> = batch.weights.iter().map(|&w| w as f32).collect();
        builder.add_1d_f32("weights", weights)?;
        let indices: Vec<u64> = batch.indices.iter().map(|&i| i as u64).collect();
        builder
            .dict
            .set_item("indices", PyArray1::from_vec(py, indices))?;

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
        let batch = py
            .allow_threads(|| self.inner.sample(batch_size, seed))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let builder = BatchDictBuilder::new(py);
        builder.add_2d("obs", batch.observations, batch.batch_size, batch.obs_dim)?;
        builder.add_2d(
            "next_obs",
            batch.next_observations,
            batch.batch_size,
            batch.obs_dim,
        )?;
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

// ---------- OfflineDatasetBuffer ----------

/// Read-only offline dataset buffer for offline RL.
///
/// Loaded once from numpy arrays or D4RL datasets. Supports uniform
/// transition sampling and trajectory subsequence sampling.
///
/// Args:
///     obs: (N, obs_dim) float32
///     next_obs: (N, obs_dim) float32
///     actions: (N, act_dim) float32
///     rewards: (N,) float32
///     terminated: (N,) bool/uint8
///     truncated: (N,) bool/uint8
///     normalize: whether to compute normalization stats (default False)
#[pyclass(name = "OfflineDatasetBuffer")]
pub struct PyOfflineDatasetBuffer {
    inner: OfflineDatasetBuffer,
}

#[pymethods]
impl PyOfflineDatasetBuffer {
    #[new]
    #[pyo3(signature = (obs, next_obs, actions, rewards, terminated, truncated, normalize=false))]
    fn new(
        obs: PyReadonlyArray1<'_, f32>,
        next_obs: PyReadonlyArray1<'_, f32>,
        actions: PyReadonlyArray1<'_, f32>,
        rewards: PyReadonlyArray1<'_, f32>,
        terminated: PyReadonlyArray1<'_, u8>,
        truncated: PyReadonlyArray1<'_, u8>,
        normalize: bool,
    ) -> PyResult<Self> {
        let rew_slice = rewards.as_slice()?;
        let obs_slice = obs.as_slice()?;
        let act_slice = actions.as_slice()?;
        let n = rew_slice.len();
        let obs_len = obs_slice.len();
        let act_len = act_slice.len();

        if n == 0 {
            return Err(PyRuntimeError::new_err(
                "Dataset must have at least 1 transition",
            ));
        }
        if obs_len % n != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "obs length {} not divisible by n_transitions {}",
                obs_len, n
            )));
        }

        let obs_dim = obs_len / n;
        let act_dim = act_len / n;

        let mut buf = OfflineDatasetBuffer::from_arrays(
            obs.as_slice()?.to_vec(),
            next_obs.as_slice()?.to_vec(),
            actions.as_slice()?.to_vec(),
            rewards.as_slice()?.to_vec(),
            terminated.as_slice()?.to_vec(),
            truncated.as_slice()?.to_vec(),
            obs_dim,
            act_dim,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if normalize {
            buf.compute_normalization();
        }

        Ok(Self { inner: buf })
    }

    /// Sample i.i.d. transitions.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = py.allow_threads(|| self.inner.sample(batch_size, seed));

        let dict = PyDict::new(py);
        let obs_dim = batch.obs_dim;
        let act_dim = batch.act_dim;

        // Return 2D arrays for obs/next_obs/actions
        let obs_arr = PyArray1::from_vec(py, batch.obs);
        let obs_2d = obs_arr
            .reshape([batch_size, obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("obs", obs_2d)?;

        let next_obs_arr = PyArray1::from_vec(py, batch.next_obs);
        let next_obs_2d = next_obs_arr
            .reshape([batch_size, obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("next_obs", next_obs_2d)?;

        if act_dim > 1 {
            let act_arr = PyArray1::from_vec(py, batch.actions);
            let act_2d = act_arr
                .reshape([batch_size, act_dim])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            dict.set_item("actions", act_2d)?;
        } else {
            dict.set_item("actions", PyArray1::from_vec(py, batch.actions))?;
        }

        dict.set_item("rewards", PyArray1::from_vec(py, batch.rewards))?;
        dict.set_item(
            "terminated",
            PyArray1::from_vec(
                py,
                batch
                    .terminated
                    .iter()
                    .map(|&x| x as f32)
                    .collect::<Vec<_>>(),
            ),
        )?;

        Ok(dict)
    }

    /// Sample contiguous trajectory subsequences (for Decision Transformer).
    #[pyo3(signature = (batch_size, seq_len, seed))]
    fn sample_trajectories<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        seq_len: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = py.allow_threads(|| self.inner.sample_trajectories(batch_size, seq_len, seed));

        let dict = PyDict::new(py);
        let obs_dim = batch.obs_dim;
        let act_dim = batch.act_dim;

        let obs_arr = PyArray1::from_vec(py, batch.obs);
        let obs_3d = obs_arr
            .reshape([batch_size, seq_len, obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("obs", obs_3d)?;

        let act_arr = PyArray1::from_vec(py, batch.actions);
        let act_3d = act_arr
            .reshape([batch_size, seq_len, act_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("actions", act_3d)?;

        let rew_arr = PyArray1::from_vec(py, batch.rewards);
        dict.set_item(
            "rewards",
            rew_arr
                .reshape([batch_size, seq_len])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )?;

        let rtg_arr = PyArray1::from_vec(py, batch.returns_to_go);
        dict.set_item(
            "returns_to_go",
            rtg_arr
                .reshape([batch_size, seq_len])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )?;

        let ts_arr = PyArray1::from_vec(py, batch.timesteps);
        dict.set_item(
            "timesteps",
            ts_arr
                .reshape([batch_size, seq_len])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )?;

        let mask_arr = PyArray1::from_vec(py, batch.mask);
        dict.set_item(
            "mask",
            mask_arr
                .reshape([batch_size, seq_len])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )?;

        Ok(dict)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of episodes in the dataset.
    fn n_episodes(&self) -> usize {
        self.inner.n_episodes()
    }

    /// Dataset statistics as a dict.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("n_transitions", s.n_transitions)?;
        dict.set_item("n_episodes", s.n_episodes)?;
        dict.set_item("obs_dim", s.obs_dim)?;
        dict.set_item("act_dim", s.act_dim)?;
        dict.set_item("mean_return", s.mean_return)?;
        dict.set_item("std_return", s.std_return)?;
        dict.set_item("min_return", s.min_return)?;
        dict.set_item("max_return", s.max_return)?;
        dict.set_item("mean_episode_length", s.mean_episode_length)?;
        Ok(dict)
    }
}
