use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rlox_core::buffer::columnar::ExperienceTable;
use rlox_core::buffer::priority::PrioritizedReplayBuffer;
use rlox_core::buffer::ringbuf::ReplayBuffer;
use rlox_core::buffer::varlen::VarLenStore;
use rlox_core::buffer::ExperienceRecord;

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
}

#[pymethods]
impl PyReplayBuffer {
    #[new]
    fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            inner: ReplayBuffer::new(capacity, obs_dim, act_dim),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
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

    /// Sample a batch. Returns a dict with numpy arrays.
    #[pyo3(signature = (batch_size, seed))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = self
            .inner
            .sample(batch_size, seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        let obs_1d = PyArray1::from_vec(py, batch.observations);
        let obs_2d = obs_1d
            .reshape([batch.batch_size, batch.obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("obs", obs_2d)?;
        let next_obs_1d = PyArray1::from_vec(py, batch.next_observations);
        let next_obs_2d = next_obs_1d
            .reshape([batch.batch_size, batch.obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("next_obs", next_obs_2d)?;
        let act_1d = PyArray1::from_vec(py, batch.actions);
        if batch.act_dim > 1 {
            let act_2d = act_1d
                .reshape([batch.batch_size, batch.act_dim])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            dict.set_item("actions", act_2d)?;
        } else {
            dict.set_item("actions", act_1d)?;
        }
        dict.set_item("rewards", PyArray1::from_vec(py, batch.rewards))?;

        let term: Vec<u8> = batch.terminated.iter().map(|&b| b as u8).collect();
        let trunc: Vec<u8> = batch.truncated.iter().map(|&b| b as u8).collect();
        dict.set_item("terminated", PyArray1::from_slice(py, &term))?;
        dict.set_item("truncated", PyArray1::from_slice(py, &trunc))?;
        Ok(dict)
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
            .push(record, priority)
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
        let batch = self
            .inner
            .sample(batch_size, seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        let obs_1d = PyArray1::from_vec(py, batch.observations);
        let obs_2d = obs_1d
            .reshape([batch.batch_size, batch.obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("obs", obs_2d)?;
        let next_obs_1d = PyArray1::from_vec(py, batch.next_observations);
        let next_obs_2d = next_obs_1d
            .reshape([batch.batch_size, batch.obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("next_obs", next_obs_2d)?;
        let act_1d = PyArray1::from_vec(py, batch.actions);
        if batch.act_dim > 1 {
            let act_2d = act_1d
                .reshape([batch.batch_size, batch.act_dim])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            dict.set_item("actions", act_2d)?;
        } else {
            dict.set_item("actions", act_1d)?;
        }
        dict.set_item("rewards", PyArray1::from_vec(py, batch.rewards))?;

        let term: Vec<u8> = batch.terminated.iter().map(|&b| b as u8).collect();
        let trunc: Vec<u8> = batch.truncated.iter().map(|&b| b as u8).collect();
        dict.set_item("terminated", PyArray1::from_slice(py, &term))?;
        dict.set_item("truncated", PyArray1::from_slice(py, &trunc))?;

        // Prioritized-specific fields
        let weights: Vec<f32> = batch.weights.iter().map(|&w| w as f32).collect();
        dict.set_item("weights", PyArray1::from_vec(py, weights))?;
        let indices: Vec<u64> = batch.indices.iter().map(|&i| i as u64).collect();
        dict.set_item("indices", PyArray1::from_vec(py, indices))?;

        Ok(dict)
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
