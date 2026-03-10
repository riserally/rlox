use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rlox_core::buffer::columnar::ExperienceTable;
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

    #[pyo3(signature = (obs, action, reward, terminated, truncated))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: f32,
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> PyResult<()> {
        let record = ExperienceRecord {
            obs: obs.as_slice()?.to_vec(),
            action,
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

    #[pyo3(signature = (obs, action, reward, terminated, truncated))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: f32,
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> PyResult<()> {
        let record = ExperienceRecord {
            obs: obs.as_slice()?.to_vec(),
            action,
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
        dict.set_item("actions", PyArray1::from_vec(py, batch.actions))?;
        dict.set_item("rewards", PyArray1::from_vec(py, batch.rewards))?;

        let term: Vec<u8> = batch.terminated.iter().map(|&b| b as u8).collect();
        let trunc: Vec<u8> = batch.truncated.iter().map(|&b| b as u8).collect();
        dict.set_item("terminated", PyArray1::from_slice(py, &term))?;
        dict.set_item("truncated", PyArray1::from_slice(py, &trunc))?;
        Ok(dict)
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
