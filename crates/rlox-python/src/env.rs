use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rlox_core::env::builtins::CartPole;
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::spaces::Action;
use rlox_core::env::{RLEnv, Transition};
use rlox_core::error::RloxError;
use rlox_core::seed::derive_seed;

fn rlox_err_to_py(e: RloxError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn transition_to_pydict<'py>(py: Python<'py>, t: &Transition) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let obs_array = PyArray1::from_slice(py, t.obs.as_slice());
    dict.set_item("obs", obs_array)?;
    dict.set_item("reward", t.reward)?;
    dict.set_item("terminated", t.terminated)?;
    dict.set_item("truncated", t.truncated)?;
    Ok(dict)
}

/// Python-facing CartPole environment.
#[pyclass(name = "CartPole")]
pub struct PyCartPole {
    inner: CartPole,
}

#[pymethods]
impl PyCartPole {
    #[new]
    #[pyo3(signature = (seed = None))]
    fn new(seed: Option<u64>) -> Self {
        PyCartPole {
            inner: CartPole::new(seed),
        }
    }

    fn step<'py>(&mut self, py: Python<'py>, action: u32) -> PyResult<Bound<'py, PyDict>> {
        let t = self
            .inner
            .step(&Action::Discrete(action))
            .map_err(rlox_err_to_py)?;
        transition_to_pydict(py, &t)
    }

    #[pyo3(signature = (seed = None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let obs = self.inner.reset(seed).map_err(rlox_err_to_py)?;
        Ok(PyArray1::from_slice(py, obs.as_slice()))
    }

    fn render(&self) -> Option<String> {
        self.inner.render()
    }
}

/// Python-facing vectorized environment for parallel stepping.
#[pyclass(name = "VecEnv")]
pub struct PyVecEnv {
    inner: VecEnv,
}

#[pymethods]
impl PyVecEnv {
    /// Create a VecEnv with `n` environments.
    ///
    /// For Rust-native envs (e.g. "CartPole-v1"), creates them directly.
    /// The `env_id` parameter defaults to "CartPole-v1" for backward compatibility.
    #[new]
    #[pyo3(signature = (n, seed = None, env_id = None))]
    fn new(n: usize, seed: Option<u64>, env_id: Option<&str>) -> PyResult<Self> {
        let env_name = env_id.unwrap_or("CartPole-v1");
        let master_seed = seed.unwrap_or(0);

        match env_name {
            "CartPole-v1" | "CartPole" => {
                let envs: Vec<Box<dyn RLEnv>> = (0..n)
                    .map(|i| {
                        let s = derive_seed(master_seed, i);
                        Box::new(CartPole::new(Some(s))) as Box<dyn RLEnv>
                    })
                    .collect();
                Ok(PyVecEnv {
                    inner: VecEnv::new(envs),
                })
            }
            unknown => Err(PyValueError::new_err(format!(
                "Unknown env_id '{}'. Supported native env IDs: \
                 [\"CartPole-v1\", \"CartPole\"]. \
                 For Gymnasium environments, use GymVecEnv instead.",
                unknown,
            ))),
        }
    }

    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    /// Step all envs. `actions` is a list/array of integer actions.
    fn step_all<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let actions: Vec<Action> = actions.into_iter().map(Action::Discrete).collect();
        let batch = self.inner.step_all_flat(&actions).map_err(rlox_err_to_py)?;

        let n = self.inner.num_envs();
        let obs_dim = batch.obs_dim;

        let dict = PyDict::new(py);
        // Use flat obs directly — single copy, no per-env Vec overhead
        let obs_flat = PyArray1::from_vec(py, batch.obs_flat);
        let obs_array = obs_flat
            .call_method1("reshape", ((n, obs_dim),))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape obs: {}", e)))?;
        dict.set_item("obs", obs_array)?;

        let rewards = PyArray1::from_slice(py, &batch.rewards);
        dict.set_item("rewards", rewards)?;

        let terminated: Vec<u8> = batch.terminated.iter().map(|&b| b as u8).collect();
        let terminated_arr = PyArray1::from_slice(py, &terminated);
        dict.set_item("terminated", terminated_arr)?;

        let truncated: Vec<u8> = batch.truncated.iter().map(|&b| b as u8).collect();
        let truncated_arr = PyArray1::from_slice(py, &truncated);
        dict.set_item("truncated", truncated_arr)?;

        // terminal_obs: list of Optional[ndarray]
        let terminal_obs_list = pyo3::types::PyList::empty(py);
        for tobs in &batch.terminal_obs {
            match tobs {
                Some(obs) => {
                    let arr = PyArray1::from_slice(py, obs);
                    terminal_obs_list.append(arr)?;
                }
                None => {
                    terminal_obs_list.append(py.None())?;
                }
            }
        }
        dict.set_item("terminal_obs", terminal_obs_list)?;

        Ok(dict)
    }

    /// Reset all envs with an optional master seed.
    #[pyo3(signature = (seed = None))]
    fn reset_all<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let observations = self.inner.reset_all(seed).map_err(rlox_err_to_py)?;
        let vecs: Vec<Vec<f32>> = observations.into_iter().map(|o| o.into_inner()).collect();
        PyArray2::from_vec2(py, &vecs)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create obs array: {}", e)))
    }
}

/// Python-facing wrapper for a Gymnasium environment.
///
/// This allows Python gymnasium.Env objects to be used from Rust-side code
/// (stepped sequentially due to GIL constraints).
#[pyclass(name = "GymEnv")]
pub struct PyGymEnv {
    gym_env: PyObject,
}

#[pymethods]
impl PyGymEnv {
    /// Wrap an existing gymnasium.Env instance.
    #[new]
    fn new(env: PyObject) -> Self {
        PyGymEnv { gym_env: env }
    }

    fn step<'py>(&self, py: Python<'py>, action: PyObject) -> PyResult<Bound<'py, PyDict>> {
        let result = self.gym_env.call_method1(py, "step", (action,))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

        let obs = tuple.get_item(0)?;
        let reward: f64 = tuple.get_item(1)?.extract()?;
        let terminated: bool = tuple.get_item(2)?.extract()?;
        let truncated: bool = tuple.get_item(3)?.extract()?;

        let dict = PyDict::new(py);
        dict.set_item("obs", obs)?;
        dict.set_item("reward", reward)?;
        dict.set_item("terminated", terminated)?;
        dict.set_item("truncated", truncated)?;
        Ok(dict)
    }

    #[pyo3(signature = (seed = None))]
    fn reset<'py>(&self, py: Python<'py>, seed: Option<u64>) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py);
        if let Some(s) = seed {
            kwargs.set_item("seed", s)?;
        }
        let result = self.gym_env.call_method(py, "reset", (), Some(&kwargs))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
        let obs = tuple.get_item(0)?;
        Ok(obs.into_pyobject(py)?.into_any().unbind())
    }
}
