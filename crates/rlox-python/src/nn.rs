use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use rlox_nn::{ActorCritic, PPOStepConfig, TensorData};

type BurnBackend = Autodiff<NdArray>;

enum Backend {
    Burn(rlox_burn::actor_critic::BurnActorCritic<BurnBackend>),
    Candle(rlox_candle::actor_critic::CandleActorCritic),
}

/// A discrete actor-critic policy backed by a pure-Rust NN (Burn or Candle).
///
/// Args:
///     backend: ``"burn"`` or ``"candle"``
///     obs_dim: observation dimension
///     n_actions: number of discrete actions
///     hidden: hidden layer width (default 64)
///     lr: learning rate (default 2.5e-4)
///     seed: RNG seed (default 42)
#[pyclass(name = "ActorCritic", unsendable)]
pub struct PyActorCritic {
    inner: Backend,
    obs_dim: usize,
}

#[pymethods]
impl PyActorCritic {
    #[new]
    #[pyo3(signature = (backend, obs_dim, n_actions, hidden = 64, lr = 2.5e-4, seed = 42))]
    fn new(
        backend: &str,
        obs_dim: usize,
        n_actions: usize,
        hidden: usize,
        lr: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = match backend {
            "burn" => {
                let ac = rlox_burn::actor_critic::BurnActorCritic::<BurnBackend>::new(
                    obs_dim,
                    n_actions,
                    hidden,
                    lr as f32,
                    Default::default(),
                    seed,
                );
                Backend::Burn(ac)
            }
            "candle" => {
                let ac = rlox_candle::actor_critic::CandleActorCritic::new(
                    obs_dim,
                    n_actions,
                    hidden,
                    lr,
                    candle_core::Device::Cpu,
                    seed,
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Backend::Candle(ac)
            }
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown backend '{}'. Use 'burn' or 'candle'.",
                    backend
                )));
            }
        };
        Ok(Self { inner, obs_dim })
    }

    /// Sample actions from the policy (no gradient tracking).
    ///
    /// Args:
    ///     obs: flat float32 array of length ``batch * obs_dim``
    ///
    /// Returns:
    ///     ``(actions, log_probs)`` — two 1-D float32 arrays of length ``batch``.
    fn act<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let obs_slice = obs.as_slice()?;
        let total = obs_slice.len();
        if total % self.obs_dim != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "obs length {} not divisible by obs_dim {}",
                total, self.obs_dim
            )));
        }
        let n = total / self.obs_dim;
        let td = TensorData::new(obs_slice.to_vec(), vec![n, self.obs_dim]);

        let out = match &self.inner {
            Backend::Burn(ac) => ac.act(&td),
            Backend::Candle(ac) => ac.act(&td),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let actions = PyArray1::from_vec(py, out.actions.data);
        let log_probs = PyArray1::from_vec(py, out.log_probs.data);
        Ok((actions, log_probs))
    }

    /// Compute state values (no gradient tracking).
    ///
    /// Args:
    ///     obs: flat float32 array of length ``batch * obs_dim``
    ///
    /// Returns:
    ///     1-D float32 array of values, length ``batch``.
    fn value<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let obs_slice = obs.as_slice()?;
        let n = obs_slice.len() / self.obs_dim;
        let td = TensorData::new(obs_slice.to_vec(), vec![n, self.obs_dim]);

        let out = match &self.inner {
            Backend::Burn(ac) => ac.value(&td),
            Backend::Candle(ac) => ac.value(&td),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyArray1::from_vec(py, out.data))
    }

    /// Perform one PPO gradient step.
    ///
    /// All array arguments are flat float32 of length ``batch``,
    /// except ``obs`` which is ``batch * obs_dim``.
    ///
    /// Returns:
    ///     dict with keys ``policy_loss``, ``value_loss``, ``entropy``,
    ///     ``approx_kl``, ``clip_fraction``.
    #[pyo3(signature = (obs, actions, old_log_probs, advantages, returns, old_values,
                        clip_eps = 0.2, vf_coef = 0.5, ent_coef = 0.01,
                        max_grad_norm = 0.5, clip_vloss = true))]
    fn ppo_step<'py>(
        &mut self,
        py: Python<'py>,
        obs: PyReadonlyArray1<'py, f32>,
        actions: PyReadonlyArray1<'py, f32>,
        old_log_probs: PyReadonlyArray1<'py, f32>,
        advantages: PyReadonlyArray1<'py, f32>,
        returns: PyReadonlyArray1<'py, f32>,
        old_values: PyReadonlyArray1<'py, f32>,
        clip_eps: f32,
        vf_coef: f32,
        ent_coef: f32,
        max_grad_norm: f32,
        clip_vloss: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let obs_slice = obs.as_slice()?;
        let batch_size = obs_slice.len() / self.obs_dim;

        let obs_td = TensorData::new(obs_slice.to_vec(), vec![batch_size, self.obs_dim]);
        let actions_td = TensorData::new(actions.as_slice()?.to_vec(), vec![batch_size]);
        let old_lp_td = TensorData::new(old_log_probs.as_slice()?.to_vec(), vec![batch_size]);
        let adv_td = TensorData::new(advantages.as_slice()?.to_vec(), vec![batch_size]);
        let ret_td = TensorData::new(returns.as_slice()?.to_vec(), vec![batch_size]);
        let old_v_td = TensorData::new(old_values.as_slice()?.to_vec(), vec![batch_size]);

        let config = PPOStepConfig {
            clip_eps,
            vf_coef,
            ent_coef,
            max_grad_norm,
            clip_vloss,
        };

        let metrics = match &mut self.inner {
            Backend::Burn(ac) => {
                ac.ppo_step(&obs_td, &actions_td, &old_lp_td, &adv_td, &ret_td, &old_v_td, &config)
            }
            Backend::Candle(ac) => {
                ac.ppo_step(&obs_td, &actions_td, &old_lp_td, &adv_td, &ret_td, &old_v_td, &config)
            }
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        for (k, v) in &metrics.entries {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// Get the current learning rate.
    #[getter]
    fn learning_rate(&self) -> f32 {
        match &self.inner {
            Backend::Burn(ac) => ac.learning_rate(),
            Backend::Candle(ac) => ac.learning_rate(),
        }
    }

    /// Set the learning rate.
    #[setter]
    fn set_learning_rate(&mut self, lr: f32) {
        match &mut self.inner {
            Backend::Burn(ac) => ac.set_learning_rate(lr),
            Backend::Candle(ac) => ac.set_learning_rate(lr),
        }
    }
}
