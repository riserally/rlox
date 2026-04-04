use candle_core::Device;
use candle_nn::{Optimizer, VarBuilder, VarMap};

use rlox_nn::{Activation, MLPConfig, NNError, QFunction, TensorData};

use crate::convert::*;
use crate::mlp::MLP;

#[allow(dead_code)]
pub struct CandleDQN {
    q_network: MLP,
    target_network: MLP,
    varmap: VarMap,
    target_varmap: VarMap,
    optimizer: candle_nn::AdamW,
    device: Device,
    n_actions: usize,
    obs_dim: usize,
    hidden: usize,
    lr: f64,
}

impl CandleDQN {
    pub fn new(
        obs_dim: usize,
        n_actions: usize,
        hidden: usize,
        lr: f64,
        device: Device,
    ) -> Result<Self, NNError> {
        let config = MLPConfig::new(obs_dim, n_actions)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::ReLU);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let q_network = MLP::new(&config, vb.pp("q")).nn_err()?;

        let target_varmap = VarMap::new();
        let tvb = VarBuilder::from_varmap(&target_varmap, candle_core::DType::F32, &device);
        let target_network = MLP::new(&config, tvb.pp("q")).nn_err()?;

        // Copy weights from q to target
        let q_data = varmap.data().lock().unwrap();
        let t_data = target_varmap.data().lock().unwrap();
        for (name, var) in q_data.iter() {
            if let Some(tvar) = t_data.get(name) {
                tvar.set(&var.as_tensor().clone()).unwrap();
            }
        }
        drop(q_data);
        drop(t_data);

        let params = varmap.all_vars();
        let optimizer = candle_nn::AdamW::new(
            params,
            candle_nn::ParamsAdamW {
                lr,
                ..Default::default()
            },
        )
        .nn_err()?;

        Ok(Self {
            q_network,
            target_network,
            varmap,
            target_varmap,
            optimizer,
            device,
            n_actions,
            obs_dim,
            hidden,
            lr,
        })
    }
}

impl QFunction for CandleDQN {
    fn q_values(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let q = self.q_network.forward(&obs_t).nn_err()?;
        from_tensor_2d(&q).nn_err()
    }

    fn q_value_at(&self, obs: &TensorData, actions: &TensorData) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let q = self.q_network.forward(&obs_t).nn_err()?;
        let actions_idx = to_int_tensor_1d(actions, &self.device).nn_err()?;
        let actions_2d = actions_idx.unsqueeze(1).nn_err()?;
        let gathered = q.gather(&actions_2d, 1).nn_err()?.squeeze(1).nn_err()?;
        from_tensor_1d(&gathered).nn_err()
    }

    fn td_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        targets: &TensorData,
        weights: Option<&TensorData>,
    ) -> Result<(f64, TensorData), NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let q_all = self.q_network.forward(&obs_t).nn_err()?;
        let actions_idx = to_int_tensor_1d(actions, &self.device).nn_err()?;
        let actions_2d = actions_idx.unsqueeze(1).nn_err()?;
        let q = q_all.gather(&actions_2d, 1).nn_err()?.squeeze(1).nn_err()?;

        let target = to_tensor_1d(targets, &self.device).nn_err()?;
        let td_error = (&q - &target).nn_err()?;

        let loss = if let Some(w) = weights {
            let w_t = to_tensor_1d(w, &self.device).nn_err()?;
            (&w_t * &td_error.sqr().nn_err()?)
                .nn_err()?
                .mean_all()
                .nn_err()?
        } else {
            td_error.sqr().nn_err()?.mean_all().nn_err()?
        };

        self.optimizer.backward_step(&loss).nn_err()?;

        let loss_val: f32 = loss.to_scalar().nn_err()?;
        let td_err_data = from_tensor_1d(&td_error.detach()).nn_err()?;

        Ok((loss_val as f64, td_err_data))
    }

    fn target_q_values(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let q = self.target_network.forward(&obs_t).nn_err()?;
        from_tensor_2d(&q).nn_err()
    }

    fn hard_update_target(&mut self) {
        let q_data = self.varmap.data().lock().unwrap();
        let t_data = self.target_varmap.data().lock().unwrap();
        for (name, var) in q_data.iter() {
            if let Some(tvar) = t_data.get(name) {
                tvar.set(&var.as_tensor().clone()).unwrap();
            }
        }
    }
}

unsafe impl Send for CandleDQN {}
unsafe impl Sync for CandleDQN {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_values_shape() {
        let dqn = CandleDQN::new(4, 2, 64, 1e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![8, 4]);
        let q = dqn.q_values(&obs).unwrap();
        assert_eq!(q.shape, vec![8, 2]);
    }

    #[test]
    fn test_q_value_at() {
        let dqn = CandleDQN::new(4, 2, 64, 1e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![4, 4]);
        let actions = TensorData::new(vec![0.0, 1.0, 0.0, 1.0], vec![4]);
        let q = dqn.q_value_at(&obs, &actions).unwrap();
        assert_eq!(q.shape, vec![4]);
    }

    #[test]
    fn test_td_step() {
        let mut dqn = CandleDQN::new(4, 2, 64, 1e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![32, 4]);
        let actions = TensorData::new(vec![0.0; 32], vec![32]);
        let targets = TensorData::new(vec![1.0; 32], vec![32]);

        let (loss, td_errors) = dqn.td_step(&obs, &actions, &targets, None).unwrap();
        assert!(loss.is_finite());
        assert_eq!(td_errors.shape, vec![32]);
    }

    #[test]
    fn test_hard_update() {
        let mut dqn = CandleDQN::new(4, 2, 64, 1e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![4, 4]);

        // Train to change q_network
        let actions = TensorData::new(vec![0.0; 4], vec![4]);
        let targets = TensorData::new(vec![10.0; 4], vec![4]);
        dqn.td_step(&obs, &actions, &targets, None).unwrap();

        // Should differ before update
        let q = dqn.q_values(&obs).unwrap();
        let tq = dqn.target_q_values(&obs).unwrap();
        assert_ne!(q.data, tq.data);

        // Should match after hard update
        dqn.hard_update_target();
        let q2 = dqn.q_values(&obs).unwrap();
        let tq2 = dqn.target_q_values(&obs).unwrap();
        assert_eq!(q2.data, tq2.data);
    }
}
