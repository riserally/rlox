use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use rlox_nn::{Activation, ContinuousQFunction, MLPConfig, NNError, TensorData, TrainMetrics};

use crate::convert::*;
use crate::mlp::MLP;

pub struct CandleTwinQ {
    q1: MLP,
    q2: MLP,
    q1_target: MLP,
    q2_target: MLP,
    varmap: VarMap,
    target_varmap: VarMap,
    q1_optimizer: candle_nn::AdamW,
    q2_optimizer: candle_nn::AdamW,
    device: Device,
    lr: f64,
}

impl CandleTwinQ {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: usize,
        lr: f64,
        device: Device,
    ) -> Result<Self, NNError> {
        let config = MLPConfig::new(obs_dim + act_dim, 1)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::ReLU);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let q1 = MLP::new(&config, vb.pp("q1")).nn_err()?;
        let q2 = MLP::new(&config, vb.pp("q2")).nn_err()?;

        let target_varmap = VarMap::new();
        let tvb = VarBuilder::from_varmap(&target_varmap, candle_core::DType::F32, &device);
        let q1_target = MLP::new(&config, tvb.pp("q1")).nn_err()?;
        let q2_target = MLP::new(&config, tvb.pp("q2")).nn_err()?;

        // Copy weights
        {
            let src = varmap.data().lock().unwrap();
            let tgt = target_varmap.data().lock().unwrap();
            for (name, var) in src.iter() {
                if let Some(tvar) = tgt.get(name) {
                    tvar.set(&var.as_tensor().clone()).unwrap();
                }
            }
        }

        // Separate optimizers for q1 and q2 params
        let all_params = varmap.all_vars();
        let q1_optimizer =
            candle_nn::AdamW::new(all_params.clone(), candle_nn::ParamsAdamW { lr, ..Default::default() })
                .nn_err()?;
        let q2_optimizer =
            candle_nn::AdamW::new(all_params, candle_nn::ParamsAdamW { lr, ..Default::default() })
                .nn_err()?;

        Ok(Self {
            q1,
            q2,
            q1_target,
            q2_target,
            varmap,
            target_varmap,
            q1_optimizer,
            q2_optimizer,
            device,
            lr,
        })
    }

    fn forward_q(q: &MLP, obs: &Tensor, actions: &Tensor) -> candle_core::Result<Tensor> {
        let input = Tensor::cat(&[obs, actions], 1)?;
        q.forward(&input)
    }

    /// Forward through Q1 with autograd preserved (for actor loss in SAC/TD3).
    pub fn q1_forward(&self, obs: &Tensor, actions: &Tensor) -> candle_core::Result<Tensor> {
        Self::forward_q(&self.q1, obs, actions)
    }

    /// Forward through both Q-networks with autograd preserved (for SAC actor loss).
    pub fn twin_q_forward(
        &self,
        obs: &Tensor,
        actions: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let q1 = Self::forward_q(&self.q1, obs, actions)?;
        let q2 = Self::forward_q(&self.q2, obs, actions)?;
        Ok((q1, q2))
    }
}

impl ContinuousQFunction for CandleTwinQ {
    fn q_value(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let act_t = to_tensor_2d(actions, &self.device).nn_err()?;
        let q = Self::forward_q(&self.q1, &obs_t, &act_t)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;
        from_tensor_1d(&q).nn_err()
    }

    fn twin_q_values(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<(TensorData, TensorData), NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let act_t = to_tensor_2d(actions, &self.device).nn_err()?;
        let q1 = Self::forward_q(&self.q1, &obs_t, &act_t)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;
        let q2 = Self::forward_q(&self.q2, &obs_t, &act_t)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;
        Ok((from_tensor_1d(&q1).nn_err()?, from_tensor_1d(&q2).nn_err()?))
    }

    fn target_twin_q_values(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<(TensorData, TensorData), NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let act_t = to_tensor_2d(actions, &self.device).nn_err()?;
        let q1 = Self::forward_q(&self.q1_target, &obs_t, &act_t)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;
        let q2 = Self::forward_q(&self.q2_target, &obs_t, &act_t)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;
        Ok((from_tensor_1d(&q1).nn_err()?, from_tensor_1d(&q2).nn_err()?))
    }

    fn critic_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        targets: &TensorData,
    ) -> Result<TrainMetrics, NNError> {
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let act_t = to_tensor_2d(actions, &self.device).nn_err()?;
        let target_t = to_tensor_1d(targets, &self.device).nn_err()?;

        let q1 = Self::forward_q(&self.q1, &obs_t, &act_t)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;
        let q1_loss = (&q1 - &target_t)
            .nn_err()?
            .sqr()
            .nn_err()?
            .mean_all()
            .nn_err()?;

        let q2 = Self::forward_q(&self.q2, &obs_t, &act_t)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;
        let q2_loss = (&q2 - &target_t)
            .nn_err()?
            .sqr()
            .nn_err()?
            .mean_all()
            .nn_err()?;

        let total = (&q1_loss + &q2_loss).nn_err()?;
        self.q1_optimizer.backward_step(&total).nn_err()?;

        let q1_val: f32 = q1_loss.to_scalar().nn_err()?;
        let q2_val: f32 = q2_loss.to_scalar().nn_err()?;

        let mut metrics = TrainMetrics::new();
        metrics.insert("q1_loss", q1_val as f64);
        metrics.insert("q2_loss", q2_val as f64);
        metrics.insert("critic_loss", ((q1_val + q2_val) / 2.0) as f64);
        Ok(metrics)
    }

    fn soft_update_targets(&mut self, tau: f32) {
        let src = self.varmap.data().lock().unwrap();
        let tgt = self.target_varmap.data().lock().unwrap();
        for (name, var) in src.iter() {
            if let Some(tvar) = tgt.get(name) {
                let src_t = var.as_tensor();
                let tgt_t = tvar.as_tensor();
                let new_val = ((src_t * tau as f64).unwrap()
                    + (tgt_t * (1.0 - tau) as f64).unwrap())
                .unwrap();
                tvar.set(&new_val).unwrap();
            }
        }
    }
}

unsafe impl Send for CandleTwinQ {}
unsafe impl Sync for CandleTwinQ {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twin_q_shapes() {
        let q = CandleTwinQ::new(3, 1, 64, 3e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![4, 3]);
        let actions = TensorData::zeros(vec![4, 1]);
        let (q1, q2) = q.twin_q_values(&obs, &actions).unwrap();
        assert_eq!(q1.shape, vec![4]);
        assert_eq!(q2.shape, vec![4]);
    }

    #[test]
    fn test_critic_step() {
        let mut q = CandleTwinQ::new(3, 1, 64, 3e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![16, 3]);
        let actions = TensorData::zeros(vec![16, 1]);
        let targets = TensorData::new(vec![1.0; 16], vec![16]);
        let metrics = q.critic_step(&obs, &actions, &targets).unwrap();
        assert!(metrics.get("critic_loss").unwrap().is_finite());
    }

    #[test]
    fn test_soft_update() {
        let mut q = CandleTwinQ::new(3, 1, 64, 3e-4, Device::Cpu).unwrap();
        let obs = TensorData::zeros(vec![4, 3]);
        let actions = TensorData::zeros(vec![4, 1]);

        // Train
        let targets = TensorData::new(vec![10.0; 4], vec![4]);
        q.critic_step(&obs, &actions, &targets).unwrap();

        // Before update
        let (q1, _) = q.twin_q_values(&obs, &actions).unwrap();
        let (tq1, _) = q.target_twin_q_values(&obs, &actions).unwrap();
        assert_ne!(q1.data, tq1.data);

        // Hard update
        q.soft_update_targets(1.0);
        let (q1b, _) = q.twin_q_values(&obs, &actions).unwrap();
        let (tq1b, _) = q.target_twin_q_values(&obs, &actions).unwrap();
        for (a, b) in q1b.data.iter().zip(tq1b.data.iter()) {
            assert!((a - b).abs() < 1e-4, "should match: {a} vs {b}");
        }
    }
}
