use burn::module::AutodiffModule;
use burn::module::Param;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use rlox_nn::{Activation, ContinuousQFunction, MLPConfig, NNError, TensorData, TrainMetrics};

use crate::convert::*;
use crate::mlp::{apply_activation, ActivationKind, MLPParams, MLP};

/// Learnable parameters for continuous Q-network — Module-derivable.
#[derive(Module, Debug)]
pub struct ContinuousQParams<B: Backend> {
    pub net: MLPParams<B>,
}

/// Single Q-network that takes (obs, action) concatenated.
#[derive(Debug)]
pub struct ContinuousQModel<B: Backend> {
    pub params: ContinuousQParams<B>,
    activation: ActivationKind,
}

impl<B: Backend> ContinuousQModel<B> {
    pub fn new(obs_dim: usize, act_dim: usize, hidden: usize, device: &B::Device) -> Self {
        let config = MLPConfig::new(obs_dim + act_dim, 1)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::ReLU);
        let mlp = MLP::new(&config, device);
        Self {
            params: ContinuousQParams { net: mlp.params },
            activation: config.activation.into(),
        }
    }

    fn mlp_forward(layers: &[burn::nn::Linear<B>], input: Tensor<B, 2>, act: ActivationKind) -> Tensor<B, 2> {
        let n = layers.len();
        let mut x = input;
        for (i, layer) in layers.iter().enumerate() {
            x = layer.forward(x);
            if i < n - 1 {
                x = apply_activation(x, act);
            }
        }
        x
    }

    pub fn forward(&self, obs: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![obs, actions], 1);
        Self::mlp_forward(&self.params.net.layers, input, self.activation)
    }

    pub fn valid(&self) -> ContinuousQModel<B::InnerBackend>
    where
        B: AutodiffBackend,
    {
        ContinuousQModel {
            params: self.params.valid(),
            activation: self.activation,
        }
    }
}

impl<B: Backend> Clone for ContinuousQModel<B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            activation: self.activation,
        }
    }
}

/// Twin Q-networks with target networks for SAC/TD3.
pub struct BurnTwinQ<B: AutodiffBackend> {
    q1: ContinuousQModel<B>,
    q2: ContinuousQModel<B>,
    q1_target: ContinuousQModel<B::InnerBackend>,
    q2_target: ContinuousQModel<B::InnerBackend>,
    q1_optimizer: OptimizerAdaptor<Adam, ContinuousQParams<B>, B>,
    q2_optimizer: OptimizerAdaptor<Adam, ContinuousQParams<B>, B>,
    device: B::Device,
    lr: f32,
}

impl<B: AutodiffBackend> BurnTwinQ<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: usize,
        lr: f32,
        device: B::Device,
    ) -> Self {
        let q1 = ContinuousQModel::new(obs_dim, act_dim, hidden, &device);
        let q2 = ContinuousQModel::new(obs_dim, act_dim, hidden, &device);
        let q1_target = q1.valid();
        let q2_target = q2.valid();

        Self {
            q1,
            q2,
            q1_target,
            q2_target,
            q1_optimizer: AdamConfig::new().init(),
            q2_optimizer: AdamConfig::new().init(),
            device,
            lr,
        }
    }

    /// Forward through Q1 with autograd (for actor loss in SAC/TD3).
    pub fn q1_forward_ad(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.q1.forward(obs, actions)
    }

    /// Forward through both Q-networks with autograd (for SAC actor loss).
    pub fn twin_q_forward_ad(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let q1 = self.q1.forward(obs.clone(), actions.clone());
        let q2 = self.q2.forward(obs, actions);
        (q1, q2)
    }
}

impl<B: AutodiffBackend> ContinuousQFunction for BurnTwinQ<B>
where
    B::Device: Clone,
{
    fn q_value(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<TensorData, NNError> {
        let obs_t = to_tensor_2d::<B::InnerBackend>(obs, &self.device.clone().into());
        let act_t = to_tensor_2d::<B::InnerBackend>(actions, &self.device.clone().into());
        let q = self.q1.valid().forward(obs_t, act_t).squeeze::<1>(1);
        Ok(from_tensor_1d(q))
    }

    fn twin_q_values(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<(TensorData, TensorData), NNError> {
        let dev: <B::InnerBackend as Backend>::Device = self.device.clone().into();
        let obs_t = to_tensor_2d::<B::InnerBackend>(obs, &dev);
        let act_t = to_tensor_2d::<B::InnerBackend>(actions, &dev);
        let q1 = self.q1.valid().forward(obs_t.clone(), act_t.clone()).squeeze::<1>(1);
        let q2 = self.q2.valid().forward(obs_t, act_t).squeeze::<1>(1);
        Ok((from_tensor_1d(q1), from_tensor_1d(q2)))
    }

    fn target_twin_q_values(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<(TensorData, TensorData), NNError> {
        let dev: <B::InnerBackend as Backend>::Device = self.device.clone().into();
        let obs_t = to_tensor_2d::<B::InnerBackend>(obs, &dev);
        let act_t = to_tensor_2d::<B::InnerBackend>(actions, &dev);
        let q1 = self.q1_target.forward(obs_t.clone(), act_t.clone()).squeeze::<1>(1);
        let q2 = self.q2_target.forward(obs_t, act_t).squeeze::<1>(1);
        Ok((from_tensor_1d(q1), from_tensor_1d(q2)))
    }

    fn critic_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        targets: &TensorData,
    ) -> Result<TrainMetrics, NNError> {
        let obs_t = to_tensor_2d::<B>(obs, &self.device);
        let act_t = to_tensor_2d::<B>(actions, &self.device);
        let target_t = to_tensor_1d::<B>(targets, &self.device);

        // Q1 loss
        let q1 = self.q1.forward(obs_t.clone(), act_t.clone()).squeeze::<1>(1);
        let q1_loss = (q1 - target_t.clone()).powf_scalar(2.0).mean();

        let grads1 = q1_loss.clone().backward();
        let grads1 = GradientsParams::from_grads(grads1, &self.q1.params);
        self.q1.params = self.q1_optimizer.step(self.lr.into(), self.q1.params.clone(), grads1);

        // Q2 loss
        let q2 = self.q2.forward(obs_t, act_t).squeeze::<1>(1);
        let q2_loss = (q2 - target_t).powf_scalar(2.0).mean();

        let grads2 = q2_loss.clone().backward();
        let grads2 = GradientsParams::from_grads(grads2, &self.q2.params);
        self.q2.params = self.q2_optimizer.step(self.lr.into(), self.q2.params.clone(), grads2);

        let q1_val: f32 = q1_loss.inner().into_data().to_vec::<f32>().unwrap()[0];
        let q2_val: f32 = q2_loss.inner().into_data().to_vec::<f32>().unwrap()[0];

        let mut metrics = TrainMetrics::new();
        metrics.insert("q1_loss", q1_val as f64);
        metrics.insert("q2_loss", q2_val as f64);
        metrics.insert("critic_loss", ((q1_val + q2_val) / 2.0) as f64);

        Ok(metrics)
    }

    fn soft_update_targets(&mut self, tau: f32) {
        if tau >= 1.0 - 1e-6 {
            self.q1_target = self.q1.valid();
            self.q2_target = self.q2.valid();
        } else {
            self.q1_target = polyak_update_q_model(
                &self.q1.valid(),
                &self.q1_target,
                tau,
                &self.device.clone().into(),
            );
            self.q2_target = polyak_update_q_model(
                &self.q2.valid(),
                &self.q2_target,
                tau,
                &self.device.clone().into(),
            );
        }
    }
}

/// Polyak update a ContinuousQModel: target = tau * source + (1-tau) * target.
fn polyak_update_q_model<B: Backend>(
    source: &ContinuousQModel<B>,
    target: &ContinuousQModel<B>,
    tau: f32,
    device: &B::Device,
) -> ContinuousQModel<B> {
    let src_record = source.params.net.clone().into_record();
    let mut tgt_record = target.params.net.clone().into_record();

    for (src_layer, tgt_layer) in src_record.layers.iter().zip(tgt_record.layers.iter_mut()) {
        let src_w: Tensor<B, 2> = Tensor::from_data(src_layer.weight.val().into_data(), device);
        let tgt_w: Tensor<B, 2> = Tensor::from_data(tgt_layer.weight.val().into_data(), device);
        let new_w = src_w * tau + tgt_w * (1.0 - tau);
        tgt_layer.weight = Param::from_tensor(new_w);

        if let (Some(src_b), Some(tgt_b)) = (&src_layer.bias, &mut tgt_layer.bias) {
            let src_bt: Tensor<B, 1> = Tensor::from_data(src_b.val().into_data(), device);
            let tgt_bt: Tensor<B, 1> = Tensor::from_data(tgt_b.val().into_data(), device);
            let new_b = src_bt * tau + tgt_bt * (1.0 - tau);
            *tgt_b = Param::from_tensor(new_b);
        }
    }

    let new_net_params = target.params.net.clone().load_record(tgt_record);
    ContinuousQModel {
        params: ContinuousQParams { net: new_net_params },
        activation: target.activation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;

    type TestBackend = Autodiff<NdArray>;
    type TestDevice = <NdArray as Backend>::Device;

    fn device() -> TestDevice {
        Default::default()
    }

    #[test]
    fn test_continuous_q_shape() {
        let model = ContinuousQModel::<NdArray>::new(3, 1, 64, &device());
        let obs = Tensor::<NdArray, 2>::zeros([4, 3], &device());
        let act = Tensor::<NdArray, 2>::zeros([4, 1], &device());
        let q = model.forward(obs, act);
        assert_eq!(q.shape().dims, [4, 1]);
    }

    #[test]
    fn test_twin_q_values() {
        let twin = BurnTwinQ::<TestBackend>::new(3, 1, 64, 3e-4, device().into());
        let obs = TensorData::zeros(vec![4, 3]);
        let actions = TensorData::zeros(vec![4, 1]);
        let (q1, q2) = twin.twin_q_values(&obs, &actions).unwrap();
        assert_eq!(q1.shape, vec![4]);
        assert_eq!(q2.shape, vec![4]);
    }

    #[test]
    fn test_target_twin_q() {
        let twin = BurnTwinQ::<TestBackend>::new(3, 1, 64, 3e-4, device().into());
        let obs = TensorData::zeros(vec![4, 3]);
        let actions = TensorData::zeros(vec![4, 1]);
        let (q1, q2) = twin.target_twin_q_values(&obs, &actions).unwrap();
        assert_eq!(q1.shape, vec![4]);
        assert_eq!(q2.shape, vec![4]);
    }

    #[test]
    fn test_critic_step() {
        let mut twin = BurnTwinQ::<TestBackend>::new(3, 1, 64, 3e-4, device().into());
        let obs = TensorData::zeros(vec![16, 3]);
        let actions = TensorData::zeros(vec![16, 1]);
        let targets = TensorData::new(vec![1.0; 16], vec![16]);

        let metrics = twin.critic_step(&obs, &actions, &targets).unwrap();
        assert!(metrics.get("q1_loss").unwrap().is_finite());
        assert!(metrics.get("q2_loss").unwrap().is_finite());
        assert!(metrics.get("critic_loss").unwrap().is_finite());
    }

    #[test]
    fn test_soft_update_hard() {
        let mut twin = BurnTwinQ::<TestBackend>::new(3, 1, 64, 3e-4, device().into());

        // Train to change Q1
        let obs = TensorData::zeros(vec![8, 3]);
        let actions = TensorData::zeros(vec![8, 1]);
        let targets = TensorData::new(vec![10.0; 8], vec![8]);
        twin.critic_step(&obs, &actions, &targets).unwrap();

        // Before soft update, targets should differ from online
        let (q1_o, _) = twin.twin_q_values(&obs, &actions).unwrap();
        let (q1_t, _) = twin.target_twin_q_values(&obs, &actions).unwrap();
        assert_ne!(q1_o.data, q1_t.data);

        // Hard update (tau=1)
        twin.soft_update_targets(1.0);
        let (q1_o2, _) = twin.twin_q_values(&obs, &actions).unwrap();
        let (q1_t2, _) = twin.target_twin_q_values(&obs, &actions).unwrap();
        for (a, b) in q1_o2.data.iter().zip(q1_t2.data.iter()) {
            assert!((a - b).abs() < 1e-5, "hard update should sync: {a} vs {b}");
        }
    }
}
