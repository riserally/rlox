use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use rlox_nn::{Activation, MLPConfig, NNError, TensorData};

use crate::convert::*;
use crate::mlp::{apply_activation, ActivationKind, MLPParams, MLP};

/// Learnable parameters for Q-network — Module-derivable.
#[derive(Module, Debug)]
pub struct QNetworkParams<B: Backend> {
    net: MLPParams<B>,
}

/// Standard DQN Q-network model.
#[derive(Debug)]
pub struct QNetworkModel<B: Backend> {
    pub params: QNetworkParams<B>,
    activation: ActivationKind,
}

impl<B: Backend> QNetworkModel<B> {
    pub fn new(obs_dim: usize, n_actions: usize, hidden: usize, device: &B::Device) -> Self {
        let config = MLPConfig::new(obs_dim, n_actions)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::ReLU);
        let mlp = MLP::new(&config, device);
        Self {
            params: QNetworkParams { net: mlp.params },
            activation: config.activation.into(),
        }
    }

    fn mlp_forward(
        layers: &[burn::nn::Linear<B>],
        input: Tensor<B, 2>,
        act: ActivationKind,
    ) -> Tensor<B, 2> {
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

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        Self::mlp_forward(&self.params.net.layers, obs, self.activation)
    }

    pub fn valid(&self) -> QNetworkModel<B::InnerBackend>
    where
        B: AutodiffBackend,
    {
        QNetworkModel {
            params: self.params.valid(),
            activation: self.activation,
        }
    }
}

impl<B: Backend> Clone for QNetworkModel<B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            activation: self.activation,
        }
    }
}

/// Learnable parameters for dueling Q-network.
#[derive(Module, Debug)]
pub struct DuelingQNetworkParams<B: Backend> {
    feature: MLPParams<B>,
    value_stream: MLPParams<B>,
    advantage_stream: MLPParams<B>,
}

/// Dueling DQN Q-network model.
#[derive(Debug)]
pub struct DuelingQNetworkModel<B: Backend> {
    pub params: DuelingQNetworkParams<B>,
    feature_activation: ActivationKind,
    value_activation: ActivationKind,
    advantage_activation: ActivationKind,
}

impl<B: Backend> DuelingQNetworkModel<B> {
    pub fn new(obs_dim: usize, n_actions: usize, hidden: usize, device: &B::Device) -> Self {
        let feature_config = MLPConfig::new(obs_dim, hidden)
            .with_hidden(vec![])
            .with_activation(Activation::ReLU);
        let value_config = MLPConfig::new(hidden, 1)
            .with_hidden(vec![hidden])
            .with_activation(Activation::ReLU);
        let advantage_config = MLPConfig::new(hidden, n_actions)
            .with_hidden(vec![hidden])
            .with_activation(Activation::ReLU);

        let feature_mlp = MLP::new(&feature_config, device);
        let value_mlp = MLP::new(&value_config, device);
        let advantage_mlp = MLP::new(&advantage_config, device);

        Self {
            params: DuelingQNetworkParams {
                feature: feature_mlp.params,
                value_stream: value_mlp.params,
                advantage_stream: advantage_mlp.params,
            },
            feature_activation: feature_config.activation.into(),
            value_activation: value_config.activation.into(),
            advantage_activation: advantage_config.activation.into(),
        }
    }

    fn mlp_forward(
        layers: &[burn::nn::Linear<B>],
        input: Tensor<B, 2>,
        act: ActivationKind,
    ) -> Tensor<B, 2> {
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

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let features = Self::mlp_forward(&self.params.feature.layers, obs, self.feature_activation);
        let value = Self::mlp_forward(
            &self.params.value_stream.layers,
            features.clone(),
            self.value_activation,
        );
        let advantage = Self::mlp_forward(
            &self.params.advantage_stream.layers,
            features,
            self.advantage_activation,
        );
        let adv_mean = advantage.clone().mean_dim(1);
        value + advantage - adv_mean
    }
}

impl<B: Backend> Clone for DuelingQNetworkModel<B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            feature_activation: self.feature_activation,
            value_activation: self.value_activation,
            advantage_activation: self.advantage_activation,
        }
    }
}

/// Burn DQN implementation supporting both standard and dueling architectures.
#[allow(dead_code)]
pub struct BurnDQN<B: AutodiffBackend> {
    q_network: QNetworkModel<B>,
    target_network: QNetworkModel<B::InnerBackend>,
    optimizer: OptimizerAdaptor<Adam, QNetworkParams<B>, B>,
    device: B::Device,
    n_actions: usize,
    lr: f32,
}

impl<B: AutodiffBackend> BurnDQN<B> {
    pub fn new(
        obs_dim: usize,
        n_actions: usize,
        hidden: usize,
        lr: f32,
        device: B::Device,
    ) -> Self {
        let q_network = QNetworkModel::new(obs_dim, n_actions, hidden, &device);
        let target_network = q_network.valid();
        let optimizer = AdamConfig::new().init();

        Self {
            q_network,
            target_network,
            optimizer,
            device,
            n_actions,
            lr,
        }
    }
}

impl<B: AutodiffBackend> rlox_nn::QFunction for BurnDQN<B>
where
    B::Device: Clone,
{
    fn q_values(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let obs_tensor = to_tensor_2d::<B::InnerBackend>(obs, &self.device.clone().into());
        let q = self.q_network.valid().forward(obs_tensor);
        Ok(from_tensor_2d(q))
    }

    fn q_value_at(&self, obs: &TensorData, actions: &TensorData) -> Result<TensorData, NNError> {
        let obs_tensor = to_tensor_2d::<B::InnerBackend>(obs, &self.device.clone().into());
        let q = self.q_network.valid().forward(obs_tensor);
        let actions_int = to_int_tensor_1d::<B::InnerBackend>(actions, &self.device.clone().into());
        let actions_2d = actions_int.unsqueeze_dim(1);
        let gathered = q.gather(1, actions_2d).squeeze::<1>(1);
        Ok(from_tensor_1d(gathered))
    }

    fn td_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        targets: &TensorData,
        weights: Option<&TensorData>,
    ) -> Result<(f64, TensorData), NNError> {
        let obs_tensor = to_tensor_2d::<B>(obs, &self.device);
        let q_all = self.q_network.forward(obs_tensor);

        let actions_int = to_int_tensor_1d::<B>(actions, &self.device);
        let actions_2d = actions_int.unsqueeze_dim(1);
        let q = q_all.gather(1, actions_2d).squeeze::<1>(1);

        let target = to_tensor_1d::<B>(targets, &self.device);
        let td_error = q.clone() - target;

        let loss = if let Some(w) = weights {
            let w_tensor = to_tensor_1d::<B>(w, &self.device);
            (w_tensor * td_error.clone().powf_scalar(2.0)).mean()
        } else {
            td_error.clone().powf_scalar(2.0).mean()
        };

        let grads = loss.clone().backward();
        let grads = GradientsParams::from_grads(grads, &self.q_network.params);
        self.q_network.params =
            self.optimizer
                .step(self.lr.into(), self.q_network.params.clone(), grads);

        let loss_val: f32 = loss.inner().into_data().to_vec::<f32>().unwrap()[0];
        let td_errors = from_tensor_1d(td_error.inner());

        Ok((loss_val as f64, td_errors))
    }

    fn target_q_values(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        let obs_tensor = to_tensor_2d::<B::InnerBackend>(obs, &self.device.clone().into());
        let q = self.target_network.forward(obs_tensor);
        Ok(from_tensor_2d(q))
    }

    fn hard_update_target(&mut self) {
        self.target_network = self.q_network.valid();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use rlox_nn::QFunction;

    type TestBackend = Autodiff<NdArray>;
    type TestDevice = <NdArray as Backend>::Device;

    fn device() -> TestDevice {
        Default::default()
    }

    #[test]
    fn test_q_network_shape() {
        let model = QNetworkModel::<NdArray>::new(4, 2, 64, &device());
        let obs = Tensor::<NdArray, 2>::zeros([8, 4], &device());
        let q = model.forward(obs);
        assert_eq!(q.shape().dims, [8, 2]);
    }

    #[test]
    fn test_dueling_shape() {
        let model = DuelingQNetworkModel::<NdArray>::new(4, 2, 64, &device());
        let obs = Tensor::<NdArray, 2>::zeros([8, 4], &device());
        let q = model.forward(obs);
        assert_eq!(q.shape().dims, [8, 2]);
    }

    #[test]
    fn test_q_values_shape() {
        let dqn = BurnDQN::<TestBackend>::new(4, 2, 64, 1e-4, device().into());
        let obs = TensorData::zeros(vec![8, 4]);
        let q = dqn.q_values(&obs).unwrap();
        assert_eq!(q.shape, vec![8, 2]);
    }

    #[test]
    fn test_q_value_at() {
        use rlox_nn::QFunction;
        let dqn = BurnDQN::<TestBackend>::new(4, 2, 64, 1e-4, device().into());
        let obs = TensorData::zeros(vec![4, 4]);
        let actions = TensorData::new(vec![0.0, 1.0, 0.0, 1.0], vec![4]);
        let q = dqn.q_value_at(&obs, &actions).unwrap();
        assert_eq!(q.shape, vec![4]);
    }

    #[test]
    fn test_td_step_runs() {
        use rlox_nn::QFunction;
        let mut dqn = BurnDQN::<TestBackend>::new(4, 2, 64, 1e-4, device().into());
        let obs = TensorData::zeros(vec![32, 4]);
        let actions = TensorData::new(vec![0.0; 32], vec![32]);
        let targets = TensorData::new(vec![1.0; 32], vec![32]);

        let (loss, td_errors) = dqn.td_step(&obs, &actions, &targets, None).unwrap();
        assert!(loss.is_finite());
        assert_eq!(td_errors.shape, vec![32]);
    }

    #[test]
    fn test_td_step_with_weights() {
        use rlox_nn::QFunction;
        let mut dqn = BurnDQN::<TestBackend>::new(4, 2, 64, 1e-4, device().into());
        let obs = TensorData::zeros(vec![16, 4]);
        let actions = TensorData::new(vec![0.0; 16], vec![16]);
        let targets = TensorData::new(vec![1.0; 16], vec![16]);
        let weights = TensorData::ones(vec![16]);

        let (loss, _) = dqn
            .td_step(&obs, &actions, &targets, Some(&weights))
            .unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_hard_update_target() {
        use rlox_nn::QFunction;
        let mut dqn = BurnDQN::<TestBackend>::new(4, 2, 64, 1e-4, device().into());
        let obs = TensorData::zeros(vec![4, 4]);

        // Do a training step to change q_network
        let actions = TensorData::new(vec![0.0; 4], vec![4]);
        let targets = TensorData::new(vec![10.0; 4], vec![4]);
        dqn.td_step(&obs, &actions, &targets, None).unwrap();

        // Before update, target and q should differ
        let q_before = dqn.q_values(&obs).unwrap();
        let tq_before = dqn.target_q_values(&obs).unwrap();
        assert_ne!(q_before.data, tq_before.data);

        // After hard update, they should match
        dqn.hard_update_target();
        let q_after = dqn.q_values(&obs).unwrap();
        let tq_after = dqn.target_q_values(&obs).unwrap();
        assert_eq!(q_after.data, tq_after.data);
    }
}
