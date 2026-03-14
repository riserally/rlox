use burn::module::Param;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use rlox_nn::{EntropyTuner, NNError, TensorData};

use crate::convert::*;

/// Automatic entropy coefficient tuning for SAC.
///
/// Maintains log_alpha as a learnable parameter and optimizes:
///   alpha_loss = -log_alpha * (log_prob + target_entropy)
#[derive(Module, Debug)]
struct AlphaModel<B: Backend> {
    log_alpha: Param<Tensor<B, 1>>,
}

pub struct BurnEntropyTuner<B: AutodiffBackend> {
    model: AlphaModel<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<Adam, AlphaModel<B>, B>,
    device: B::Device,
    lr: f32,
}

impl<B: AutodiffBackend> BurnEntropyTuner<B> {
    pub fn new(lr: f32, device: B::Device) -> Self {
        let log_alpha = Tensor::zeros([1], &device);
        let model = AlphaModel {
            log_alpha: Param::from_tensor(log_alpha),
        };
        let optimizer = AdamConfig::new().init();

        Self {
            model,
            optimizer,
            device,
            lr,
        }
    }
}

impl<B: AutodiffBackend> EntropyTuner for BurnEntropyTuner<B>
where
    B::Device: Clone,
{
    fn alpha(&self) -> f32 {
        let val: Vec<f32> = self
            .model
            .log_alpha
            .val()
            .inner()
            .exp()
            .into_data()
            .to_vec()
            .unwrap();
        val[0]
    }

    fn update(&mut self, log_probs: &TensorData, target_entropy: f32) -> Result<f64, NNError> {
        let log_probs_t = to_tensor_1d::<B>(log_probs, &self.device);
        let log_alpha = self.model.log_alpha.val();

        // SAC dual: alpha_loss = -(alpha * (log_probs + target_entropy)).mean()
        let alpha = log_alpha.exp();
        let alpha_loss = -(alpha * (log_probs_t + target_entropy)).mean();

        let grads = alpha_loss.clone().backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(self.lr.into(), self.model.clone(), grads);

        let loss_val: f32 = alpha_loss.inner().into_data().to_vec::<f32>().unwrap()[0];
        Ok(loss_val as f64)
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
    fn test_initial_alpha() {
        let tuner = BurnEntropyTuner::<TestBackend>::new(3e-4, device().into());
        let alpha = tuner.alpha();
        assert!((alpha - 1.0).abs() < 1e-4, "exp(0) = 1: got {alpha}");
    }

    #[test]
    fn test_update_runs() {
        let mut tuner = BurnEntropyTuner::<TestBackend>::new(3e-4, device().into());
        let log_probs = TensorData::new(vec![-1.0; 32], vec![32]);
        let loss = tuner.update(&log_probs, -1.0).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_alpha_changes_after_update() {
        let mut tuner = BurnEntropyTuner::<TestBackend>::new(1e-2, device().into());
        let alpha_before = tuner.alpha();

        let log_probs = TensorData::new(vec![-2.0; 32], vec![32]);
        tuner.update(&log_probs, -1.0).unwrap();

        let alpha_after = tuner.alpha();
        assert!(
            (alpha_before - alpha_after).abs() > 1e-6,
            "alpha should change: {alpha_before} vs {alpha_after}"
        );
    }

    #[test]
    fn test_alpha_increases_when_too_deterministic() {
        let mut tuner = BurnEntropyTuner::<TestBackend>::new(1e-1, device().into());

        // Positive log_probs = peaked distribution (density > 1), entropy below target.
        // SAC dual should increase alpha to encourage more exploration.
        let log_probs = TensorData::new(vec![2.0; 64], vec![64]);
        for _ in 0..20 {
            tuner.update(&log_probs, -1.0).unwrap();
        }

        let alpha = tuner.alpha();
        assert!(alpha > 1.0, "alpha should increase when policy is too deterministic: {alpha}");
    }
}
