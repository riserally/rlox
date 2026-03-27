use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use rlox_nn::{EntropyTuner, NNError, TensorData};

use crate::convert::*;

pub struct CandleEntropyTuner {
    varmap: VarMap,
    log_alpha: Tensor,
    optimizer: candle_nn::AdamW,
    device: Device,
}

impl CandleEntropyTuner {
    pub fn new(lr: f64, device: Device) -> Result<Self, NNError> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let log_alpha = vb
            .get_with_hints(1, "log_alpha", candle_nn::Init::Const(0.0))
            .nn_err()?;

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
            varmap,
            log_alpha,
            optimizer,
            device,
        })
    }
}

impl EntropyTuner for CandleEntropyTuner {
    fn alpha(&self) -> f32 {
        let val: Vec<f32> = self.log_alpha.exp().unwrap().to_vec1().unwrap();
        val[0]
    }

    fn update(&mut self, log_probs: &TensorData, target_entropy: f32) -> Result<f64, NNError> {
        let lp = to_tensor_1d(log_probs, &self.device).nn_err()?;
        let alpha = self.log_alpha.exp().nn_err()?;
        let batch_size = log_probs.data.len();
        let alpha_broadcast = alpha.broadcast_as(batch_size).nn_err()?;
        let alpha_loss = (&alpha_broadcast * &(&lp + target_entropy as f64).nn_err()?)
            .nn_err()?
            .mean_all()
            .nn_err()?
            .neg()
            .nn_err()?;

        self.optimizer.backward_step(&alpha_loss).nn_err()?;

        let loss_val: f32 = alpha_loss.to_scalar().nn_err()?;
        Ok(loss_val as f64)
    }
}

unsafe impl Send for CandleEntropyTuner {}
unsafe impl Sync for CandleEntropyTuner {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_alpha() {
        let tuner = CandleEntropyTuner::new(3e-4, Device::Cpu).unwrap();
        let alpha = tuner.alpha();
        assert!((alpha - 1.0).abs() < 1e-4, "exp(0) = 1: got {alpha}");
    }

    #[test]
    fn test_update_runs() {
        let mut tuner = CandleEntropyTuner::new(3e-4, Device::Cpu).unwrap();
        let log_probs = TensorData::new(vec![-1.0; 32], vec![32]);
        let loss = tuner.update(&log_probs, -1.0).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_alpha_changes() {
        let mut tuner = CandleEntropyTuner::new(1e-2, Device::Cpu).unwrap();
        let before = tuner.alpha();
        let log_probs = TensorData::new(vec![-2.0; 32], vec![32]);
        tuner.update(&log_probs, -1.0).unwrap();
        let after = tuner.alpha();
        assert!((before - after).abs() > 1e-6, "{before} vs {after}");
    }
}
