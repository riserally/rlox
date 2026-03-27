use std::cell::RefCell;
use std::path::Path;

use candle_core::Device;
use candle_nn::{Optimizer, VarBuilder, VarMap};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rlox_nn::distributions::{categorical_entropy, categorical_log_prob, categorical_sample};
use rlox_nn::{
    ActionOutput, Activation, EvalOutput, MLPConfig, NNError, PPOStepConfig, TensorData,
    TrainMetrics,
};

use crate::convert::*;
use crate::mlp::MLP;

pub struct CandleActorCritic {
    actor: MLP,
    critic: MLP,
    varmap: VarMap,
    optimizer: candle_nn::AdamW,
    device: Device,
    n_actions: usize,
    lr: f64,
    rng: RefCell<ChaCha8Rng>,
}

impl CandleActorCritic {
    pub fn new(
        obs_dim: usize,
        n_actions: usize,
        hidden: usize,
        lr: f64,
        device: Device,
        seed: u64,
    ) -> Result<Self, NNError> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let actor_config = MLPConfig::new(obs_dim, n_actions)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::Tanh);
        let critic_config = MLPConfig::new(obs_dim, 1)
            .with_hidden(vec![hidden, hidden])
            .with_activation(Activation::Tanh);

        let actor = MLP::new(&actor_config, vb.pp("actor")).nn_err()?;
        let critic = MLP::new(&critic_config, vb.pp("critic")).nn_err()?;

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
            actor,
            critic,
            varmap,
            optimizer,
            device,
            n_actions,
            lr,
            rng: RefCell::new(ChaCha8Rng::seed_from_u64(seed)),
        })
    }

    fn compute_logits(&self, obs: &TensorData) -> Result<Vec<Vec<f32>>, NNError> {
        let batch_size = obs.shape[0];
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let logits = self.actor.forward(&obs_t).nn_err()?;
        let logits_flat: Vec<f32> = logits.flatten_all().nn_err()?.to_vec1().nn_err()?;

        Ok((0..batch_size)
            .map(|i| logits_flat[i * self.n_actions..(i + 1) * self.n_actions].to_vec())
            .collect())
    }
}

impl rlox_nn::ActorCritic for CandleActorCritic {
    fn act(&self, obs: &TensorData) -> Result<ActionOutput, NNError> {
        if obs.shape.len() != 2 {
            return Err(NNError::ShapeMismatch {
                expected: "2D [batch, obs_dim]".into(),
                got: format!("{:?}", obs.shape),
            });
        }

        let batch_size = obs.shape[0];
        let batch_logits = self.compute_logits(obs)?;

        let mut actions = Vec::with_capacity(batch_size);
        let mut log_probs = Vec::with_capacity(batch_size);

        let mut rng = self.rng.borrow_mut();
        for logits in &batch_logits {
            let u: f32 = rand::Rng::gen(&mut *rng);
            let action = categorical_sample(logits, u);
            let lp = categorical_log_prob(logits, action);
            actions.push(action as f32);
            log_probs.push(lp);
        }

        Ok(ActionOutput {
            actions: TensorData::new(actions, vec![batch_size]),
            log_probs: TensorData::new(log_probs, vec![batch_size]),
        })
    }

    fn value(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        if obs.shape.len() != 2 {
            return Err(NNError::ShapeMismatch {
                expected: "2D [batch, obs_dim]".into(),
                got: format!("{:?}", obs.shape),
            });
        }

        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;
        let values = self.critic.forward(&obs_t).nn_err()?;
        let values = values.squeeze(1).nn_err()?;
        from_tensor_1d(&values).nn_err()
    }

    fn evaluate(&self, obs: &TensorData, actions: &TensorData) -> Result<EvalOutput, NNError> {
        let batch_size = obs.shape[0];
        let batch_logits = self.compute_logits(obs)?;

        let mut log_probs = Vec::with_capacity(batch_size);
        let mut entropies = Vec::with_capacity(batch_size);

        for (i, logits) in batch_logits.iter().enumerate() {
            let action = actions.data[i] as usize;
            log_probs.push(categorical_log_prob(logits, action));
            entropies.push(categorical_entropy(logits));
        }

        let values = self.value(obs)?;

        Ok(EvalOutput {
            log_probs: TensorData::new(log_probs, vec![batch_size]),
            entropy: TensorData::new(entropies, vec![batch_size]),
            values,
        })
    }

    fn ppo_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        old_log_probs: &TensorData,
        advantages: &TensorData,
        returns: &TensorData,
        old_values: &TensorData,
        config: &PPOStepConfig,
    ) -> Result<TrainMetrics, NNError> {
        let batch_size = obs.shape[0];
        let obs_t = to_tensor_2d(obs, &self.device).nn_err()?;

        // Actor forward
        let logits = self.actor.forward(&obs_t).nn_err()?;
        let log_probs_all = candle_nn::ops::log_softmax(&logits, 1).nn_err()?;

        // Gather action log-probs
        let actions_idx = to_int_tensor_1d(actions, &self.device).nn_err()?;
        let actions_2d = actions_idx.unsqueeze(1).nn_err()?;
        let new_log_probs = log_probs_all
            .gather(&actions_2d, 1)
            .nn_err()?
            .squeeze(1)
            .nn_err()?;

        // Entropy
        let probs = candle_nn::ops::softmax(&logits, 1).nn_err()?;
        let entropy = (&probs * &log_probs_all)
            .nn_err()?
            .sum(1)
            .nn_err()?
            .neg()
            .nn_err()?;

        // Critic forward
        let new_values = self.critic.forward(&obs_t).nn_err()?.squeeze(1).nn_err()?;

        // PPO loss
        let old_lp = to_tensor_1d(old_log_probs, &self.device).nn_err()?;
        let adv = to_tensor_1d(advantages, &self.device).nn_err()?;
        let ret = to_tensor_1d(returns, &self.device).nn_err()?;

        let log_ratio = (&new_log_probs - &old_lp).nn_err()?;
        let ratio = log_ratio.exp().nn_err()?;

        let pg_loss1 = (&adv.neg().nn_err()? * &ratio).nn_err()?;
        let clamped = ratio
            .clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps)
            .nn_err()?;
        let pg_loss2 = (&adv.neg().nn_err()? * &clamped).nn_err()?;
        let policy_loss = pg_loss1.maximum(&pg_loss2).nn_err()?.mean_all().nn_err()?;

        // Value loss
        let value_loss = if config.clip_vloss {
            let old_v = to_tensor_1d(old_values, &self.device).nn_err()?;
            let v_diff = (&new_values - &old_v).nn_err()?;
            let v_clipped =
                (&old_v + v_diff.clamp(-config.clip_eps, config.clip_eps).nn_err()?).nn_err()?;
            let vf1 = (&new_values - &ret).nn_err()?.sqr().nn_err()?;
            let vf2 = (&v_clipped - &ret).nn_err()?.sqr().nn_err()?;
            (vf1.maximum(&vf2).nn_err()?.mean_all().nn_err()? * 0.5).nn_err()?
        } else {
            ((&new_values - &ret)
                .nn_err()?
                .sqr()
                .nn_err()?
                .mean_all()
                .nn_err()?
                * 0.5)
                .nn_err()?
        };

        let entropy_loss = entropy.mean_all().nn_err()?;

        let total_loss = ((&policy_loss + (&value_loss * config.vf_coef as f64).nn_err()?)
            .nn_err()?
            - (&entropy_loss * config.ent_coef as f64).nn_err()?)
        .nn_err()?;

        self.optimizer.backward_step(&total_loss).nn_err()?;

        // Extract metrics
        let policy_loss_val: f32 = policy_loss.to_scalar().nn_err()?;
        let value_loss_val: f32 = value_loss.to_scalar().nn_err()?;
        let entropy_val: f32 = entropy_loss.to_scalar().nn_err()?;

        let ratio_data: Vec<f32> = ratio.to_vec1().nn_err()?;
        let log_ratio_data: Vec<f32> = log_ratio.to_vec1().nn_err()?;
        let approx_kl: f32 = ratio_data
            .iter()
            .zip(log_ratio_data.iter())
            .map(|(&r, &lr)| (r - 1.0) - lr)
            .sum::<f32>()
            / batch_size as f32;
        let clip_fraction: f32 = ratio_data
            .iter()
            .filter(|&&r| (r - 1.0).abs() > config.clip_eps)
            .count() as f32
            / batch_size as f32;

        let mut metrics = TrainMetrics::new();
        metrics.insert("policy_loss", policy_loss_val as f64);
        metrics.insert("value_loss", value_loss_val as f64);
        metrics.insert("entropy", entropy_val as f64);
        metrics.insert("approx_kl", approx_kl as f64);
        metrics.insert("clip_fraction", clip_fraction as f64);

        Ok(metrics)
    }

    fn learning_rate(&self) -> f32 {
        self.lr as f32
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr as f64;
        self.optimizer.set_learning_rate(lr as f64);
    }

    fn save(&self, path: &Path) -> Result<(), NNError> {
        self.varmap
            .save(path)
            .map_err(|e| NNError::Serialization(e.to_string()))
    }

    fn load(&mut self, path: &Path) -> Result<(), NNError> {
        self.varmap
            .load(path)
            .map_err(|e| NNError::Serialization(e.to_string()))
    }
}

// Send + Sync are required by the trait. Candle tensors on CPU are Send + Sync.
unsafe impl Send for CandleActorCritic {}
unsafe impl Sync for CandleActorCritic {}

#[cfg(test)]
mod tests {
    use super::*;
    use rlox_nn::ActorCritic;

    #[test]
    fn test_act_shapes() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![8, 4]);
        let result = ac.act(&obs).unwrap();
        assert_eq!(result.actions.shape, vec![8]);
        assert_eq!(result.log_probs.shape, vec![8]);
    }

    #[test]
    fn test_value_shape() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![8, 4]);
        let values = ac.value(&obs).unwrap();
        assert_eq!(values.shape, vec![8]);
    }

    #[test]
    fn test_evaluate_shapes() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![4, 4]);
        let actions = TensorData::new(vec![0.0, 1.0, 0.0, 1.0], vec![4]);
        let eval = ac.evaluate(&obs, &actions).unwrap();
        assert_eq!(eval.log_probs.shape, vec![4]);
        assert_eq!(eval.entropy.shape, vec![4]);
        assert_eq!(eval.values.shape, vec![4]);
    }

    #[test]
    fn test_ppo_step_runs() {
        let mut ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let bs = 32;
        let obs = TensorData::zeros(vec![bs, 4]);
        let actions = TensorData::new(vec![0.0; bs], vec![bs]);
        let old_lp = TensorData::new(vec![-0.7; bs], vec![bs]);
        let adv = TensorData::new(vec![1.0; bs], vec![bs]);
        let ret = TensorData::new(vec![1.0; bs], vec![bs]);
        let old_v = TensorData::zeros(vec![bs]);
        let config = PPOStepConfig::default();

        let metrics = ac
            .ppo_step(&obs, &actions, &old_lp, &adv, &ret, &old_v, &config)
            .unwrap();
        assert!(metrics.get("policy_loss").is_some());
        assert!(metrics.get("value_loss").is_some());
        assert!(metrics.get("entropy").is_some());
    }

    #[test]
    fn test_lr_get_set() {
        let mut ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        assert!((ac.learning_rate() - 2.5e-4).abs() < 1e-8);
        ac.set_learning_rate(1e-3);
        assert!((ac.learning_rate() - 1e-3).abs() < 1e-8);
    }

    #[test]
    fn test_act_invalid_shape() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![4]); // 1D should fail
        assert!(ac.act(&obs).is_err());
    }

    #[test]
    fn test_act_rng_advances() {
        let ac = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();
        let obs = TensorData::zeros(vec![1, 4]);
        // Same obs, same logits → without RNG advance, same action every time.
        // With the fix, the RNG state advances so we may get different random draws.
        let mut seen_different = false;
        let first = ac.act(&obs).unwrap().log_probs.data[0];
        for _ in 0..20 {
            let lp = ac.act(&obs).unwrap().log_probs.data[0];
            if (lp - first).abs() > 1e-6 {
                seen_different = true;
                break;
            }
        }
        // With 2 actions and advancing RNG, probability of 20 identical draws is (0.5)^20 ≈ 1e-6
        assert!(seen_different, "RNG should advance between act() calls");
    }
}
