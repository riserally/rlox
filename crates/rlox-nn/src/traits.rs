use std::path::Path;

use crate::error::NNError;
use crate::tensor_data::TensorData;

/// Action output from a policy.
#[derive(Debug, Clone)]
pub struct ActionOutput {
    /// Actions: [batch_size] for discrete, [batch_size, act_dim] for continuous.
    pub actions: TensorData,
    /// Log-probabilities of the selected actions: [batch_size].
    pub log_probs: TensorData,
}

/// Output from evaluating a policy on (obs, actions) pairs.
#[derive(Debug, Clone)]
pub struct EvalOutput {
    /// Log-probabilities: [batch_size].
    pub log_probs: TensorData,
    /// Entropy of the distribution: [batch_size].
    pub entropy: TensorData,
    /// State values: [batch_size].
    pub values: TensorData,
}

/// PPO step configuration.
#[derive(Debug, Clone)]
pub struct PPOStepConfig {
    pub clip_eps: f32,
    pub vf_coef: f32,
    pub ent_coef: f32,
    pub max_grad_norm: f32,
    pub clip_vloss: bool,
}

impl Default for PPOStepConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            max_grad_norm: 0.5,
            clip_vloss: true,
        }
    }
}

/// DQN step configuration.
#[derive(Debug, Clone)]
pub struct DQNStepConfig {
    pub gamma: f32,
    pub n_step: usize,
    pub double_dqn: bool,
}

impl Default for DQNStepConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            n_step: 1,
            double_dqn: true,
        }
    }
}

/// SAC step configuration.
#[derive(Debug, Clone)]
pub struct SACStepConfig {
    pub gamma: f32,
    pub tau: f32,
    pub target_entropy: f32,
    pub auto_entropy: bool,
}

impl Default for SACStepConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            tau: 0.005,
            target_entropy: -1.0,
            auto_entropy: true,
        }
    }
}

/// TD3 step configuration.
#[derive(Debug, Clone)]
pub struct TD3StepConfig {
    pub gamma: f32,
    pub tau: f32,
    pub policy_delay: usize,
    pub target_noise: f32,
    pub noise_clip: f32,
}

impl Default for TD3StepConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            tau: 0.005,
            policy_delay: 2,
            target_noise: 0.2,
            noise_clip: 0.5,
        }
    }
}

/// Training metrics dictionary.
#[derive(Debug, Clone, Default)]
pub struct TrainMetrics {
    pub entries: Vec<(String, f64)>,
}

impl TrainMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: f64) {
        self.entries.push((key.into(), value));
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }
}

// ────────────────────────────────────────────────────────────
// Phase 1 Traits: Actor-Critic (PPO/A2C) and Q-Function (DQN)
// ────────────────────────────────────────────────────────────

/// Actor-Critic policy for on-policy algorithms (PPO, A2C).
pub trait ActorCritic {
    /// Sample actions from the policy (inference, no gradient tracking).
    fn act(&self, obs: &TensorData) -> Result<ActionOutput, NNError>;

    /// Compute state values (inference, no gradient tracking).
    fn value(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    /// Evaluate the policy on (obs, actions) pairs. Differentiable.
    fn evaluate(&self, obs: &TensorData, actions: &TensorData) -> Result<EvalOutput, NNError>;

    /// Perform one PPO gradient step. Bundles forward→loss→backward→clip→step.
    #[allow(clippy::too_many_arguments)]
    fn ppo_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        old_log_probs: &TensorData,
        advantages: &TensorData,
        returns: &TensorData,
        old_values: &TensorData,
        config: &PPOStepConfig,
    ) -> Result<TrainMetrics, NNError>;

    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);

    fn save(&self, path: &Path) -> Result<(), NNError>;
    fn load(&mut self, path: &Path) -> Result<(), NNError>;
}

/// Q-value network for off-policy algorithms (DQN).
pub trait QFunction {
    /// Compute Q-values for all actions. Returns [batch_size, n_actions].
    fn q_values(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    /// Compute Q-value for (obs, action) pairs. Returns [batch_size].
    fn q_value_at(&self, obs: &TensorData, actions: &TensorData) -> Result<TensorData, NNError>;

    /// Perform one DQN TD gradient step.
    /// Returns (loss, td_errors) where td_errors can be used for PER.
    fn td_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        targets: &TensorData,
        weights: Option<&TensorData>,
    ) -> Result<(f64, TensorData), NNError>;

    /// Compute target Q-values using the target network.
    fn target_q_values(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    /// Hard-copy parameters to the target network.
    fn hard_update_target(&mut self);
}

// ────────────────────────────────────────────────────────────
// Phase 2 Traits: Stochastic Policy (SAC) and Deterministic (TD3)
// ────────────────────────────────────────────────────────────

/// Continuous stochastic policy for SAC.
///
/// Training steps (`sac_actor_step`) are intentionally NOT on this trait because
/// they require autograd to flow through the critic's Q-network. Trait methods
/// convert tensors to `TensorData` (Vec<f32>), severing the computation graph.
/// Use the backend-specific inherent `sac_actor_step` method instead.
pub trait StochasticPolicy {
    /// Sample actions with reparameterization trick.
    /// Returns (squashed_actions [batch, act_dim], log_probs [batch]).
    fn sample_actions(&self, obs: &TensorData) -> Result<(TensorData, TensorData), NNError>;

    /// Deterministic action (mean through squashing).
    fn deterministic_action(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);

    fn save(&self, path: &Path) -> Result<(), NNError>;
    fn load(&mut self, path: &Path) -> Result<(), NNError>;
}

/// Continuous Q-function for SAC/TD3 (takes obs + action as input).
pub trait ContinuousQFunction {
    /// Compute Q-value for (obs, action). Returns [batch_size].
    fn q_value(&self, obs: &TensorData, actions: &TensorData) -> Result<TensorData, NNError>;

    /// Compute twin Q-values for (obs, action). Returns (q1, q2), each [batch_size].
    fn twin_q_values(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<(TensorData, TensorData), NNError>;

    /// Compute target twin Q-values (from target networks).
    fn target_twin_q_values(
        &self,
        obs: &TensorData,
        actions: &TensorData,
    ) -> Result<(TensorData, TensorData), NNError>;

    /// Perform one TD gradient step on both critics.
    fn critic_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        targets: &TensorData,
    ) -> Result<TrainMetrics, NNError>;

    /// Polyak soft update of target networks.
    fn soft_update_targets(&mut self, tau: f32);
}

/// Deterministic policy for TD3.
///
/// Training steps (`td3_actor_step`) are intentionally NOT on this trait because
/// they require autograd to flow through the critic's Q-network. Trait methods
/// convert tensors to `TensorData` (Vec<f32>), severing the computation graph.
/// Use the backend-specific inherent `td3_actor_step` method instead.
pub trait DeterministicPolicy {
    /// Compute deterministic action. Returns [batch_size, act_dim].
    fn act(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    /// Compute target policy action (from target network).
    fn target_act(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    /// Polyak soft update of target network.
    fn soft_update_target(&mut self, tau: f32);

    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);

    fn save(&self, path: &Path) -> Result<(), NNError>;
    fn load(&mut self, path: &Path) -> Result<(), NNError>;
}

/// Entropy tuning for SAC.
pub trait EntropyTuner {
    fn alpha(&self) -> f32;
    fn update(&mut self, log_probs: &TensorData, target_entropy: f32) -> Result<f64, NNError>;
}

// ────────────────────────────────────────────────────────────
// Network builder config (shared across backends)
// ────────────────────────────────────────────────────────────

/// Activation function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Tanh,
}

/// Configuration for building an MLP.
#[derive(Debug, Clone)]
pub struct MLPConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub activation: Activation,
    pub output_activation: Option<Activation>,
}

impl MLPConfig {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            hidden_dims: vec![64, 64],
            activation: Activation::Tanh,
            output_activation: None,
        }
    }

    pub fn with_hidden(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    pub fn with_activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    pub fn with_output_activation(mut self, act: Activation) -> Self {
        self.output_activation = Some(act);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_config_default() {
        let cfg = PPOStepConfig::default();
        assert!((cfg.clip_eps - 0.2).abs() < 1e-6);
        assert!((cfg.vf_coef - 0.5).abs() < 1e-6);
        assert!((cfg.ent_coef - 0.01).abs() < 1e-6);
        assert!((cfg.max_grad_norm - 0.5).abs() < 1e-6);
        assert!(cfg.clip_vloss);
    }

    #[test]
    fn test_train_metrics() {
        let mut m = TrainMetrics::new();
        m.insert("loss", 0.5);
        m.insert("entropy", 1.2);
        assert_eq!(m.get("loss"), Some(0.5));
        assert_eq!(m.get("entropy"), Some(1.2));
        assert_eq!(m.get("missing"), None);
    }

    #[test]
    fn test_mlp_config_builder() {
        let cfg = MLPConfig::new(4, 2)
            .with_hidden(vec![128, 128])
            .with_activation(Activation::ReLU)
            .with_output_activation(Activation::Tanh);
        assert_eq!(cfg.input_dim, 4);
        assert_eq!(cfg.output_dim, 2);
        assert_eq!(cfg.hidden_dims, vec![128, 128]);
        assert_eq!(cfg.activation, Activation::ReLU);
        assert_eq!(cfg.output_activation, Some(Activation::Tanh));
    }

    #[test]
    fn test_action_output_shape() {
        let out = ActionOutput {
            actions: TensorData::zeros(vec![8]),
            log_probs: TensorData::zeros(vec![8]),
        };
        assert_eq!(out.actions.shape, vec![8]);
        assert_eq!(out.log_probs.shape, vec![8]);
    }

    #[test]
    fn test_eval_output_shape() {
        let out = EvalOutput {
            log_probs: TensorData::zeros(vec![32]),
            entropy: TensorData::zeros(vec![32]),
            values: TensorData::zeros(vec![32]),
        };
        assert_eq!(out.log_probs.numel(), 32);
        assert_eq!(out.entropy.numel(), 32);
        assert_eq!(out.values.numel(), 32);
    }

    // Compile-time trait object safety checks
    fn _assert_actor_critic_object_safe(_: &dyn ActorCritic) {}
    fn _assert_q_function_object_safe(_: &dyn QFunction) {}
    fn _assert_stochastic_policy_object_safe(_: &dyn StochasticPolicy) {}
    fn _assert_continuous_q_object_safe(_: &dyn ContinuousQFunction) {}
    fn _assert_deterministic_policy_object_safe(_: &dyn DeterministicPolicy) {}
    fn _assert_entropy_tuner_object_safe(_: &dyn EntropyTuner) {}
}
