use crate::env::parallel::BatchTransition;
use crate::env::spaces::{Action, ActionSpace, ObsSpace, Observation};
use crate::error::RloxError;

/// Trait for anything that can step a batch of environments.
///
/// Separates the parallelism strategy from step logic, so that both
/// Rust-native `VecEnv` (Rayon) and future Python-backed `GymVecEnv`
/// can share a common interface for rollout collectors and training loops.
///
/// The `Send` bound enables use from async / threaded contexts.
pub trait BatchSteppable: Send {
    /// Step all environments with the given actions (one per env).
    fn step_batch(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError>;

    /// Reset all environments, optionally seeding them deterministically.
    fn reset_batch(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, RloxError>;

    /// Number of sub-environments in this batch.
    fn num_envs(&self) -> usize;

    /// The shared action space (all sub-environments must have the same space).
    fn action_space(&self) -> &ActionSpace;

    /// The shared observation space.
    fn obs_space(&self) -> &ObsSpace;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::builtins::CartPole;
    use crate::env::parallel::VecEnv;
    use crate::env::RLEnv;
    use crate::seed::derive_seed;

    fn make_batch(n: usize, seed: u64) -> VecEnv {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| Box::new(CartPole::new(Some(derive_seed(seed, i)))) as Box<dyn RLEnv>)
            .collect();
        VecEnv::new(envs)
    }

    /// Compile-time proof that `BatchSteppable` is object-safe.
    fn _assert_object_safe(_: &dyn BatchSteppable) {}

    #[test]
    fn test_vecenv_implements_batch_steppable() {
        let mut batch: Box<dyn BatchSteppable> = Box::new(make_batch(4, 42));
        let _obs = batch.reset_batch(Some(42)).unwrap();

        let actions: Vec<Action> = (0..4).map(|i| Action::Discrete((i % 2) as u32)).collect();
        let result = batch.step_batch(&actions).unwrap();

        assert_eq!(result.obs.len(), 4);
        assert_eq!(result.rewards.len(), 4);
        assert_eq!(result.terminated.len(), 4);
        assert_eq!(result.truncated.len(), 4);
    }

    #[test]
    fn test_batch_steppable_action_space_propagates() {
        let batch = make_batch(4, 42);
        let steppable: &dyn BatchSteppable = &batch;
        assert_eq!(steppable.action_space(), &ActionSpace::Discrete(2));
    }

    #[test]
    fn test_batch_steppable_wrong_action_count() {
        let mut batch: Box<dyn BatchSteppable> = Box::new(make_batch(4, 42));
        let actions = vec![Action::Discrete(0); 3]; // 3 actions for 4 envs
        let result = batch.step_batch(&actions);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_steppable_is_object_safe() {
        // If this compiles, the trait is object-safe.
        fn _check(_: &dyn BatchSteppable) {}
    }
}
