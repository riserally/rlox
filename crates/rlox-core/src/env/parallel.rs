use rayon::prelude::*;

use crate::env::spaces::{Action, Observation};
use crate::env::{RLEnv, Transition};
use crate::error::RloxError;
use crate::seed::derive_seed;

/// Columnar batch of transitions from parallel stepping.
#[derive(Debug, Clone)]
pub struct BatchTransition {
    /// Observations: `[num_envs][obs_dim]`
    pub obs: Vec<Vec<f32>>,
    /// Rewards: `[num_envs]`
    pub rewards: Vec<f64>,
    /// Terminated flags: `[num_envs]`
    pub terminated: Vec<bool>,
    /// Truncated flags: `[num_envs]`
    pub truncated: Vec<bool>,
}

/// A vectorized environment that steps multiple sub-environments in parallel.
pub struct VecEnv {
    envs: Vec<Box<dyn RLEnv>>,
}

impl VecEnv {
    pub fn new(envs: Vec<Box<dyn RLEnv>>) -> Self {
        VecEnv { envs }
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Step all environments in parallel using Rayon.
    ///
    /// If an environment is done after stepping, it is automatically reset
    /// and the returned observation is from the fresh episode.
    pub fn step_all(&mut self, actions: &[Action]) -> Result<BatchTransition, RloxError> {
        if actions.len() != self.envs.len() {
            return Err(RloxError::ShapeMismatch {
                expected: format!("{}", self.envs.len()),
                got: format!("{}", actions.len()),
            });
        }

        let results: Vec<Result<Transition, RloxError>> = self
            .envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, action)| {
                let mut transition = env.step(action)?;
                // Auto-reset on done
                if transition.terminated || transition.truncated {
                    let new_obs = env.reset(None)?;
                    transition.obs = new_obs;
                }
                Ok(transition)
            })
            .collect();

        // Unpack results into columnar format
        let n = results.len();
        let mut obs = Vec::with_capacity(n);
        let mut rewards = Vec::with_capacity(n);
        let mut terminated = Vec::with_capacity(n);
        let mut truncated = Vec::with_capacity(n);

        for result in results {
            let t = result?;
            obs.push(t.obs.into_inner());
            rewards.push(t.reward);
            terminated.push(t.terminated);
            truncated.push(t.truncated);
        }

        Ok(BatchTransition {
            obs,
            rewards,
            terminated,
            truncated,
        })
    }

    /// Reset all environments, optionally seeding them deterministically.
    ///
    /// When a master seed is provided, each env `i` gets `derive_seed(master, i)`.
    pub fn reset_all(&mut self, seed: Option<u64>) -> Result<Vec<Observation>, RloxError> {
        self.envs
            .iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let env_seed = seed.map(|s| derive_seed(s, i));
                env.reset(env_seed)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::builtins::CartPole;

    fn make_vec_env(n: usize, seed: u64) -> VecEnv {
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| {
                let s = derive_seed(seed, i);
                Box::new(CartPole::new(Some(s))) as Box<dyn RLEnv>
            })
            .collect();
        VecEnv::new(envs)
    }

    #[test]
    fn vec_env_num_envs() {
        let venv = make_vec_env(4, 42);
        assert_eq!(venv.num_envs(), 4);
    }

    #[test]
    fn vec_env_step_all_returns_correct_shapes() {
        let mut venv = make_vec_env(4, 42);
        let actions: Vec<Action> = (0..4).map(|i| Action::Discrete((i % 2) as u32)).collect();
        let batch = venv.step_all(&actions).unwrap();
        assert_eq!(batch.obs.len(), 4);
        assert_eq!(batch.rewards.len(), 4);
        assert_eq!(batch.terminated.len(), 4);
        assert_eq!(batch.truncated.len(), 4);
        for obs in &batch.obs {
            assert_eq!(obs.len(), 4);
        }
    }

    #[test]
    fn vec_env_step_all_wrong_action_count() {
        let mut venv = make_vec_env(4, 42);
        let actions = vec![Action::Discrete(0); 3];
        let result = venv.step_all(&actions);
        assert!(result.is_err());
    }

    #[test]
    fn vec_env_reset_all_deterministic() {
        let mut venv1 = make_vec_env(4, 0);
        let mut venv2 = make_vec_env(4, 0);

        let obs1 = venv1.reset_all(Some(99)).unwrap();
        let obs2 = venv2.reset_all(Some(99)).unwrap();

        for (o1, o2) in obs1.iter().zip(obs2.iter()) {
            assert_eq!(o1.as_slice(), o2.as_slice());
        }
    }

    #[test]
    fn vec_env_large_parallel_stepping() {
        // Validate 256+ envs step correctly in parallel
        let mut venv = make_vec_env(256, 42);
        let actions: Vec<Action> = (0..256).map(|i| Action::Discrete((i % 2) as u32)).collect();
        let batch = venv.step_all(&actions).unwrap();
        assert_eq!(batch.obs.len(), 256);
        assert_eq!(batch.rewards.len(), 256);
        // All rewards should be 1.0 (first step)
        for &r in &batch.rewards {
            assert!((r - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn vec_env_1024_envs_no_panic() {
        // Ensure 1024 parallel envs don't cause thread pool issues
        let mut venv = make_vec_env(1024, 42);
        let actions: Vec<Action> =
            (0..1024).map(|i| Action::Discrete((i % 2) as u32)).collect();
        // Step 10 times
        for _ in 0..10 {
            let batch = venv.step_all(&actions).unwrap();
            assert_eq!(batch.obs.len(), 1024);
        }
    }

    #[test]
    fn vec_env_parallel_determinism() {
        // Parallel stepping must be deterministic across runs
        let run = || {
            let mut venv = make_vec_env(64, 42);
            venv.reset_all(Some(42)).unwrap();
            let actions: Vec<Action> =
                (0..64).map(|i| Action::Discrete((i % 2) as u32)).collect();
            let mut all_rewards = Vec::new();
            for _ in 0..50 {
                let batch = venv.step_all(&actions).unwrap();
                all_rewards.extend(batch.rewards);
            }
            all_rewards
        };
        let run1 = run();
        let run2 = run();
        assert_eq!(run1, run2);
    }

    #[test]
    fn vec_env_auto_reset_on_done() {
        let mut venv = make_vec_env(2, 42);

        // Step many times - eventually envs will terminate and auto-reset
        for _ in 0..100 {
            let actions: Vec<Action> = (0..2).map(|_| Action::Discrete(1)).collect();
            match venv.step_all(&actions) {
                Ok(_batch) => {} // should always succeed due to auto-reset
                Err(e) => panic!("step_all should not error with auto-reset: {}", e),
            }
        }
    }
}
