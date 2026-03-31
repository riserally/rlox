//! MuJoCo environment bindings.
//!
//! Gated behind the `mujoco` feature flag. When the flag is not enabled,
//! this module provides [`SimplifiedMuJoCoEnv`] -- a placeholder that
//! implements `RLEnv` with basic linear dynamics, allowing the full
//! architecture (VecEnv, PyVecEnv factory, training loops) to be validated
//! without the MuJoCo C library installed.

#[cfg(feature = "mujoco")]
mod inner {
    // TODO: Full MuJoCo implementation using `mujoco-rs` crate.
    //
    // When the `mujoco` feature is enabled, this module will provide:
    //   - `MuJoCoEnv`: a generic wrapper around any MuJoCo XML model
    //   - Pre-built constructors for standard envs (HalfCheetah, Ant, Hopper, etc.)
    //   - Proper contact physics, joint limits, and reward shaping
    //
    // For now, re-export the simplified env so downstream code compiles
    // regardless of feature state.
    pub use super::simplified::SimplifiedMuJoCoEnv;
}

#[cfg(not(feature = "mujoco"))]
mod inner {
    pub use super::simplified::SimplifiedMuJoCoEnv;
}

pub use inner::*;

mod simplified {
    use std::collections::HashMap;

    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    use crate::env::spaces::{Action, ActionSpace, ObsSpace, Observation};
    use crate::env::{RLEnv, Transition};
    use crate::error::RloxError;
    use crate::seed::rng_from_seed;

    // ---------------------------------------------------------------------------
    // HalfCheetah-v4 dimensions (matching Gymnasium / MuJoCo)
    // ---------------------------------------------------------------------------

    /// Observation dimensionality for HalfCheetah-v4.
    ///
    /// 17 = 8 joint positions (excluding rootx) + 9 joint velocities.
    const OBS_DIM: usize = 17;

    /// Action dimensionality for HalfCheetah-v4.
    ///
    /// 6 torque actuators on the 6 non-root joints.
    const ACT_DIM: usize = 6;

    /// Integration timestep (seconds).
    const DT: f64 = 0.05;

    /// Maximum episode length before truncation.
    const MAX_STEPS: u32 = 1000;

    /// Control cost coefficient (matching Gymnasium default).
    const CTRL_COST_WEIGHT: f64 = 0.1;

    /// Simplified MuJoCo-like environment with HalfCheetah-v4 dimensions.
    ///
    /// Uses basic linear dynamics as a placeholder:
    ///   `next_state = state + dt * f(action)`
    ///
    /// This is **not** physically accurate -- it exists solely to validate
    /// the trait implementation, VecEnv integration, and Python bindings
    /// before the real MuJoCo backend is wired up.
    pub struct SimplifiedMuJoCoEnv {
        /// Internal state vector (17-dim).
        state: Vec<f64>,
        rng: ChaCha8Rng,
        steps: u32,
        action_space: ActionSpace,
        obs_space: ObsSpace,
        done: bool,
        /// Previous x-position for velocity reward computation.
        prev_x_pos: f64,
    }

    impl SimplifiedMuJoCoEnv {
        /// Create a new simplified HalfCheetah-v4 environment.
        pub fn new(seed: Option<u64>) -> Self {
            let seed_val = seed.unwrap_or(0);
            let rng = rng_from_seed(seed_val);

            let action_low = vec![-1.0_f32; ACT_DIM];
            let action_high = vec![1.0_f32; ACT_DIM];

            // Observation bounds: generous range to avoid clipping during rollouts.
            let obs_low = vec![-f32::INFINITY; OBS_DIM];
            let obs_high = vec![f32::INFINITY; OBS_DIM];

            let mut env = SimplifiedMuJoCoEnv {
                state: vec![0.0; OBS_DIM],
                rng,
                steps: 0,
                action_space: ActionSpace::Box {
                    low: action_low,
                    high: action_high,
                    shape: vec![ACT_DIM],
                },
                obs_space: ObsSpace::Box {
                    low: obs_low,
                    high: obs_high,
                    shape: vec![OBS_DIM],
                },
                done: true,
                prev_x_pos: 0.0,
            };
            let _ = env.reset(Some(seed_val));
            env
        }

        /// Build observation from current state.
        fn obs(&self) -> Observation {
            Observation::Flat(self.state.iter().map(|&v| v as f32).collect())
        }

        /// Forward velocity (used as the primary reward signal).
        ///
        /// In real HalfCheetah this comes from the MuJoCo simulator's
        /// `(x_after - x_before) / dt`. Here we approximate it from the
        /// velocity component of the state (index 8 = rootx velocity in the
        /// simplified model).
        fn forward_velocity(&self) -> f64 {
            // state[8] represents the x-velocity in our simplified layout.
            // Indices 0..8 are joint positions, 8..17 are joint velocities.
            self.state[8]
        }
    }

    impl RLEnv for SimplifiedMuJoCoEnv {
        fn step(&mut self, action: &Action) -> Result<Transition, RloxError> {
            if self.done {
                return Err(RloxError::EnvError(
                    "Environment is done. Call reset() before stepping.".into(),
                ));
            }

            let torques = match action {
                Action::Continuous(vals) if vals.len() == ACT_DIM => vals,
                _ => {
                    return Err(RloxError::InvalidAction(format!(
                        "HalfCheetah expects a Continuous action with {} elements",
                        ACT_DIM
                    )));
                }
            };

            // --- Simplified dynamics ---
            // Position updates: affected by current velocities
            // state[0..8] = joint positions, state[8..17] = joint velocities
            //
            // Velocity updates: affected by actions (torques) on joints 0..6
            // This is a gross simplification -- real MuJoCo solves the full
            // equations of motion with contacts, inertia, Coriolis forces, etc.

            // Update velocities from torques (joints 0..6 map to actions 0..6)
            for i in 0..ACT_DIM {
                let torque = (torques[i] as f64).clamp(-1.0, 1.0);
                self.state[8 + i] += DT * torque;
            }

            // Update positions from velocities
            for i in 0..8 {
                let vel_idx = 8 + i.min(OBS_DIM - 9);
                self.state[i] += DT * self.state[vel_idx];
            }

            self.steps += 1;

            // --- Reward ---
            // HalfCheetah-v4 reward = forward_velocity - ctrl_cost
            let forward_vel = self.forward_velocity();
            let ctrl_cost: f64 = CTRL_COST_WEIGHT
                * torques.iter().map(|&t| (t as f64) * (t as f64)).sum::<f64>();
            let reward = forward_vel - ctrl_cost;

            // HalfCheetah never terminates early, only truncates at max_steps.
            let truncated = self.steps >= MAX_STEPS;
            self.done = truncated;

            Ok(Transition {
                obs: self.obs(),
                reward,
                terminated: false,
                truncated,
                info: {
                    let mut info = HashMap::new();
                    info.insert("x_velocity".to_string(), forward_vel);
                    info.insert("reward_forward".to_string(), forward_vel);
                    info.insert("reward_ctrl".to_string(), -ctrl_cost);
                    info
                },
            })
        }

        fn reset(&mut self, seed: Option<u64>) -> Result<Observation, RloxError> {
            if let Some(s) = seed {
                self.rng = rng_from_seed(s);
            }

            // Initialize state with small random values (matching Gymnasium's
            // `init_qpos + noise` and `init_qvel + noise` strategy).
            for s in self.state.iter_mut() {
                *s = self.rng.random_range(-0.1..0.1);
            }

            self.steps = 0;
            self.done = false;
            self.prev_x_pos = 0.0;

            Ok(self.obs())
        }

        fn action_space(&self) -> &ActionSpace {
            &self.action_space
        }

        fn obs_space(&self) -> &ObsSpace {
            &self.obs_space
        }

        fn render(&self) -> Option<String> {
            Some(format!(
                "SimplifiedHalfCheetah | step={} | x_vel={:.4}",
                self.steps,
                self.forward_velocity()
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::SimplifiedMuJoCoEnv;
    use crate::env::parallel::VecEnv;
    use crate::env::spaces::{Action, ActionSpace, ObsSpace};
    use crate::env::RLEnv;
    use crate::seed::derive_seed;

    fn zero_action() -> Action {
        Action::Continuous(vec![0.0; 6])
    }

    fn random_action(seed: u32) -> Action {
        // Deterministic pseudo-random action for testing
        let vals: Vec<f32> = (0..6)
            .map(|i| ((seed as f32 + i as f32) * 0.31415).sin() * 0.8)
            .collect();
        Action::Continuous(vals)
    }

    // ----- Observation shape -----

    #[test]
    fn obs_dim_is_17() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        let obs = env.reset(Some(42)).unwrap();
        assert_eq!(obs.as_slice().len(), 17, "HalfCheetah obs must be 17-dim");
    }

    #[test]
    fn obs_space_shape_is_17() {
        let env = SimplifiedMuJoCoEnv::new(Some(42));
        match env.obs_space() {
            ObsSpace::Box { shape, .. } => {
                assert_eq!(shape, &[17]);
            }
            other => panic!("Expected Box obs space, got {:?}", other),
        }
    }

    // ----- Action shape -----

    #[test]
    fn action_dim_is_6() {
        let env = SimplifiedMuJoCoEnv::new(Some(42));
        match env.action_space() {
            ActionSpace::Box { low, high, shape } => {
                assert_eq!(shape, &[6]);
                assert_eq!(low.len(), 6);
                assert_eq!(high.len(), 6);
                for (&lo, &hi) in low.iter().zip(high.iter()) {
                    assert!((lo - (-1.0)).abs() < f32::EPSILON);
                    assert!((hi - 1.0).abs() < f32::EPSILON);
                }
            }
            other => panic!("Expected Box action space, got {:?}", other),
        }
    }

    // ----- Reset -----

    #[test]
    fn reset_returns_valid_obs() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        let obs = env.reset(Some(99)).unwrap();
        assert_eq!(obs.as_slice().len(), 17);
        // Initial state should be small random values in [-0.1, 0.1]
        for &v in obs.as_slice() {
            assert!(
                v.abs() <= 0.1 + f32::EPSILON,
                "initial obs element out of range: {}",
                v
            );
        }
    }

    #[test]
    fn reset_clears_step_counter() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        // Step a few times
        for _ in 0..10 {
            env.step(&zero_action()).unwrap();
        }
        // Reset and step again -- should not truncate immediately
        env.reset(Some(42)).unwrap();
        let t = env.step(&zero_action()).unwrap();
        assert!(!t.truncated);
    }

    // ----- Step produces valid output -----

    #[test]
    fn step_returns_17_dim_obs() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        let t = env.step(&zero_action()).unwrap();
        assert_eq!(t.obs.as_slice().len(), 17);
    }

    #[test]
    fn step_never_terminates() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        for _ in 0..1000 {
            let t = env.step(&random_action(0)).unwrap();
            assert!(!t.terminated, "HalfCheetah should never terminate early");
            if t.truncated {
                break;
            }
        }
    }

    // ----- Truncation at 1000 steps -----

    #[test]
    fn truncates_at_1000_steps() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        for i in 0..1000 {
            let t = env.step(&zero_action()).unwrap();
            if i < 999 {
                assert!(!t.truncated, "should not truncate at step {}", i + 1);
            } else {
                assert!(t.truncated, "should truncate at step 1000");
            }
        }
        // Stepping after truncation should error
        let result = env.step(&zero_action());
        assert!(result.is_err());
    }

    // ----- Reward structure -----

    #[test]
    fn zero_action_gives_zero_ctrl_cost() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        let t = env.step(&zero_action()).unwrap();
        // With zero action, ctrl_cost = 0, so reward = forward_velocity
        let x_vel = t.info.get("x_velocity").copied().unwrap_or(0.0);
        let ctrl = t.info.get("reward_ctrl").copied().unwrap_or(0.0);
        assert!(
            ctrl.abs() < 1e-10,
            "ctrl cost should be ~0 for zero action, got {}",
            ctrl
        );
        assert!(
            (t.reward - x_vel).abs() < 1e-10,
            "reward should equal x_velocity when ctrl_cost=0"
        );
    }

    #[test]
    fn nonzero_action_incurs_ctrl_cost() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        let action = Action::Continuous(vec![0.5; 6]);
        let t = env.step(&action).unwrap();
        let ctrl = t.info.get("reward_ctrl").copied().unwrap_or(0.0);
        // ctrl_cost = 0.1 * sum(0.5^2 * 6) = 0.1 * 1.5 = 0.15
        assert!(ctrl < 0.0, "ctrl reward should be negative, got {}", ctrl);
    }

    // ----- Invalid actions -----

    #[test]
    fn discrete_action_rejected() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        let result = env.step(&Action::Discrete(0));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_dim_action_rejected() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        let result = env.step(&Action::Continuous(vec![0.0; 3]));
        assert!(result.is_err());
    }

    // ----- Seeded determinism -----

    #[test]
    fn seeded_determinism() {
        let run = |seed: u64| -> Vec<f64> {
            let mut env = SimplifiedMuJoCoEnv::new(Some(seed));
            let mut rewards = Vec::with_capacity(100);
            for i in 0..100 {
                let t = env.step(&random_action(i)).unwrap();
                rewards.push(t.reward);
            }
            rewards
        };

        let r1 = run(123);
        let r2 = run(123);
        assert_eq!(r1, r2, "same seed must produce identical trajectories");

        let r3 = run(456);
        assert_ne!(r1, r3, "different seeds should produce different trajectories");
    }

    // ----- Step after done errors -----

    #[test]
    fn step_after_done_errors() {
        let mut env = SimplifiedMuJoCoEnv::new(Some(42));
        // Run to truncation
        for _ in 0..1000 {
            let _ = env.step(&zero_action()).unwrap();
        }
        let result = env.step(&zero_action());
        assert!(result.is_err());
    }

    // ----- VecEnv integration -----

    #[test]
    fn vec_env_with_multiple_half_cheetahs() {
        let n = 4;
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| {
                let s = derive_seed(42, i);
                Box::new(SimplifiedMuJoCoEnv::new(Some(s))) as Box<dyn RLEnv>
            })
            .collect();

        let mut venv = VecEnv::new(envs);
        assert_eq!(venv.num_envs(), 4);

        // Step all
        let actions: Vec<Action> = (0..n).map(|i| random_action(i as u32)).collect();
        let batch = venv.step_all(&actions).unwrap();

        assert_eq!(batch.obs.len(), 4);
        assert_eq!(batch.rewards.len(), 4);
        assert_eq!(batch.terminated.len(), 4);
        assert_eq!(batch.truncated.len(), 4);

        for obs in &batch.obs {
            assert_eq!(obs.len(), 17, "each env obs must be 17-dim");
        }
    }

    #[test]
    fn vec_env_flat_stepping() {
        let n = 4;
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| {
                let s = derive_seed(42, i);
                Box::new(SimplifiedMuJoCoEnv::new(Some(s))) as Box<dyn RLEnv>
            })
            .collect();

        let mut venv = VecEnv::new(envs);
        let actions: Vec<Action> = (0..n).map(|_| zero_action()).collect();
        let batch = venv.step_all_flat(&actions).unwrap();

        assert!(batch.obs.is_empty());
        assert_eq!(batch.obs_flat.len(), 4 * 17);
        assert_eq!(batch.obs_dim, 17);
    }

    #[test]
    fn vec_env_auto_reset_across_truncation() {
        let n = 2;
        let envs: Vec<Box<dyn RLEnv>> = (0..n)
            .map(|i| {
                let s = derive_seed(42, i);
                Box::new(SimplifiedMuJoCoEnv::new(Some(s))) as Box<dyn RLEnv>
            })
            .collect();

        let mut venv = VecEnv::new(envs);
        let actions: Vec<Action> = (0..n).map(|_| zero_action()).collect();

        // Step past truncation -- VecEnv auto-resets so this should not error
        for _ in 0..1100 {
            let batch = venv.step_all(&actions).unwrap();
            assert_eq!(batch.obs.len(), 2);
        }
    }

    #[test]
    fn vec_env_determinism() {
        let run = || {
            let n = 8;
            let envs: Vec<Box<dyn RLEnv>> = (0..n)
                .map(|i| {
                    let s = derive_seed(42, i);
                    Box::new(SimplifiedMuJoCoEnv::new(Some(s))) as Box<dyn RLEnv>
                })
                .collect();

            let mut venv = VecEnv::new(envs);
            venv.reset_all(Some(42)).unwrap();

            let actions: Vec<Action> = (0..n).map(|i| random_action(i as u32)).collect();
            let mut all_rewards = Vec::new();
            for _ in 0..50 {
                let batch = venv.step_all(&actions).unwrap();
                all_rewards.extend(batch.rewards);
            }
            all_rewards
        };

        let r1 = run();
        let r2 = run();
        assert_eq!(r1, r2);
    }
}
