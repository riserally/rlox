//! Potential-based reward shaping (PBRS) and goal-distance potentials.
//!
//! Implements the PBRS transform `r' = r + gamma * Phi(s') - Phi(s)` which
//! preserves the optimal policy (Ng et al., 1999), plus goal-distance
//! potential computation for goal-conditioned RL.

use crate::error::RloxError;

/// Context passed to reward transforms for access to transition metadata.
pub struct RewardContext<'a> {
    pub dones: &'a [f64],
    pub gamma: f64,
    /// Arbitrary f64 slices keyed by name, for potential values, goal distances, etc.
    pub extras: &'a [(&'a str, &'a [f64])],
}

/// Trait for composable reward transformations.
///
/// Each transform takes raw rewards and a context and produces shaped rewards.
/// Transforms can be chained: `ClipReward -> PBRSTransform -> ScaleReward`.
pub trait RewardTransform: Send + Sync {
    /// Transform a batch of rewards.
    fn transform(
        &self,
        rewards: &[f64],
        context: &RewardContext<'_>,
    ) -> Result<Vec<f64>, RloxError>;

    /// Human-readable name.
    fn name(&self) -> &str;
}

/// Potential-based reward shaping transform.
///
/// Expects `extras` to contain entries named `"phi_current"` and `"phi_next"`.
pub struct PBRSTransform;

impl RewardTransform for PBRSTransform {
    fn transform(
        &self,
        rewards: &[f64],
        context: &RewardContext<'_>,
    ) -> Result<Vec<f64>, RloxError> {
        let phi_current = context
            .extras
            .iter()
            .find(|(name, _)| *name == "phi_current")
            .map(|(_, v)| *v)
            .ok_or_else(|| RloxError::BufferError("missing 'phi_current' in extras".into()))?;
        let phi_next = context
            .extras
            .iter()
            .find(|(name, _)| *name == "phi_next")
            .map(|(_, v)| *v)
            .ok_or_else(|| RloxError::BufferError("missing 'phi_next' in extras".into()))?;

        shape_rewards_pbrs(rewards, phi_current, phi_next, context.gamma, context.dones)
    }

    fn name(&self) -> &str {
        "PBRS"
    }
}

/// Goal-distance reward transform.
///
/// Computes `phi(s) = -scale * ||s[goal_slice] - goal||` and applies PBRS.
/// Expects `extras` to contain `"phi_current"` and `"phi_next"` entries
/// pre-computed via [`compute_goal_distance_potentials`].
pub struct GoalDistanceTransform {
    pub scale: f64,
    pub goal_start: usize,
    pub goal_dim: usize,
}

impl RewardTransform for GoalDistanceTransform {
    fn transform(
        &self,
        rewards: &[f64],
        context: &RewardContext<'_>,
    ) -> Result<Vec<f64>, RloxError> {
        let phi_current = context
            .extras
            .iter()
            .find(|(name, _)| *name == "phi_current")
            .map(|(_, v)| *v)
            .ok_or_else(|| RloxError::BufferError("missing 'phi_current' in extras".into()))?;
        let phi_next = context
            .extras
            .iter()
            .find(|(name, _)| *name == "phi_next")
            .map(|(_, v)| *v)
            .ok_or_else(|| RloxError::BufferError("missing 'phi_next' in extras".into()))?;

        shape_rewards_pbrs(rewards, phi_current, phi_next, context.gamma, context.dones)
    }

    fn name(&self) -> &str {
        "GoalDistance"
    }
}

/// Compute shaped rewards: `r' = r + gamma * Phi(s') - Phi(s)`
///
/// At episode boundaries (`dones[i] == 1.0`), the potential difference
/// is zeroed out: `r'_i = r_i` (no shaping across episode boundaries).
///
/// # Arguments
/// * `rewards` - raw rewards, length N
/// * `potentials_current` - Phi(s_t), length N
/// * `potentials_next` - Phi(s_{t+1}), length N
/// * `gamma` - discount factor
/// * `dones` - episode termination flags (1.0 = done), length N
#[inline]
pub fn shape_rewards_pbrs(
    rewards: &[f64],
    potentials_current: &[f64],
    potentials_next: &[f64],
    gamma: f64,
    dones: &[f64],
) -> Result<Vec<f64>, RloxError> {
    let n = rewards.len();
    if potentials_current.len() != n || potentials_next.len() != n || dones.len() != n {
        return Err(RloxError::ShapeMismatch {
            expected: format!("all slices length {n}"),
            got: format!(
                "phi_current={}, phi_next={}, dones={}",
                potentials_current.len(),
                potentials_next.len(),
                dones.len()
            ),
        });
    }

    let mut output = Vec::with_capacity(n);
    for i in 0..n {
        if dones[i] == 1.0 {
            output.push(rewards[i]);
        } else {
            output.push(rewards[i] + gamma * potentials_next[i] - potentials_current[i]);
        }
    }
    Ok(output)
}

/// Goal-distance potential: `Phi(s) = -scale * ||s[goal_slice] - goal||_2`
///
/// # Arguments
/// * `observations` - flat `(N * obs_dim)` array
/// * `goal` - target goal vector, length `goal_dim`
/// * `obs_dim` - dimensionality of each observation
/// * `goal_start` - starting index within obs where goal-relevant dims begin
/// * `goal_dim` - number of goal-relevant dimensions
/// * `scale` - scaling factor for the potential
#[inline]
pub fn compute_goal_distance_potentials(
    observations: &[f64],
    goal: &[f64],
    obs_dim: usize,
    goal_start: usize,
    goal_dim: usize,
    scale: f64,
) -> Result<Vec<f64>, RloxError> {
    if goal.len() != goal_dim {
        return Err(RloxError::ShapeMismatch {
            expected: format!("goal.len() == goal_dim={goal_dim}"),
            got: format!("goal.len()={}", goal.len()),
        });
    }
    if obs_dim == 0 {
        return Err(RloxError::ShapeMismatch {
            expected: "obs_dim > 0".into(),
            got: "obs_dim=0".into(),
        });
    }
    if !observations.len().is_multiple_of(obs_dim) {
        return Err(RloxError::ShapeMismatch {
            expected: format!("observations.len() divisible by obs_dim={obs_dim}"),
            got: format!("observations.len()={}", observations.len()),
        });
    }
    if goal_start + goal_dim > obs_dim {
        return Err(RloxError::ShapeMismatch {
            expected: format!("goal_start + goal_dim <= obs_dim={obs_dim}"),
            got: format!("goal_start={goal_start}, goal_dim={goal_dim}"),
        });
    }

    let n = observations.len() / obs_dim;
    let mut potentials = Vec::with_capacity(n);

    for i in 0..n {
        let obs_start = i * obs_dim + goal_start;
        let obs_slice = &observations[obs_start..obs_start + goal_dim];
        let dist_sq: f64 = obs_slice
            .iter()
            .zip(goal.iter())
            .map(|(&o, &g)| (o - g) * (o - g))
            .sum();
        potentials.push(-scale * dist_sq.sqrt());
    }

    Ok(potentials)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbrs_known_values() {
        let rewards = &[1.0, 2.0];
        let phi = &[0.5, 0.3];
        let phi_next = &[0.3, 0.8];
        let gamma = 0.99;
        let dones = &[0.0, 0.0];
        let result = shape_rewards_pbrs(rewards, phi, phi_next, gamma, dones).unwrap();
        // r'[0] = 1.0 + 0.99*0.3 - 0.5 = 0.797
        assert!(
            (result[0] - 0.797).abs() < 1e-10,
            "expected 0.797, got {}",
            result[0]
        );
        // r'[1] = 2.0 + 0.99*0.8 - 0.3 = 2.492
        assert!(
            (result[1] - 2.492).abs() < 1e-10,
            "expected 2.492, got {}",
            result[1]
        );
    }

    #[test]
    fn test_pbrs_done_resets_potential() {
        let rewards = &[1.0, 2.0];
        let phi = &[0.5, 0.3];
        let phi_next = &[0.3, 0.8];
        let gamma = 0.99;
        let dones = &[0.0, 1.0];
        let result = shape_rewards_pbrs(rewards, phi, phi_next, gamma, dones).unwrap();
        assert!(
            (result[1] - 2.0).abs() < 1e-10,
            "done step should return raw reward, got {}",
            result[1]
        );
    }

    #[test]
    fn test_pbrs_zero_potentials_no_change() {
        let rewards = &[1.0, 2.0, 3.0];
        let phi = &[0.0, 0.0, 0.0];
        let phi_next = &[0.0, 0.0, 0.0];
        let dones = &[0.0, 0.0, 0.0];
        let result = shape_rewards_pbrs(rewards, phi, phi_next, 0.99, dones).unwrap();
        for i in 0..3 {
            assert!(
                (result[i] - rewards[i]).abs() < 1e-10,
                "zero potentials should not change reward"
            );
        }
    }

    #[test]
    fn test_pbrs_preserves_optimal_policy() {
        // For a complete episode, sum(shaped) = sum(raw) + gamma*Phi(s_T) - Phi(s_0)
        // With done at the last step, the last step gets raw reward
        let rewards = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let phi = &[0.5, 0.3, 0.8, 0.1, 0.9];
        let phi_next = &[0.3, 0.8, 0.1, 0.9, 0.0];
        let gamma = 0.99;
        // No dones (all within episode)
        let dones = &[0.0, 0.0, 0.0, 0.0, 0.0];
        let shaped = shape_rewards_pbrs(rewards, phi, phi_next, gamma, dones).unwrap();
        let sum_raw: f64 = rewards.iter().sum();
        let sum_shaped: f64 = shaped.iter().sum();
        // Telescoping sum: sum(gamma*phi_next[i] - phi[i]) for non-done steps
        let shaping_sum: f64 = (0..5).map(|i| gamma * phi_next[i] - phi[i]).sum::<f64>();
        assert!(
            (sum_shaped - (sum_raw + shaping_sum)).abs() < 1e-10,
            "sum_shaped={sum_shaped}, expected {}",
            sum_raw + shaping_sum
        );
    }

    #[test]
    fn test_pbrs_length_mismatch_errors() {
        let result = shape_rewards_pbrs(
            &[1.0, 2.0, 3.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            0.99,
            &[0.0, 0.0],
        );
        assert!(matches!(result, Err(RloxError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_goal_distance_decreasing_near_goal() {
        let goal = &[0.0, 0.0];
        // obs far from goal, obs close to goal
        let observations = &[
            1.0, 0.0, 1.0, 0.0, // obs_dim=4, goal at [0..2], far (dist=1)
            0.1, 0.0, 0.1, 0.0, // close (dist=0.1)
        ];
        let potentials =
            compute_goal_distance_potentials(observations, goal, 4, 0, 2, 1.0).unwrap();
        assert!(
            potentials[1] > potentials[0],
            "closer obs should have less negative potential: far={}, close={}",
            potentials[0],
            potentials[1]
        );
    }

    #[test]
    fn test_goal_distance_at_goal_is_zero() {
        let goal = &[3.0, 4.0];
        let observations = &[3.0, 4.0, 0.0, 0.0]; // obs_dim=4, goal at [0..2]
        let potentials =
            compute_goal_distance_potentials(observations, goal, 4, 0, 2, 1.0).unwrap();
        assert!(
            potentials[0].abs() < 1e-10,
            "at goal, potential should be 0, got {}",
            potentials[0]
        );
    }

    #[test]
    fn test_goal_distance_scale_factor() {
        let goal = &[0.0];
        let observations = &[1.0, 0.0]; // obs_dim=2, goal at [0..1]
        let phi_1 = compute_goal_distance_potentials(observations, goal, 2, 0, 1, 1.0).unwrap();
        let phi_2 = compute_goal_distance_potentials(observations, goal, 2, 0, 1, 2.0).unwrap();
        assert!(
            (phi_2[0] - 2.0 * phi_1[0]).abs() < 1e-10,
            "scale=2 should double potential: phi_1={}, phi_2={}",
            phi_1[0],
            phi_2[0]
        );
    }

    #[test]
    fn test_goal_distance_validates_dimensions() {
        let result = compute_goal_distance_potentials(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0, 0.0, 0.0], // goal_dim=3 but we say 2
            4,
            0,
            2,
            1.0,
        );
        assert!(matches!(result, Err(RloxError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_trait_object_safety() {
        let transform: Box<dyn RewardTransform> = Box::new(PBRSTransform);
        assert_eq!(transform.name(), "PBRS");
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_pbrs_length_matches_input(n in 1usize..500) {
                let rewards: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
                let phi: Vec<f64> = vec![0.5; n];
                let phi_next: Vec<f64> = vec![0.3; n];
                let dones: Vec<f64> = vec![0.0; n];
                let result = shape_rewards_pbrs(&rewards, &phi, &phi_next, 0.99, &dones).unwrap();
                prop_assert_eq!(result.len(), n);
            }

            #[test]
            fn prop_pbrs_zero_gamma_no_future(n in 1usize..100) {
                let rewards: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let phi: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
                let phi_next: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
                let dones: Vec<f64> = vec![0.0; n];
                let result = shape_rewards_pbrs(&rewards, &phi, &phi_next, 0.0, &dones).unwrap();
                for i in 0..n {
                    let expected = rewards[i] - phi[i]; // gamma=0 -> no phi_next term
                    prop_assert!(
                        (result[i] - expected).abs() < 1e-10,
                        "index {i}: got {}, expected {expected}",
                        result[i]
                    );
                }
            }

            #[test]
            fn prop_goal_distance_non_positive(
                n in 1usize..50,
                goal_dim in 1usize..4,
            ) {
                let obs_dim = goal_dim + 2;
                let obs: Vec<f64> = (0..(n * obs_dim)).map(|i| i as f64 * 0.1).collect();
                let goal: Vec<f64> = vec![0.0; goal_dim];
                let potentials = compute_goal_distance_potentials(
                    &obs, &goal, obs_dim, 0, goal_dim, 1.0
                ).unwrap();
                for (i, &p) in potentials.iter().enumerate() {
                    prop_assert!(p <= 0.0 + 1e-10,
                        "potential[{i}] = {p} should be <= 0 for positive scale");
                }
            }
        }
    }
}
