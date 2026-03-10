/// Compute Generalized Advantage Estimation.
///
/// Iterates backwards over the rollout, computing:
///   delta_t = reward_t + gamma * V(t+1) * (1 - done_t) - V(t)
///   A_t = delta_t + gamma * lambda * (1 - done_t) * A(t+1)
///   return_t = A_t + V(t)
///
/// The `dones` slice uses `f64` where 0.0 = not done, 1.0 = done,
/// matching the common Python/numpy convention.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[f64],
    last_value: f64,
    gamma: f64,
    gae_lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut advantages = vec![0.0; n];
    let mut last_gae = 0.0;

    for t in (0..n).rev() {
        let next_non_terminal = 1.0 - dones[t];
        let next_value = if t == n - 1 {
            last_value
        } else {
            values[t + 1]
        };
        let delta = rewards[t] + gamma * next_value * next_non_terminal - values[t];
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;
        advantages[t] = last_gae;
    }

    let returns: Vec<f64> = advantages
        .iter()
        .zip(values.iter())
        .map(|(a, v)| a + v)
        .collect();

    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to convert bool slices to f64 for the test interface
    fn bools_to_f64(bools: &[bool]) -> Vec<f64> {
        bools.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
    }

    #[test]
    fn gae_single_step_episode() {
        let rewards = &[1.0];
        let values = &[0.5];
        let dones = bools_to_f64(&[true]);
        let last_value = 0.0;
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let (advantages, _returns) = compute_gae(rewards, values, &dones, last_value, gamma, gae_lambda);
        assert_eq!(advantages.len(), 1);
        assert!((advantages[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gae_multi_step_no_termination() {
        let rewards = &[1.0, 1.0, 1.0];
        let values = &[0.0, 0.0, 0.0];
        let dones = bools_to_f64(&[false, false, false]);
        let last_value = 0.0;
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let (advantages, _returns) = compute_gae(rewards, values, &dones, last_value, gamma, gae_lambda);
        assert_eq!(advantages.len(), 3);
        // Last step: delta = 1.0 + 0.99*0 - 0 = 1.0, A = 1.0
        assert!((advantages[2] - 1.0).abs() < 1e-6);
        // Second step: delta = 1.0, A = 1.0 + 0.99*0.95*1.0 = 1.9405
        assert!((advantages[1] - 1.9405).abs() < 1e-4);
        // First step: A = 1.0 + 0.99*0.95*1.9405 = 2.82...
        assert!(advantages[0] > advantages[1]);
    }

    #[test]
    fn gae_resets_at_episode_boundary() {
        let rewards = &[1.0, 1.0, 1.0];
        let values = &[0.0, 0.0, 0.0];
        let dones = bools_to_f64(&[false, true, false]);
        let last_value = 0.0;
        let gamma = 0.99;
        let gae_lambda = 0.95;
        let (advantages, _) = compute_gae(rewards, values, &dones, last_value, gamma, gae_lambda);
        // Step 1 (terminal): delta = 1.0 + 0 - 0 = 1.0
        assert!((advantages[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gae_returns_are_advantages_plus_values() {
        let rewards = &[1.0, 2.0, 3.0];
        let values = &[0.5, 1.0, 1.5];
        let dones = bools_to_f64(&[false, false, true]);
        let last_value = 0.0;
        let (advantages, returns) = compute_gae(rewards, values, &dones, last_value, 0.99, 0.95);
        for i in 0..3 {
            assert!((returns[i] - (advantages[i] + values[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn gae_empty_input() {
        let (advantages, returns) = compute_gae(&[], &[], &[], 0.0, 0.99, 0.95);
        assert!(advantages.is_empty());
        assert!(returns.is_empty());
    }

    #[test]
    fn gae_lambda_zero_is_one_step_td() {
        let rewards = &[1.0, 1.0];
        let values = &[0.5, 0.5];
        let dones = bools_to_f64(&[false, false]);
        let last_value = 0.5;
        let (advantages, _) = compute_gae(rewards, values, &dones, last_value, 0.99, 0.0);
        // delta_1 = 1.0 + 0.99*0.5 - 0.5 = 0.995, advantage = delta (lambda=0)
        assert!((advantages[1] - 0.995).abs() < 1e-6);
    }

    #[test]
    fn gae_lambda_one_is_monte_carlo() {
        let rewards = &[1.0, 1.0, 1.0];
        let values = &[0.0, 0.0, 0.0];
        let dones = bools_to_f64(&[false, false, true]);
        let (advantages, _) = compute_gae(rewards, values, &dones, 0.0, 0.99, 1.0);
        // Monte Carlo return from step 0: 1 + 0.99 + 0.99^2 = 2.9701
        assert!((advantages[0] - 2.9701).abs() < 1e-3);
    }
}
