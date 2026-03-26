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

/// Batched GAE: compute GAE for multiple environments in a single call.
///
/// All inputs are flat slices of length `n_envs * n_steps`, laid out as
/// `[env0_step0, env0_step1, ..., env1_step0, env1_step1, ...]`.
/// `last_values` has length `n_envs`.
///
/// Returns `(advantages, returns)` each of length `n_envs * n_steps`.
pub fn compute_gae_batched(
    rewards: &[f64],
    values: &[f64],
    dones: &[f64],
    last_values: &[f64],
    n_steps: usize,
    gamma: f64,
    gae_lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_envs = last_values.len();
    if n_envs == 0 || n_steps == 0 {
        return (Vec::new(), Vec::new());
    }

    use rayon::prelude::*;

    let mut all_advantages = vec![0.0; n_envs * n_steps];
    let mut all_returns = vec![0.0; n_envs * n_steps];

    all_advantages
        .par_chunks_mut(n_steps)
        .zip(all_returns.par_chunks_mut(n_steps))
        .enumerate()
        .for_each(|(env_idx, (adv_chunk, ret_chunk))| {
            let offset = env_idx * n_steps;
            let r = &rewards[offset..offset + n_steps];
            let v = &values[offset..offset + n_steps];
            let d = &dones[offset..offset + n_steps];
            let lv = last_values[env_idx];

            let mut last_gae = 0.0;
            for t in (0..n_steps).rev() {
                let next_non_terminal = 1.0 - d[t];
                let next_value = if t == n_steps - 1 { lv } else { v[t + 1] };
                let delta = r[t] + gamma * next_value * next_non_terminal - v[t];
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;
                adv_chunk[t] = last_gae;
                ret_chunk[t] = last_gae + v[t];
            }
        });

    (all_advantages, all_returns)
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

    #[test]
    fn gae_batched_matches_unbatched() {
        let gamma = 0.99;
        let lam = 0.95;
        // Two envs, 3 steps each
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let values = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let dones = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let last_values = vec![0.0, 0.5];

        let (adv_b, ret_b) = compute_gae_batched(&rewards, &values, &dones, &last_values, 3, gamma, lam);

        let (adv0, ret0) = compute_gae(&rewards[..3], &values[..3], &dones[..3], last_values[0], gamma, lam);
        let (adv1, ret1) = compute_gae(&rewards[3..], &values[3..], &dones[3..], last_values[1], gamma, lam);

        for i in 0..3 {
            assert!((adv_b[i] - adv0[i]).abs() < 1e-12, "env0 adv mismatch at {i}");
            assert!((ret_b[i] - ret0[i]).abs() < 1e-12, "env0 ret mismatch at {i}");
            assert!((adv_b[3 + i] - adv1[i]).abs() < 1e-12, "env1 adv mismatch at {i}");
            assert!((ret_b[3 + i] - ret1[i]).abs() < 1e-12, "env1 ret mismatch at {i}");
        }
    }

    #[test]
    fn gae_batched_empty() {
        let (adv, ret) = compute_gae_batched(&[], &[], &[], &[], 0, 0.99, 0.95);
        assert!(adv.is_empty());
        assert!(ret.is_empty());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn gae_returns_equal_advantages_plus_values(n in 1..500usize) {
                let rewards: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
                let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.05).collect();
                let dones: Vec<f64> = (0..n).map(|i| if i % 10 == 9 { 1.0 } else { 0.0 }).collect();
                let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);
                for i in 0..n {
                    let diff = (returns[i] - (advantages[i] + values[i])).abs();
                    prop_assert!(diff < 1e-10, "mismatch at index {}: returns={}, adv+val={}", i, returns[i], advantages[i] + values[i]);
                }
            }

            #[test]
            fn gae_length_matches_input(n in 0..500usize) {
                let rewards = vec![1.0; n];
                let values = vec![0.5; n];
                let dones = vec![0.0; n];
                let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);
                prop_assert_eq!(advantages.len(), n);
                prop_assert_eq!(returns.len(), n);
            }
        }
    }
}
