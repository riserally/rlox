/// GRPO group advantage: `(reward - mean) / std`.
/// Returns zeros if std < 1e-8.
pub fn compute_group_advantages(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }

    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let variance = rewards.iter().map(|&r| (r - mean) * (r - mean)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std < 1e-8 {
        return vec![0.0; rewards.len()];
    }

    let inv_std = 1.0 / std;
    rewards.iter().map(|&r| (r - mean) * inv_std).collect()
}

/// Token-level KL divergence: `sum(exp(log_p) * (log_p - log_q))`.
///
/// Returns `Err` if slices have different lengths.
pub fn compute_token_kl(
    log_probs_policy: &[f64],
    log_probs_ref: &[f64],
) -> Result<f64, crate::error::RloxError> {
    if log_probs_policy.len() != log_probs_ref.len() {
        return Err(crate::error::RloxError::ShapeMismatch {
            expected: format!("len={}", log_probs_policy.len()),
            got: format!("len={}", log_probs_ref.len()),
        });
    }

    Ok(log_probs_policy
        .iter()
        .zip(log_probs_ref.iter())
        .map(|(&log_p, &log_q)| log_p.exp() * (log_p - log_q))
        .sum())
}

/// Batched GRPO group advantages: process all groups in a single call.
///
/// `rewards` is a flat slice of length `n_prompts * group_size`.
/// Returns a Vec of the same length with per-group z-score normalisation.
pub fn compute_batch_group_advantages(rewards: &[f64], group_size: usize) -> Result<Vec<f64>, crate::error::RloxError> {
    if group_size == 0 {
        return Err(crate::error::RloxError::ShapeMismatch {
            expected: "group_size > 0".to_string(),
            got: "0".to_string(),
        });
    }
    if rewards.len() % group_size != 0 {
        return Err(crate::error::RloxError::ShapeMismatch {
            expected: format!("len divisible by {group_size}"),
            got: format!("len={}", rewards.len()),
        });
    }

    let mut out = Vec::with_capacity(rewards.len());
    for group in rewards.chunks_exact(group_size) {
        out.extend_from_slice(&compute_group_advantages(group));
    }
    Ok(out)
}

/// Token-level KL divergence using the Schulman (2020) estimator:
/// `sum(exp(log_p - log_q) - (log_p - log_q) - 1)`.
///
/// This is the estimator used by TRL (HuggingFace). It is unbiased and
/// numerically more stable than the exact `exp(log_p) * (log_p - log_q)`.
pub fn compute_token_kl_schulman(
    log_probs_policy: &[f64],
    log_probs_ref: &[f64],
) -> Result<f64, crate::error::RloxError> {
    if log_probs_policy.len() != log_probs_ref.len() {
        return Err(crate::error::RloxError::ShapeMismatch {
            expected: format!("len={}", log_probs_policy.len()),
            got: format!("len={}", log_probs_ref.len()),
        });
    }

    Ok(log_probs_policy
        .iter()
        .zip(log_probs_ref.iter())
        .map(|(&log_p, &log_q)| {
            let r = log_p - log_q;
            r.exp() - r - 1.0
        })
        .sum())
}

/// A DPO preference pair holding tokenized prompt, chosen, and rejected sequences.
#[derive(Debug, Clone)]
pub struct DPOPair {
    pub prompt_tokens: Vec<u32>,
    pub chosen_tokens: Vec<u32>,
    pub rejected_tokens: Vec<u32>,
}

impl DPOPair {
    pub fn new(prompt_tokens: Vec<u32>, chosen_tokens: Vec<u32>, rejected_tokens: Vec<u32>) -> Self {
        Self {
            prompt_tokens,
            chosen_tokens,
            rejected_tokens,
        }
    }

    pub fn chosen_len(&self) -> usize {
        self.chosen_tokens.len()
    }

    pub fn rejected_len(&self) -> usize {
        self.rejected_tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_advantages_basic() {
        let rewards = [1.0, 0.5, 0.8];
        let adv = compute_group_advantages(&rewards);
        assert_eq!(adv.len(), 3);
        // Mean of advantages should be ~0
        let mean: f64 = adv.iter().sum::<f64>() / adv.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_group_advantages_constant_rewards() {
        let rewards = [5.0, 5.0, 5.0];
        let adv = compute_group_advantages(&rewards);
        assert!(adv.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_group_advantages_empty() {
        let adv = compute_group_advantages(&[]);
        assert!(adv.is_empty());
    }

    #[test]
    fn test_token_kl_identical() {
        let log_p = [-1.0, -2.0, -0.5];
        let kl = compute_token_kl(&log_p, &log_p).unwrap();
        assert!(kl.abs() < 1e-15);
    }

    #[test]
    fn test_token_kl_known_value() {
        // Manual: exp(-1) * (-1 - (-2)) = exp(-1) * 1 = 0.36787944...
        let log_p = [-1.0];
        let log_q = [-2.0];
        let kl = compute_token_kl(&log_p, &log_q).unwrap();
        assert!((kl - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_token_kl_mismatched_lengths_returns_err() {
        let result = compute_token_kl(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    // --- Bug 0.3 verification tests ---

    #[test]
    fn token_kl_mismatched_lengths_returns_err_not_panic() {
        let log_p = vec![-1.0f64, -2.0];
        let log_q = vec![-1.0f64];
        let result = compute_token_kl(&log_p, &log_q);
        assert!(result.is_err(), "mismatched lengths must return Err");
    }

    #[test]
    fn token_kl_matching_lengths_returns_ok() {
        let log_p = vec![-1.0f64, -2.0, -0.5];
        let log_q = vec![-1.0f64, -2.0, -0.5];
        let result = compute_token_kl(&log_p, &log_q);
        assert!(result.is_ok());
        assert!(result.unwrap().abs() < 1e-15);
    }

    #[test]
    fn token_kl_empty_slices_returns_zero() {
        let result = compute_token_kl(&[], &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn token_kl_nan_input_propagates_to_output() {
        let log_p = vec![f64::NAN];
        let log_q = vec![-1.0f64];
        let result = compute_token_kl(&log_p, &log_q);
        match result {
            Ok(v) => assert!(v.is_nan(), "NaN input should produce NaN output"),
            Err(_) => {}
        }
    }

    #[test]
    fn token_kl_inf_input_does_not_panic() {
        let log_p = vec![f64::INFINITY];
        let log_q = vec![-1.0f64];
        let _result = compute_token_kl(&log_p, &log_q);
    }

    #[test]
    fn token_kl_known_value_still_correct_after_refactor() {
        let log_p = vec![-1.0f64];
        let log_q = vec![-2.0f64];
        let kl = compute_token_kl(&log_p, &log_q).unwrap();
        assert!((kl - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_batch_group_advantages() {
        // Two groups of 3: [1,2,3] and [10,10,10]
        let rewards = [1.0, 2.0, 3.0, 10.0, 10.0, 10.0];
        let adv = compute_batch_group_advantages(&rewards, 3).unwrap();
        assert_eq!(adv.len(), 6);
        // First group should have mean ~0
        let g1_mean: f64 = adv[..3].iter().sum::<f64>() / 3.0;
        assert!(g1_mean.abs() < 1e-10);
        // Second group (constant) should be zeros
        assert!(adv[3..6].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_batch_group_advantages_bad_size() {
        assert!(compute_batch_group_advantages(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(compute_batch_group_advantages(&[1.0], 0).is_err());
    }

    #[test]
    fn test_token_kl_schulman_identical() {
        let log_p = [-1.0, -2.0, -0.5];
        let kl = compute_token_kl_schulman(&log_p, &log_p).unwrap();
        assert!(kl.abs() < 1e-15);
    }

    #[test]
    fn test_token_kl_schulman_known_value() {
        // r = log_p - log_q = -1 - (-2) = 1
        // schulman: exp(1) - 1 - 1 = e - 2 ≈ 0.71828
        let log_p = [-1.0];
        let log_q = [-2.0];
        let kl = compute_token_kl_schulman(&log_p, &log_q).unwrap();
        assert!((kl - (1.0_f64.exp() - 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_token_kl_schulman_non_negative() {
        // Schulman estimator is always >= 0 for any r (convexity of exp)
        let log_p = [-0.5, -1.0, -3.0, 0.0];
        let log_q = [-1.0, -0.5, -0.1, -2.0];
        let kl = compute_token_kl_schulman(&log_p, &log_q).unwrap();
        assert!(kl >= 0.0, "Schulman KL should be non-negative, got {kl}");
    }

    #[test]
    fn test_dpo_pair() {
        let pair = DPOPair::new(vec![1, 2, 3], vec![4, 5], vec![6, 7, 8]);
        assert_eq!(pair.chosen_len(), 2);
        assert_eq!(pair.rejected_len(), 3);
        assert_eq!(pair.prompt_tokens.len(), 3);
    }
}
