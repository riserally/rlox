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
/// # Panics
/// Panics if slices have different lengths.
pub fn compute_token_kl(log_probs_policy: &[f64], log_probs_ref: &[f64]) -> f64 {
    assert_eq!(
        log_probs_policy.len(),
        log_probs_ref.len(),
        "log_probs_policy and log_probs_ref must have the same length"
    );

    log_probs_policy
        .iter()
        .zip(log_probs_ref.iter())
        .map(|(&log_p, &log_q)| log_p.exp() * (log_p - log_q))
        .sum()
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
        let kl = compute_token_kl(&log_p, &log_p);
        assert!(kl.abs() < 1e-15);
    }

    #[test]
    fn test_token_kl_known_value() {
        // Manual: exp(-1) * (-1 - (-2)) = exp(-1) * 1 = 0.36787944...
        let log_p = [-1.0];
        let log_q = [-2.0];
        let kl = compute_token_kl(&log_p, &log_q);
        assert!((kl - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_token_kl_mismatched_lengths() {
        compute_token_kl(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn test_dpo_pair() {
        let pair = DPOPair::new(vec![1, 2, 3], vec![4, 5], vec![6, 7, 8]);
        assert_eq!(pair.chosen_len(), 2);
        assert_eq!(pair.rejected_len(), 3);
        assert_eq!(pair.prompt_tokens.len(), 3);
    }
}
