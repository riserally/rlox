//! Compute GAE advantages from a sample trajectory.
//!
//! Shows how to use the Rust GAE implementation directly,
//! matching the 140x speedup over Python loops.
//!
//! ```bash
//! cd examples/rust && cargo run --bin compute_gae
//! ```

use rlox_core::training::gae::compute_gae;

fn main() {
    // Simulated trajectory: 10 steps with a terminal at step 7
    let rewards = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let values = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 0.3, 0.5, 0.6];
    let dones = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let gamma = 0.99;
    let gae_lambda = 0.95;
    let last_value = 0.7; // V(s_{T+1}) bootstrap

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    println!("Step | Reward | Value  | Advantage | Return");
    println!("-----|--------|--------|-----------|-------");
    for t in 0..rewards.len() {
        println!(
            "  {t:2} |  {:.2}  | {:.3}  |  {:+.4}  | {:.4}",
            rewards[t], values[t], advantages[t], returns[t]
        );
    }

    // Verify invariant: returns[t] == advantages[t] + values[t]
    for t in 0..rewards.len() {
        let diff = (returns[t] - (advantages[t] + values[t])).abs();
        assert!(diff < 1e-10, "Invariant violated at step {t}");
    }
    println!("\nInvariant verified: returns[t] == advantages[t] + values[t]");
}
