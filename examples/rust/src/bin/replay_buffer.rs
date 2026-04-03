//! Replay buffer push and sample demonstration.
//!
//! Shows ReplayBuffer (uniform) and PrioritizedReplayBuffer (PER)
//! with deterministic seeding for reproducibility.
//!
//! ```bash
//! cd examples/rust && cargo run --bin replay_buffer
//! ```

use rlox_core::buffer::ringbuf::ReplayBuffer;
use rlox_core::buffer::priority::PrioritizedReplayBuffer;
use rlox_core::buffer::ExperienceRecord;

fn main() {
    // --- Uniform Replay Buffer ---
    let obs_dim = 4;
    let act_dim = 1;
    let mut buf = ReplayBuffer::new(10_000, obs_dim, act_dim);

    // Fill with synthetic data
    for i in 0..5_000u32 {
        buf.push(ExperienceRecord {
            obs: vec![i as f32; obs_dim],
            next_obs: vec![(i + 1) as f32; obs_dim],
            action: vec![(i % 2) as f32],
            reward: if i % 100 == 0 { 10.0 } else { 1.0 },
            terminated: i % 500 == 499,
            truncated: false,
        })
        .unwrap();
    }
    println!("Uniform buffer: {} transitions stored", buf.len());

    // Sample a batch (deterministic with seed)
    let batch = buf.sample(32, 42).unwrap();
    println!(
        "Sampled batch: {} obs, {} rewards",
        batch.observations.len() / obs_dim,
        batch.rewards.len()
    );

    // --- Prioritized Replay Buffer ---
    let mut per = PrioritizedReplayBuffer::new(10_000, obs_dim, act_dim, 0.6, 0.4);

    for i in 0..5_000u32 {
        let priority = if i % 100 == 0 { 5.0 } else { 1.0 }; // high priority for rare events
        per.push(
            ExperienceRecord {
                obs: vec![i as f32; obs_dim],
                next_obs: vec![(i + 1) as f32; obs_dim],
                action: vec![0.0],
                reward: 1.0,
                terminated: false,
                truncated: false,
            },
            priority,
        )
        .unwrap();
    }
    println!("\nPrioritized buffer: {} transitions", per.len());

    let per_batch = per.sample(32, 42).unwrap();
    println!(
        "PER batch: {} samples, IS weights range [{:.3}, {:.3}]",
        per_batch.batch_size,
        per_batch.weights.iter().cloned().fold(f64::INFINITY, f64::min),
        per_batch.weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
}
