//! Parallel environment stepping with VecEnv.
//!
//! Demonstrates creating 64 parallel CartPole environments and
//! stepping them all at once using Rayon work-stealing.
//!
//! ```bash
//! cd examples/rust && cargo run --bin vec_env
//! ```

use rlox_core::env::builtins::CartPole;
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::spaces::Action;
use rlox_core::env::RLEnv;
use rlox_core::seed::derive_seed;
use std::time::Instant;

fn main() {
    let n_envs = 64;

    // Create parallel environments with deterministic seeding
    let envs: Vec<Box<dyn RLEnv>> = (0..n_envs)
        .map(|i| Box::new(CartPole::new(Some(derive_seed(42, i)))) as Box<dyn RLEnv>)
        .collect();
    let mut vec_env = VecEnv::new(envs).unwrap();

    let _observations = vec_env.reset_all(Some(42)).unwrap();
    println!("Created {n_envs} parallel CartPole environments");

    // Benchmark: step 100K times
    let n_steps = 100_000;
    let start = Instant::now();

    for step in 0..n_steps {
        let actions: Vec<Action> = (0..n_envs)
            .map(|i| Action::Discrete(((step + i) % 2) as u32))
            .collect();
        let _batch = vec_env.step_all(&actions).unwrap();
    }

    let elapsed = start.elapsed();
    let total_steps = n_steps * n_envs;
    let sps = total_steps as f64 / elapsed.as_secs_f64();
    println!(
        "{total_steps} total steps in {:.2}s = {sps:.0} steps/sec",
        elapsed.as_secs_f64()
    );
}
