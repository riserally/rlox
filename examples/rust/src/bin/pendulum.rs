//! Run native Pendulum-v1 with continuous actions.
//!
//! Demonstrates the continuous action space environment
//! with random torque inputs.
//!
//! ```bash
//! cd examples/rust && cargo run --bin pendulum
//! ```

use rlox_core::env::builtins::Pendulum;
use rlox_core::env::spaces::Action;
use rlox_core::env::RLEnv;

fn main() {
    let mut env = Pendulum::new(Some(42));
    let obs = env.reset(Some(42)).unwrap();
    println!("Pendulum obs (cos θ, sin θ, ω): {:?}", obs.as_slice());

    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        // Random torque in [-2.0, 2.0]
        let torque = (steps as f32 * 0.1).sin() * 2.0;
        let action = Action::Continuous(vec![torque]);
        let t = env.step(&action).unwrap();
        total_reward += t.reward;
        steps += 1;

        if steps % 50 == 0 {
            println!(
                "  step {steps:3}: obs=[{:.2}, {:.2}, {:.2}], reward={:.2}, torque={:.2}",
                t.obs.as_slice()[0],
                t.obs.as_slice()[1],
                t.obs.as_slice()[2],
                t.reward,
                torque
            );
        }

        if t.truncated {
            println!("\nEpisode done after {steps} steps, total reward: {total_reward:.1}");
            break;
        }
    }
}
