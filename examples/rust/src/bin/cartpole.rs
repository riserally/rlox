//! Run CartPole with random actions and print episode rewards.
//!
//! ```bash
//! cd examples/rust && cargo run --bin cartpole
//! ```

use rlox_core::env::builtins::CartPole;
use rlox_core::env::spaces::Action;
use rlox_core::env::RLEnv;

fn main() {
    let mut env = CartPole::new(Some(42));
    let mut total_reward = 0.0;
    let mut episode = 1;
    let mut ep_steps = 0;

    let _obs = env.reset(Some(42)).unwrap();

    for step in 0..2000 {
        let action = Action::Discrete((step % 2) as u32);
        let transition = env.step(&action).unwrap();
        total_reward += transition.reward;
        ep_steps += 1;

        if transition.terminated || transition.truncated {
            println!("Episode {episode}: reward = {total_reward:.0}, steps = {ep_steps}");
            total_reward = 0.0;
            ep_steps = 0;
            episode += 1;
            let _obs = env.reset(None).unwrap();
        }
    }
}
