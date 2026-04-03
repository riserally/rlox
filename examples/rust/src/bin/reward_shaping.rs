//! Potential-based reward shaping (PBRS) demonstration.
//!
//! Shows how shaped rewards preserve the optimal policy
//! while accelerating learning.
//!
//! ```bash
//! cd examples/rust && cargo run --bin reward_shaping
//! ```

use rlox_core::training::reward_shaping::{shape_rewards_pbrs, compute_goal_distance_potentials};

fn main() {
    // --- PBRS with custom potentials ---
    let rewards = vec![1.0, 1.0, 1.0, 1.0, 0.0];
    let phi_current = vec![0.5, 0.6, 0.7, 0.8, 0.9]; // potential of current state
    let phi_next = vec![0.6, 0.7, 0.8, 0.9, 0.0];     // potential of next state
    let gamma = 0.99;
    let dones = vec![0.0, 0.0, 0.0, 0.0, 1.0];

    let shaped = shape_rewards_pbrs(&rewards, &phi_current, &phi_next, gamma, &dones).unwrap();

    println!("PBRS: r' = r + γΦ(s') - Φ(s)");
    println!("Step | Raw    | Shaped | Φ(s)  | Φ(s')");
    println!("-----|--------|--------|-------|------");
    for i in 0..rewards.len() {
        println!(
            "  {i}  | {:.3}  | {:.3}  | {:.2}  | {:.2}{}",
            rewards[i],
            shaped[i],
            phi_current[i],
            phi_next[i],
            if dones[i] == 1.0 { "  (done — raw reward only)" } else { "" }
        );
    }

    // --- Goal-distance potentials ---
    println!("\nGoal-distance potentials: Φ(s) = -scale * ||s - goal||");

    let obs_dim = 3;
    let observations = vec![
        0.0, 0.0, 0.0, // far from goal
        0.5, 0.5, 0.5, // closer
        0.9, 0.9, 0.9, // very close
        1.0, 1.0, 1.0, // at goal
    ];
    let goal = vec![1.0, 1.0, 1.0];

    let potentials = compute_goal_distance_potentials(
        &observations,
        &goal,
        obs_dim,
        0,         // goal_start: goal-relevant dims start at index 0
        obs_dim,   // goal_dim: all 3 dims are goal-relevant
        1.0,       // scale
    ).unwrap();

    println!("Obs         | Distance | Potential");
    println!("------------|----------|----------");
    for i in 0..4 {
        let obs = &observations[i * obs_dim..(i + 1) * obs_dim];
        let dist: f64 = obs.iter().zip(goal.iter()).map(|(o, g)| (o - g).powi(2)).sum::<f64>().sqrt();
        println!("  [{:.1}, {:.1}, {:.1}] | {dist:.3}    | {:.3}", obs[0], obs[1], obs[2], potentials[i]);
    }
    println!("\nCloser to goal → less negative potential → higher shaped reward");
}
