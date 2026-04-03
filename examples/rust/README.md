# Rust Examples

Standalone Rust programs demonstrating rlox-core without Python.

## Running

```bash
cd examples/rust
cargo run --bin cartpole        # CartPole with random actions
cargo run --bin pendulum        # Pendulum with continuous actions
cargo run --bin vec_env         # 64 parallel envs benchmark
cargo run --bin compute_gae     # GAE advantage computation
cargo run --bin replay_buffer   # Uniform + prioritized buffers
cargo run --bin reward_shaping  # PBRS + goal-distance potentials
```

## Examples

| Binary | What it demonstrates |
|--------|---------------------|
| `cartpole` | `RLEnv` trait, `CartPole`, discrete actions, episode tracking |
| `pendulum` | `Pendulum`, continuous actions (`Action::Continuous`) |
| `vec_env` | `VecEnv` parallel stepping with Rayon, throughput benchmark |
| `compute_gae` | `compute_gae` with known trajectory, invariant verification |
| `replay_buffer` | `ReplayBuffer` + `PrioritizedReplayBuffer`, push/sample |
| `reward_shaping` | `shape_rewards_pbrs`, `compute_goal_distance_potentials` |

## Using rlox-core in your project

```toml
# Cargo.toml
[dependencies]
rlox-core = { git = "https://github.com/riserally/rlox", package = "rlox-core" }
```

Or with a local checkout:

```toml
[dependencies]
rlox-core = { path = "path/to/rlox/crates/rlox-core" }
```
