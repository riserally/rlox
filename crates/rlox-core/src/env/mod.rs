pub mod batch;
pub mod builtins;
pub mod mujoco;
pub mod parallel;
pub mod spaces;

use std::collections::HashMap;

pub use batch::BatchSteppable;
pub use spaces::{Action, ActionSpace, ObsSpace, Observation};

use crate::error::RloxError;

/// A single environment transition returned by `step`.
///
/// ## Precision convention
///
/// Rewards are `f64` for numerical stability during advantage computation;
/// observations and actions are `f32` throughout (see [`Observation`]).
/// When storing into replay buffers (which use `f32` rewards), a narrowing
/// cast occurs. This is intentional: environments compute in f64, buffers
/// store in f32, and training reads f32.
///
/// The `info` field is `None` when the environment provides no extra
/// metadata (the common case for CartPole, Pendulum, etc.), avoiding a
/// `HashMap` allocation on every step.
#[derive(Debug, Clone)]
pub struct Transition {
    pub obs: Observation,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Option<HashMap<String, f64>>,
}

/// The core environment trait.
///
/// All built-in environments implement this. The `Send + Sync` bounds
/// enable safe parallel stepping with Rayon.
pub trait RLEnv: Send + Sync {
    fn step(&mut self, action: &Action) -> Result<Transition, RloxError>;
    fn reset(&mut self, seed: Option<u64>) -> Result<Observation, RloxError>;
    fn action_space(&self) -> &ActionSpace;
    fn obs_space(&self) -> &ObsSpace;
    fn render(&self) -> Option<String> {
        None
    }
}
