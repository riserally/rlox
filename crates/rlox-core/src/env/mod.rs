pub mod batch;
pub mod builtins;
pub mod parallel;
pub mod spaces;

use std::collections::HashMap;

pub use batch::BatchSteppable;
pub use spaces::{Action, ActionSpace, ObsSpace, Observation};

use crate::error::RloxError;

/// A single environment transition returned by `step`.
#[derive(Debug, Clone)]
pub struct Transition {
    pub obs: Observation,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: HashMap<String, f64>,
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
