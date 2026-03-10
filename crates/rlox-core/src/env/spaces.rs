use serde::{Deserialize, Serialize};

/// Describes the action space of an environment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActionSpace {
    /// A single discrete action in `0..n`.
    Discrete(usize),
    /// A continuous box space with per-dimension bounds.
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
        shape: Vec<usize>,
    },
    /// Multiple discrete sub-spaces, each with its own cardinality.
    MultiDiscrete(Vec<usize>),
}

/// Describes the observation space of an environment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ObsSpace {
    Discrete(usize),
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
        shape: Vec<usize>,
    },
    MultiDiscrete(Vec<usize>),
}

/// A concrete action value.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Discrete(u32),
    Continuous(Vec<f32>),
}

/// A concrete observation value (flat float array for v0.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Observation(pub Vec<f32>);

impl Observation {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }
}

impl ActionSpace {
    /// Check whether an action is valid for this space.
    pub fn contains(&self, action: &Action) -> bool {
        match (self, action) {
            (ActionSpace::Discrete(n), Action::Discrete(a)) => (*a as usize) < *n,
            (ActionSpace::Box { low, high, .. }, Action::Continuous(vals)) => {
                vals.len() == low.len()
                    && vals
                        .iter()
                        .zip(low.iter().zip(high.iter()))
                        .all(|(v, (lo, hi))| *v >= *lo && *v <= *hi)
            }
            _ => false,
        }
    }
}

impl ObsSpace {
    /// Check whether an observation is valid for this space.
    pub fn contains(&self, obs: &Observation) -> bool {
        match self {
            ObsSpace::Discrete(n) => obs.0.len() == 1 && (obs.0[0] as usize) < *n,
            ObsSpace::Box { low, high, .. } => {
                obs.0.len() == low.len()
                    && obs
                        .0
                        .iter()
                        .zip(low.iter().zip(high.iter()))
                        .all(|(v, (lo, hi))| *v >= *lo && *v <= *hi)
            }
            ObsSpace::MultiDiscrete(nvec) => {
                obs.0.len() == nvec.len()
                    && obs
                        .0
                        .iter()
                        .zip(nvec.iter())
                        .all(|(v, n)| *v >= 0.0 && (*v as usize) < *n)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_action_space_contains() {
        let space = ActionSpace::Discrete(3);
        assert!(space.contains(&Action::Discrete(0)));
        assert!(space.contains(&Action::Discrete(2)));
        assert!(!space.contains(&Action::Discrete(3)));
        assert!(!space.contains(&Action::Continuous(vec![0.0])));
    }

    #[test]
    fn box_action_space_contains() {
        let space = ActionSpace::Box {
            low: vec![-1.0, -2.0],
            high: vec![1.0, 2.0],
            shape: vec![2],
        };
        assert!(space.contains(&Action::Continuous(vec![0.0, 0.0])));
        assert!(space.contains(&Action::Continuous(vec![-1.0, 2.0])));
        assert!(!space.contains(&Action::Continuous(vec![1.5, 0.0])));
        assert!(!space.contains(&Action::Continuous(vec![0.0])));
    }

    #[test]
    fn discrete_obs_space_contains() {
        let space = ObsSpace::Discrete(5);
        assert!(space.contains(&Observation(vec![3.0])));
        assert!(!space.contains(&Observation(vec![5.0])));
        assert!(!space.contains(&Observation(vec![1.0, 2.0])));
    }

    #[test]
    fn box_obs_space_contains() {
        let space = ObsSpace::Box {
            low: vec![-4.8; 4],
            high: vec![4.8; 4],
            shape: vec![4],
        };
        assert!(space.contains(&Observation(vec![0.0, 0.0, 0.0, 0.0])));
        assert!(!space.contains(&Observation(vec![0.0, 0.0, 5.0, 0.0])));
    }

    #[test]
    fn multi_discrete_obs_space_contains() {
        let space = ObsSpace::MultiDiscrete(vec![3, 5]);
        assert!(space.contains(&Observation(vec![2.0, 4.0])));
        assert!(!space.contains(&Observation(vec![3.0, 0.0])));
    }
}
