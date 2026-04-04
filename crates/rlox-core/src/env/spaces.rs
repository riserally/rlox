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
    /// Dict observation space: ordered `(key, dim)` pairs.
    Dict(Vec<(String, usize)>),
}

/// A concrete action value.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Discrete(u32),
    Continuous(Vec<f32>),
}

/// A concrete observation value.
///
/// `Flat` is the default: a single dense vector of f32 values.
/// `Dict` supports multi-modal observations (e.g. image + proprioception),
/// stored as ordered key-value pairs of named sub-vectors.
#[derive(Debug, Clone, PartialEq)]
pub enum Observation {
    /// A flat float vector (the common case).
    Flat(Vec<f32>),
    /// Named sub-observations in a fixed order.
    Dict(Vec<(String, Vec<f32>)>),
}

impl Observation {
    /// Convenience constructor matching the old tuple-struct API.
    ///
    /// `Observation::flat(vec)` is equivalent to the former `Observation(vec)`.
    pub fn flat(data: Vec<f32>) -> Self {
        Observation::Flat(data)
    }

    /// Try to view as a flat f32 slice.
    ///
    /// Returns `Some(&[f32])` for the `Flat` variant, `None` for `Dict`.
    /// Prefer this over [`as_slice`](Self::as_slice) when handling observations
    /// that may be either variant.
    pub fn try_as_slice(&self) -> Option<&[f32]> {
        match self {
            Observation::Flat(v) => Some(v),
            Observation::Dict(_) => None,
        }
    }

    /// View as a flat f32 slice.
    ///
    /// For `Flat`, returns the inner data directly.
    ///
    /// # Panics
    ///
    /// Panics if the observation is the `Dict` variant. Use
    /// [`try_as_slice`](Self::try_as_slice) for a fallible alternative, or
    /// [`flatten`](Self::flatten) if you need a contiguous copy.
    pub fn as_slice(&self) -> &[f32] {
        match self {
            Observation::Flat(v) => v,
            Observation::Dict(_) => {
                panic!("Observation::as_slice() called on Dict variant; use try_as_slice() or flatten() instead")
            }
        }
    }

    /// Consume and return the inner Vec for the `Flat` variant.
    ///
    /// For `Dict`, returns a flattened (concatenated) copy.
    pub fn into_inner(self) -> Vec<f32> {
        match self {
            Observation::Flat(v) => v,
            Observation::Dict(pairs) => {
                let total = pairs.iter().map(|(_, v)| v.len()).sum();
                let mut out = Vec::with_capacity(total);
                for (_, v) in pairs {
                    out.extend(v);
                }
                out
            }
        }
    }

    /// Total number of f32 elements across all keys.
    pub fn total_dim(&self) -> usize {
        match self {
            Observation::Flat(v) => v.len(),
            Observation::Dict(pairs) => pairs.iter().map(|(_, v)| v.len()).sum(),
        }
    }

    /// Flatten to a single `Vec<f32>` (concatenate all values in key order).
    ///
    /// For `Flat`, returns a clone of the inner data.
    /// For `Dict`, concatenates all sub-vectors.
    pub fn flatten(&self) -> Vec<f32> {
        match self {
            Observation::Flat(v) => v.clone(),
            Observation::Dict(pairs) => {
                let total = pairs.iter().map(|(_, v)| v.len()).sum();
                let mut out = Vec::with_capacity(total);
                for (_, v) in pairs {
                    out.extend_from_slice(v);
                }
                out
            }
        }
    }

    /// Get a named sub-observation (for the `Dict` variant).
    ///
    /// Returns `None` for the `Flat` variant or if the key is not found.
    pub fn get(&self, key: &str) -> Option<&[f32]> {
        match self {
            Observation::Flat(_) => None,
            Observation::Dict(pairs) => pairs
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_slice()),
        }
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
            ObsSpace::Discrete(n) => {
                let s = match obs {
                    Observation::Flat(v) => v.as_slice(),
                    _ => return false,
                };
                s.len() == 1 && (s[0] as usize) < *n
            }
            ObsSpace::Box { low, high, .. } => {
                let s = match obs {
                    Observation::Flat(v) => v.as_slice(),
                    _ => return false,
                };
                s.len() == low.len()
                    && s.iter()
                        .zip(low.iter().zip(high.iter()))
                        .all(|(v, (lo, hi))| *v >= *lo && *v <= *hi)
            }
            ObsSpace::MultiDiscrete(nvec) => {
                let s = match obs {
                    Observation::Flat(v) => v.as_slice(),
                    _ => return false,
                };
                s.len() == nvec.len()
                    && s.iter()
                        .zip(nvec.iter())
                        .all(|(v, n)| *v >= 0.0 && (*v as usize) < *n)
            }
            ObsSpace::Dict(entries) => {
                let pairs = match obs {
                    Observation::Dict(p) => p,
                    _ => return false,
                };
                if pairs.len() != entries.len() {
                    return false;
                }
                pairs
                    .iter()
                    .zip(entries.iter())
                    .all(|((ok, ov), (ek, ed))| ok == ek && ov.len() == *ed)
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
        assert!(space.contains(&Observation::Flat(vec![3.0])));
        assert!(!space.contains(&Observation::Flat(vec![5.0])));
        assert!(!space.contains(&Observation::Flat(vec![1.0, 2.0])));
    }

    #[test]
    fn box_obs_space_contains() {
        let space = ObsSpace::Box {
            low: vec![-4.8; 4],
            high: vec![4.8; 4],
            shape: vec![4],
        };
        assert!(space.contains(&Observation::Flat(vec![0.0, 0.0, 0.0, 0.0])));
        assert!(!space.contains(&Observation::Flat(vec![0.0, 0.0, 5.0, 0.0])));
    }

    #[test]
    fn multi_discrete_obs_space_contains() {
        let space = ObsSpace::MultiDiscrete(vec![3, 5]);
        assert!(space.contains(&Observation::Flat(vec![2.0, 4.0])));
        assert!(!space.contains(&Observation::Flat(vec![3.0, 0.0])));
    }

    // --- Dict Observation tests ---

    #[test]
    fn test_dict_observation_total_dim() {
        let obs = Observation::Dict(vec![
            ("image".into(), vec![0.0; 784]),
            ("proprio".into(), vec![0.0; 7]),
        ]);
        assert_eq!(obs.total_dim(), 791);
    }

    #[test]
    fn test_dict_observation_flatten() {
        let obs = Observation::Dict(vec![("a".into(), vec![1.0, 2.0]), ("b".into(), vec![3.0])]);
        assert_eq!(obs.flatten(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dict_observation_get_key() {
        let obs = Observation::Dict(vec![
            ("image".into(), vec![1.0, 2.0, 3.0]),
            ("proprio".into(), vec![4.0, 5.0]),
        ]);
        assert_eq!(obs.get("image"), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(obs.get("proprio"), Some([4.0, 5.0].as_slice()));
        assert_eq!(obs.get("missing"), None);
    }

    #[test]
    fn test_dict_obs_space_contains() {
        let space = ObsSpace::Dict(vec![("image".into(), 784), ("proprio".into(), 7)]);

        let valid = Observation::Dict(vec![
            ("image".into(), vec![0.0; 784]),
            ("proprio".into(), vec![0.0; 7]),
        ]);
        assert!(space.contains(&valid));

        // Wrong dim
        let bad_dim = Observation::Dict(vec![
            ("image".into(), vec![0.0; 784]),
            ("proprio".into(), vec![0.0; 8]),
        ]);
        assert!(!space.contains(&bad_dim));

        // Wrong key name
        let bad_key = Observation::Dict(vec![
            ("image".into(), vec![0.0; 784]),
            ("wrong".into(), vec![0.0; 7]),
        ]);
        assert!(!space.contains(&bad_key));

        // Wrong number of entries
        let bad_count = Observation::Dict(vec![("image".into(), vec![0.0; 784])]);
        assert!(!space.contains(&bad_count));

        // Flat obs should not match Dict space
        let flat = Observation::Flat(vec![0.0; 791]);
        assert!(!space.contains(&flat));
    }

    #[test]
    fn test_try_as_slice_flat_returns_some() {
        let obs = Observation::Flat(vec![1.0, 2.0, 3.0]);
        assert_eq!(obs.try_as_slice(), Some([1.0, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn test_try_as_slice_dict_returns_none() {
        let obs = Observation::Dict(vec![("a".into(), vec![1.0])]);
        assert_eq!(obs.try_as_slice(), None);
    }

    #[test]
    #[should_panic(expected = "Dict variant")]
    fn test_as_slice_dict_panics() {
        let obs = Observation::Dict(vec![("a".into(), vec![1.0])]);
        let _ = obs.as_slice();
    }

    #[test]
    fn test_flat_observation_backward_compat() {
        // Ensure Flat variant works exactly like the old tuple struct
        let obs = Observation::Flat(vec![1.0, 2.0, 3.0]);
        assert_eq!(obs.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(obs.total_dim(), 3);
        assert_eq!(obs.flatten(), vec![1.0, 2.0, 3.0]);
        assert_eq!(obs.get("anything"), None);

        let inner = obs.into_inner();
        assert_eq!(inner, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dict_observation_into_inner_flattens() {
        let obs = Observation::Dict(vec![
            ("a".into(), vec![1.0, 2.0]),
            ("b".into(), vec![3.0, 4.0, 5.0]),
        ]);
        assert_eq!(obs.into_inner(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_dict_obs_space_does_not_match_flat_obs() {
        let space = ObsSpace::Discrete(5);
        let dict_obs = Observation::Dict(vec![("x".into(), vec![3.0])]);
        assert!(!space.contains(&dict_obs));
    }
}
