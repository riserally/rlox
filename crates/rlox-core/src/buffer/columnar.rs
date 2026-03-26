use crate::error::RloxError;

use super::ExperienceRecord;

/// Append-only columnar table for RL transitions.
///
/// All data is stored in flat `Vec<f32>` arrays for efficient export
/// to numpy. The table only appends, never reallocating existing data
/// in-place.
pub struct ExperienceTable {
    obs_dim: usize,
    act_dim: usize,
    observations: Vec<f32>,
    next_observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    count: usize,
}

impl ExperienceTable {
    /// Create a new table with the given observation and action dimensions.
    pub fn new(obs_dim: usize, act_dim: usize) -> Self {
        Self {
            obs_dim,
            act_dim,
            observations: Vec::new(),
            next_observations: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            terminated: Vec::new(),
            truncated: Vec::new(),
            count: 0,
        }
    }

    /// Observation dimensionality.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimensionality.
    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    /// Number of transitions stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Append a transition from borrowed slices, avoiding intermediate allocation.
    pub fn push_slices(
        &mut self,
        obs: &[f32],
        next_obs: &[f32],
        action: &[f32],
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), RloxError> {
        if obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("obs.len()={}", obs.len()),
            });
        }
        if next_obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("next_obs.len()={}", next_obs.len()),
            });
        }
        if action.len() != self.act_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("act_dim={}", self.act_dim),
                got: format!("action.len()={}", action.len()),
            });
        }
        self.observations.extend_from_slice(obs);
        self.next_observations.extend_from_slice(next_obs);
        self.actions.extend_from_slice(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.truncated.push(truncated);
        self.count += 1;
        Ok(())
    }

    /// Append a single transition. Returns error on dimension mismatch.
    pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError> {
        self.push_slices(
            &record.obs, &record.next_obs, &record.action,
            record.reward, record.terminated, record.truncated,
        )
    }

    /// Raw slice of all observation data. Shape: [count * obs_dim].
    pub fn observations_raw(&self) -> &[f32] {
        &self.observations
    }

    /// Raw slice of all action data. Shape: [count * act_dim].
    pub fn actions_raw(&self) -> &[f32] {
        &self.actions
    }

    /// Raw slice of all rewards.
    pub fn rewards_raw(&self) -> &[f32] {
        &self.rewards
    }

    /// Slice of terminated flags.
    pub fn terminated(&self) -> &[bool] {
        &self.terminated
    }

    /// Slice of truncated flags.
    pub fn truncated(&self) -> &[bool] {
        &self.truncated
    }

    /// Drop all stored data.
    pub fn clear(&mut self) {
        self.observations.clear();
        self.next_observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::sample_record;

    #[test]
    fn empty_table_has_zero_len() {
        let table = ExperienceTable::new(4, 1);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn push_single_transition_increments_len() {
        let mut table = ExperienceTable::new(4, 1);
        table.push(sample_record(4)).unwrap();
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn push_many_transitions() {
        let mut table = ExperienceTable::new(4, 1);
        for _ in 0..1000 {
            table.push(sample_record(4)).unwrap();
        }
        assert_eq!(table.len(), 1000);
    }

    #[test]
    fn observations_column_correct_length() {
        let mut table = ExperienceTable::new(4, 1);
        for _ in 0..10 {
            table.push(sample_record(4)).unwrap();
        }
        assert_eq!(table.observations_raw().len(), 40);
    }

    #[test]
    fn rewards_column_correct_values() {
        let mut table = ExperienceTable::new(4, 1);
        let mut r = sample_record(4);
        r.reward = 42.0;
        table.push(r).unwrap();
        assert_eq!(table.rewards_raw()[0], 42.0);
    }

    #[test]
    fn obs_dim_mismatch_returns_error() {
        let mut table = ExperienceTable::new(4, 1);
        let bad = sample_record(8);
        assert!(table.push(bad).is_err());
    }

    #[test]
    fn clear_empties_all_columns() {
        let mut table = ExperienceTable::new(4, 1);
        for _ in 0..100 {
            table.push(sample_record(4)).unwrap();
        }
        table.clear();
        assert_eq!(table.len(), 0);
        assert!(table.observations_raw().is_empty());
    }

    #[test]
    fn obs_dim_getter() {
        let table = ExperienceTable::new(4, 1);
        assert_eq!(table.obs_dim(), 4);
    }
}
