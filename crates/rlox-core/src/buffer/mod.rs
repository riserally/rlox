pub mod columnar;
pub mod concurrent;
pub mod episode;
pub mod extra_columns;
#[cfg(feature = "gpu")]
pub mod flat;
pub mod her;
pub mod mixed;
pub mod mmap;
pub mod offline;
pub mod priority;
pub mod provenance;
pub mod ringbuf;
pub mod sequence;
pub mod varlen;

/// A single experience record to push into a buffer.
/// Uses f32 throughout for numpy compatibility.
#[derive(Debug, Clone)]
pub struct ExperienceRecord {
    pub obs: Vec<f32>,
    pub next_obs: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

#[cfg(test)]
pub(crate) fn sample_record(obs_dim: usize) -> ExperienceRecord {
    ExperienceRecord {
        obs: vec![1.0; obs_dim],
        next_obs: vec![2.0; obs_dim],
        action: vec![0.0],
        reward: 1.0,
        terminated: false,
        truncated: false,
    }
}

#[cfg(test)]
pub(crate) fn sample_record_multidim(obs_dim: usize, act_dim: usize) -> ExperienceRecord {
    ExperienceRecord {
        obs: vec![1.0; obs_dim],
        next_obs: vec![2.0; obs_dim],
        action: vec![0.0; act_dim],
        reward: 1.0,
        terminated: false,
        truncated: false,
    }
}

#[cfg(test)]
mod fix_verification_tests {
    use super::*;
    use crate::buffer::columnar::ExperienceTable;
    use crate::buffer::ringbuf::ReplayBuffer;

    #[test]
    fn experience_record_action_is_vec() {
        let record = ExperienceRecord {
            obs: vec![0.0f32; 17],
            next_obs: vec![0.0f32; 17],
            action: vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        assert_eq!(record.action.len(), 6);
        assert_eq!(record.obs.len(), 17);
    }

    #[test]
    fn experience_table_stores_multi_dim_action() {
        let obs_dim = 17;
        let act_dim = 6;
        let mut table = ExperienceTable::new(obs_dim, act_dim);
        let action = vec![0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        let record = ExperienceRecord {
            obs: vec![1.0; obs_dim],
            next_obs: vec![2.0; obs_dim],
            action: action.clone(),
            reward: 5.0,
            terminated: false,
            truncated: false,
        };
        table.push(record).unwrap();
        assert_eq!(table.actions_raw().len(), act_dim);
        assert_eq!(&table.actions_raw()[..act_dim], action.as_slice());
    }

    #[test]
    fn replay_buffer_multi_dim_action_roundtrip() {
        let obs_dim = 4;
        let act_dim = 3;
        let mut buf = ReplayBuffer::new(100, obs_dim, act_dim);
        let action = vec![0.5f32, -0.5, 1.0];
        let record = ExperienceRecord {
            obs: vec![0.1; obs_dim],
            next_obs: vec![0.2; obs_dim],
            action: action.clone(),
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        buf.push(record).unwrap();
        let batch = buf.sample(1, 42).unwrap();
        assert_eq!(batch.act_dim, act_dim);
        assert_eq!(batch.actions.len(), act_dim);
        assert_eq!(&batch.actions[..act_dim], action.as_slice());
    }

    #[test]
    fn experience_table_action_dim_mismatch_returns_error() {
        let mut table = ExperienceTable::new(4, 2);
        let record = ExperienceRecord {
            obs: vec![1.0; 4],
            next_obs: vec![2.0; 4],
            action: vec![0.1, 0.2, 0.3], // 3 dims, table expects 2
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        let result = table.push(record);
        assert!(result.is_err(), "action dim mismatch must return Err");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("act_dim"),
            "error must mention act_dim, got: {err_str}"
        );
    }

    #[test]
    fn experience_table_scalar_action_dim_one() {
        let mut table = ExperienceTable::new(4, 1);
        let record = ExperienceRecord {
            obs: vec![1.0; 4],
            next_obs: vec![2.0; 4],
            action: vec![0.0],
            reward: 1.0,
            terminated: false,
            truncated: false,
        };
        table.push(record).unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(table.actions_raw().len(), 1);
    }
}
