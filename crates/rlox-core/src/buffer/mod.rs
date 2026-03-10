pub mod columnar;
pub mod ringbuf;
pub mod varlen;

/// A single experience record to push into a buffer.
/// Uses f32 throughout for numpy compatibility.
#[derive(Debug, Clone)]
pub struct ExperienceRecord {
    pub obs: Vec<f32>,
    pub action: f32,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

#[cfg(test)]
pub(crate) fn sample_record(obs_dim: usize) -> ExperienceRecord {
    ExperienceRecord {
        obs: vec![1.0; obs_dim],
        action: 0.0,
        reward: 1.0,
        terminated: false,
        truncated: false,
    }
}
