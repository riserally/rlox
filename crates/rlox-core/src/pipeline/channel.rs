use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::error::RloxError;

/// A batch of rollout data ready for the learner.
///
/// All vectors are flat (row-major). For example, `observations` has length
/// `n_steps * n_envs * obs_dim`, laid out as `[step0_env0, step0_env1, ...,
/// step1_env0, ...]`.
#[derive(Debug, Clone)]
pub struct RolloutBatch {
    /// Flat observations: `[n_steps * n_envs * obs_dim]`.
    pub observations: Vec<f32>,
    /// Flat actions: `[n_steps * n_envs * act_dim]`.
    pub actions: Vec<f32>,
    /// Rewards: `[n_steps * n_envs]`.
    pub rewards: Vec<f64>,
    /// Done flags (0.0 or 1.0): `[n_steps * n_envs]`.
    pub dones: Vec<f64>,
    /// GAE advantages: `[n_steps * n_envs]`.
    pub advantages: Vec<f64>,
    /// Discounted returns: `[n_steps * n_envs]`.
    pub returns: Vec<f64>,
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Action dimensionality.
    pub act_dim: usize,
    /// Number of time steps in this batch.
    pub n_steps: usize,
    /// Number of environments that contributed to this batch.
    pub n_envs: usize,
}

/// Bounded experience pipeline for decoupled collection and training.
///
/// Uses a crossbeam bounded channel internally, providing backpressure when
/// the learner falls behind the collector. The `capacity` controls how many
/// `RolloutBatch`es can be buffered before the sender blocks.
pub struct Pipeline {
    tx: Sender<RolloutBatch>,
    rx: Receiver<RolloutBatch>,
}

impl Pipeline {
    /// Create a new pipeline with the given buffer capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Pipeline capacity must be at least 1");
        let (tx, rx) = bounded(capacity);
        Self { tx, rx }
    }

    /// Send a batch into the pipeline (blocks if full).
    pub fn send(&self, batch: RolloutBatch) -> Result<(), RloxError> {
        self.tx
            .send(batch)
            .map_err(|_| RloxError::BufferError("pipeline channel disconnected".into()))
    }

    /// Try to send a batch without blocking. Returns `Ok(())` on success,
    /// `Err` with the batch if the channel is full or disconnected.
    pub fn try_send(&self, batch: RolloutBatch) -> Result<(), RloxError> {
        self.tx.try_send(batch).map_err(|e| match e {
            TrySendError::Full(_) => RloxError::BufferError("pipeline channel full".into()),
            TrySendError::Disconnected(_) => {
                RloxError::BufferError("pipeline channel disconnected".into())
            }
        })
    }

    /// Receive a batch (blocks until one is available).
    pub fn recv(&self) -> Result<RolloutBatch, RloxError> {
        self.rx
            .recv()
            .map_err(|_| RloxError::BufferError("pipeline channel disconnected".into()))
    }

    /// Try to receive a batch without blocking.
    pub fn try_recv(&self) -> Option<RolloutBatch> {
        self.rx.try_recv().ok()
    }

    /// Number of batches currently buffered in the channel.
    pub fn len(&self) -> usize {
        self.rx.len()
    }

    /// Whether the channel is currently empty.
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }

    /// Get a clone of the sender (for use in collector threads).
    pub fn sender(&self) -> Sender<RolloutBatch> {
        self.tx.clone()
    }

    /// Get a clone of the receiver (for use in learner threads).
    pub fn receiver(&self) -> Receiver<RolloutBatch> {
        self.rx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_batch(tag: usize) -> RolloutBatch {
        let n_steps = 4;
        let n_envs = 2;
        let obs_dim = 3;
        let act_dim = 1;
        let total = n_steps * n_envs;
        RolloutBatch {
            observations: vec![tag as f32; total * obs_dim],
            actions: vec![tag as f32; total * act_dim],
            rewards: vec![tag as f64; total],
            dones: vec![0.0; total],
            advantages: vec![tag as f64 * 0.1; total],
            returns: vec![tag as f64 * 0.5; total],
            obs_dim,
            act_dim,
            n_steps,
            n_envs,
        }
    }

    #[test]
    fn test_pipeline_new_is_empty() {
        let pipe = Pipeline::new(4);
        assert!(pipe.is_empty());
        assert_eq!(pipe.len(), 0);
    }

    #[test]
    fn test_pipeline_send_recv_roundtrip() {
        let pipe = Pipeline::new(4);
        let batch = sample_batch(42);
        pipe.send(batch).unwrap();
        assert_eq!(pipe.len(), 1);

        let received = pipe.recv().unwrap();
        assert_eq!(received.observations[0], 42.0);
        assert_eq!(received.rewards[0], 42.0);
        assert_eq!(received.obs_dim, 3);
        assert_eq!(received.act_dim, 1);
        assert_eq!(received.n_steps, 4);
        assert_eq!(received.n_envs, 2);
    }

    #[test]
    fn test_pipeline_try_recv_empty_returns_none() {
        let pipe = Pipeline::new(4);
        assert!(pipe.try_recv().is_none());
    }

    #[test]
    fn test_pipeline_backpressure_blocks() {
        let pipe = Pipeline::new(1);
        // First send succeeds
        pipe.send(sample_batch(1)).unwrap();
        assert_eq!(pipe.len(), 1);

        // Second send via try_send should fail (channel full)
        let result = pipe.try_send(sample_batch(2));
        assert!(result.is_err());

        // Drain and verify the first batch is still intact
        let b = pipe.recv().unwrap();
        assert_eq!(b.observations[0], 1.0);
    }

    #[test]
    fn test_pipeline_rollout_batch_data_integrity() {
        let pipe = Pipeline::new(4);
        let n_steps = 4;
        let n_envs = 2;
        let obs_dim = 3;
        let act_dim = 1;
        let total = n_steps * n_envs;

        let batch = RolloutBatch {
            observations: (0..total * obs_dim).map(|i| i as f32).collect(),
            actions: (0..total * act_dim).map(|i| i as f32 * 0.1).collect(),
            rewards: (0..total).map(|i| i as f64).collect(),
            dones: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            advantages: (0..total).map(|i| i as f64 * 0.01).collect(),
            returns: (0..total).map(|i| i as f64 * 0.5).collect(),
            obs_dim,
            act_dim,
            n_steps,
            n_envs,
        };

        pipe.send(batch).unwrap();
        let out = pipe.recv().unwrap();

        assert_eq!(out.observations.len(), total * obs_dim);
        assert_eq!(out.actions.len(), total * act_dim);
        assert_eq!(out.rewards.len(), total);
        assert_eq!(out.dones.len(), total);
        assert_eq!(out.advantages.len(), total);
        assert_eq!(out.returns.len(), total);
        // Spot-check values
        assert_eq!(out.observations[5], 5.0);
        assert_eq!(out.dones[2], 1.0);
        assert_eq!(out.dones[7], 1.0);
        assert!((out.returns[3] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_multiple_batches_fifo_order() {
        let pipe = Pipeline::new(8);

        for i in 0..5 {
            pipe.send(sample_batch(i)).unwrap();
        }
        assert_eq!(pipe.len(), 5);

        for i in 0..5 {
            let b = pipe.recv().unwrap();
            assert_eq!(b.observations[0], i as f32, "batch {i} out of order");
        }
        assert!(pipe.is_empty());
    }

    #[test]
    #[should_panic(expected = "Pipeline capacity must be at least 1")]
    fn test_pipeline_zero_capacity_panics() {
        Pipeline::new(0);
    }

    #[test]
    fn test_pipeline_cross_thread_send_recv() {
        let pipe = Pipeline::new(4);
        let tx = pipe.sender();

        let handle = std::thread::spawn(move || {
            tx.send(sample_batch(99)).unwrap();
        });

        handle.join().unwrap();
        let b = pipe.recv().unwrap();
        assert_eq!(b.observations[0], 99.0);
    }
}
