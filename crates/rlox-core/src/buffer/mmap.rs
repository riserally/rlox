use std::fs::{File, OpenOptions};
use std::path::PathBuf;

use memmap2::{Mmap, MmapMut};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

use super::ringbuf::{ReplayBuffer, SampledBatch};
use super::ExperienceRecord;

/// Binary header: obs_dim(u64) + act_dim(u64) + record_count(u64) = 24 bytes.
const HEADER_SIZE: usize = 24;

/// When replay buffer exceeds RAM capacity, transparently spill to disk.
/// Uses mmap for lazy loading -- only pages actually accessed are read.
///
/// Architecture:
/// - `hot`: in-memory `ReplayBuffer` for recent data (fast sampling)
/// - `cold`: memory-mapped file for overflow data (ring-buffer with wrapping)
/// - Sampling prefers hot data but can reach into cold
pub struct MmapReplayBuffer {
    hot: ReplayBuffer,
    cold_path: PathBuf,
    cold_file: Option<File>,
    /// Read-only mmap for sampling. Created lazily on first sample after cold
    /// file is initialized.
    cold_mmap: Option<Mmap>,
    /// Mutable mmap for writes. Pre-allocated to full cold capacity so no
    /// remap is ever needed during push.
    cold_mmap_mut: Option<MmapMut>,
    /// Number of valid records in cold storage (saturates at cold_capacity).
    cold_count: usize,
    /// Next write position in cold ring-buffer (wraps at cold_capacity).
    cold_write_pos: usize,
    obs_dim: usize,
    act_dim: usize,
    hot_capacity: usize,
    total_capacity: usize,
    /// Whether the read mmap needs to be refreshed before the next sample.
    mmap_stale: bool,
}

impl std::fmt::Debug for MmapReplayBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapReplayBuffer")
            .field("obs_dim", &self.obs_dim)
            .field("act_dim", &self.act_dim)
            .field("hot_capacity", &self.hot_capacity)
            .field("total_capacity", &self.total_capacity)
            .field("cold_count", &self.cold_count)
            .field("cold_write_pos", &self.cold_write_pos)
            .field("cold_path", &self.cold_path)
            .finish_non_exhaustive()
    }
}

impl MmapReplayBuffer {
    /// Create a new mmap-backed replay buffer.
    ///
    /// `hot_capacity` records are kept in memory. When exceeded, the oldest
    /// hot records spill to the cold file at `cold_path`. The total number of
    /// records stored (hot + cold) is capped at `total_capacity`.
    pub fn new(
        hot_capacity: usize,
        total_capacity: usize,
        obs_dim: usize,
        act_dim: usize,
        cold_path: PathBuf,
    ) -> Result<Self, RloxError> {
        if hot_capacity == 0 {
            return Err(RloxError::BufferError(
                "hot_capacity must be > 0".to_string(),
            ));
        }
        if total_capacity < hot_capacity {
            return Err(RloxError::BufferError(
                "total_capacity must be >= hot_capacity".to_string(),
            ));
        }
        if obs_dim == 0 {
            return Err(RloxError::BufferError(
                "obs_dim must be > 0".to_string(),
            ));
        }
        if act_dim == 0 {
            return Err(RloxError::BufferError(
                "act_dim must be > 0".to_string(),
            ));
        }

        Ok(Self {
            hot: ReplayBuffer::new(hot_capacity, obs_dim, act_dim),
            cold_path,
            cold_file: None,
            cold_mmap: None,
            cold_mmap_mut: None,
            cold_count: 0,
            cold_write_pos: 0,
            obs_dim,
            act_dim,
            hot_capacity,
            total_capacity,
            mmap_stale: false,
        })
    }

    /// Byte size of a single serialized record (no padding).
    fn record_byte_size(&self) -> usize {
        // obs(f32 * obs_dim) + next_obs(f32 * obs_dim) + action(f32 * act_dim) + reward(f32) + terminated(u8) + truncated(u8)
        (self.obs_dim * 2 + self.act_dim + 1) * 4 + 2
    }

    /// Total number of records stored (hot + cold).
    pub fn len(&self) -> usize {
        self.hot.len() + self.cold_count
    }

    /// Whether the buffer contains no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a record. If the hot buffer is full, the oldest hot record is
    /// spilled to cold storage before the new record is inserted.
    pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError> {
        if record.obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("obs.len()={}", record.obs.len()),
            });
        }
        if record.next_obs.len() != self.obs_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("obs_dim={}", self.obs_dim),
                got: format!("next_obs.len()={}", record.next_obs.len()),
            });
        }
        if record.action.len() != self.act_dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("act_dim={}", self.act_dim),
                got: format!("action.len()={}", record.action.len()),
            });
        }

        // If hot is full, spill the oldest hot record to cold before pushing.
        if self.hot.len() == self.hot_capacity {
            let oldest = self.read_oldest_hot_record();
            self.write_to_cold(&oldest)?;
        }

        self.hot.push(record)?;
        Ok(())
    }

    /// Push multiple transitions at once from flat arrays.
    ///
    /// `obs_batch` shape: `[n * obs_dim]`, `next_obs_batch`: same,
    /// `actions_batch`: `[n * act_dim]`, others: `[n]`.
    ///
    /// # `terminated` / `truncated` convention
    ///
    /// These take `&[f32]` (not `bool`) for compatibility with numpy arrays
    /// from the Python side. Non-zero values are treated as `true`.
    /// This differs from [`push`](Self::push) which accepts native `bool`.
    pub fn push_batch(
        &mut self,
        obs_batch: &[f32],
        next_obs_batch: &[f32],
        actions_batch: &[f32],
        rewards: &[f32],
        terminated: &[f32],
        truncated: &[f32],
    ) -> Result<(), RloxError> {
        let n = rewards.len();
        if obs_batch.len() != n * self.obs_dim
            || next_obs_batch.len() != n * self.obs_dim
            || actions_batch.len() != n * self.act_dim
            || terminated.len() != n
            || truncated.len() != n
        {
            return Err(RloxError::ShapeMismatch {
                expected: format!("n={n}, obs_dim={}, act_dim={}", self.obs_dim, self.act_dim),
                got: format!(
                    "obs={}, next_obs={}, act={}, rew={}, term={}, trunc={}",
                    obs_batch.len(),
                    next_obs_batch.len(),
                    actions_batch.len(),
                    rewards.len(),
                    terminated.len(),
                    truncated.len()
                ),
            });
        }
        for i in 0..n {
            let obs = obs_batch[i * self.obs_dim..(i + 1) * self.obs_dim].to_vec();
            let next_obs = next_obs_batch[i * self.obs_dim..(i + 1) * self.obs_dim].to_vec();
            let action = actions_batch[i * self.act_dim..(i + 1) * self.act_dim].to_vec();
            let record = ExperienceRecord {
                obs,
                next_obs,
                action,
                reward: rewards[i],
                terminated: terminated[i] != 0.0,
                truncated: truncated[i] != 0.0,
            };
            self.push(record)?;
        }
        Ok(())
    }

    /// Sample `batch_size` records uniformly from hot + cold storage.
    /// Uses ChaCha8Rng seeded with `seed` for deterministic reproducibility.
    pub fn sample(&mut self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
        let total = self.len();
        if batch_size > total {
            return Err(RloxError::BufferError(format!(
                "batch_size {} > buffer len {}",
                batch_size, total,
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);

        let cold_len = self.cold_count;

        // Refresh the read mmap if any writes occurred since last sample.
        if cold_len > 0 {
            self.refresh_read_mmap()?;
        }

        for _ in 0..batch_size {
            let idx = rng.random_range(0..total);
            if idx < cold_len {
                // Read from cold mmap.
                self.read_cold_record_into(idx, &mut batch);
            } else {
                // Read from hot. Hot indices are [0..hot_len).
                let hot_idx = idx - cold_len;
                self.read_hot_record_into(hot_idx, &mut batch);
            }
        }
        batch.batch_size = batch_size;
        Ok(batch)
    }

    /// Unmap the cold file and delete it from disk.
    pub fn close(&mut self) -> Result<(), RloxError> {
        // Drop mmaps first so the file can be removed.
        self.cold_mmap.take();
        self.cold_mmap_mut.take();
        self.cold_file.take();
        if self.cold_path.exists() {
            std::fs::remove_file(&self.cold_path)?;
        }
        self.cold_count = 0;
        self.cold_write_pos = 0;
        Ok(())
    }

    // ---- private helpers ------------------------------------------------

    /// Read the record at position `hot_idx` from the hot buffer into `batch`.
    fn read_hot_record_into(&self, hot_idx: usize, batch: &mut SampledBatch) {
        let (obs, next_obs, act, reward, terminated, truncated) = self.hot.get(hot_idx);
        batch.observations.extend_from_slice(obs);
        batch.next_observations.extend_from_slice(next_obs);
        batch.actions.extend_from_slice(act);
        batch.rewards.push(reward);
        batch.terminated.push(terminated);
        batch.truncated.push(truncated);
    }

    /// Read the oldest record from the hot ring buffer (the one at write_pos,
    /// which is about to be overwritten).
    fn read_oldest_hot_record(&self) -> ExperienceRecord {
        let oldest_idx = self.hot.write_pos();
        let (obs, next_obs, act, reward, terminated, truncated) = self.hot.get(oldest_idx);
        ExperienceRecord {
            obs: obs.to_vec(),
            next_obs: next_obs.to_vec(),
            action: act.to_vec(),
            reward,
            terminated,
            truncated,
        }
    }

    /// Write a record to the cold ring-buffer at `cold_write_pos`, then advance.
    ///
    /// Uses a pre-allocated `MmapMut` so no remap is needed per push.
    /// The file is pre-allocated to full cold capacity via `ftruncate`.
    fn write_to_cold(&mut self, record: &ExperienceRecord) -> Result<(), RloxError> {
        self.ensure_cold_file()?;

        let cold_capacity = self.total_capacity - self.hot_capacity;
        let record_size = self.record_byte_size();

        // Serialize record into the mmap directly.
        let file_offset = HEADER_SIZE + self.cold_write_pos * record_size;
        let mmap = self
            .cold_mmap_mut
            .as_mut()
            .expect("cold_mmap_mut must be set after ensure_cold_file");

        let dst = &mut mmap[file_offset..file_offset + record_size];
        let mut pos = 0;
        for &v in &record.obs {
            dst[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        }
        for &v in &record.next_obs {
            dst[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        }
        for &v in &record.action {
            dst[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        }
        dst[pos..pos + 4].copy_from_slice(&record.reward.to_le_bytes());
        pos += 4;
        dst[pos] = record.terminated as u8;
        dst[pos + 1] = record.truncated as u8;

        // Advance ring-buffer position.
        self.cold_write_pos = (self.cold_write_pos + 1) % cold_capacity;
        if self.cold_count < cold_capacity {
            self.cold_count += 1;
        }

        // Update header in the mmap.
        self.write_cold_header_mmap();

        // Mark read mmap as stale so it refreshes before next sample.
        self.mmap_stale = true;

        Ok(())
    }

    /// Ensure the cold file is open, pre-allocated to full capacity, and mapped.
    ///
    /// The file is sized to `HEADER_SIZE + cold_capacity * record_byte_size`
    /// via `ftruncate` at creation time. A single `MmapMut` covers the entire
    /// file, eliminating per-push remap overhead.
    fn ensure_cold_file(&mut self) -> Result<(), RloxError> {
        if self.cold_file.is_some() {
            return Ok(());
        }

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.cold_path)?;

        let cold_capacity = self.total_capacity - self.hot_capacity;
        let total_file_size = HEADER_SIZE + cold_capacity * self.record_byte_size();
        file.set_len(total_file_size as u64)?;

        // SAFETY: The file is exclusively owned by this buffer. No other
        // process or thread accesses it. We write through `cold_mmap_mut`
        // and read through a separate `cold_mmap` (refreshed lazily).
        let mmap_mut = unsafe { MmapMut::map_mut(&file)? };

        self.cold_file = Some(file);
        self.cold_mmap_mut = Some(mmap_mut);

        // Write initial header (cold_count = 0).
        self.write_cold_header_mmap();

        Ok(())
    }

    /// Write the 24-byte header directly into the mutable mmap.
    fn write_cold_header_mmap(&mut self) {
        let mmap = self
            .cold_mmap_mut
            .as_mut()
            .expect("cold_mmap_mut must be set");
        mmap[0..8].copy_from_slice(&(self.obs_dim as u64).to_le_bytes());
        mmap[8..16].copy_from_slice(&(self.act_dim as u64).to_le_bytes());
        mmap[16..24].copy_from_slice(&(self.cold_count as u64).to_le_bytes());
    }

    /// Refresh the read-only mmap from the file.
    ///
    /// Called lazily before sampling when writes have occurred since the
    /// last refresh. Since the file is pre-allocated, this does not change
    /// the file size — it just picks up the bytes written via `cold_mmap_mut`.
    fn refresh_read_mmap(&mut self) -> Result<(), RloxError> {
        if !self.mmap_stale && self.cold_mmap.is_some() {
            return Ok(());
        }
        // Flush the mutable mmap so reads see the latest data.
        if let Some(ref mmap_mut) = self.cold_mmap_mut {
            mmap_mut.flush()?;
        }
        // Drop old read mmap.
        self.cold_mmap.take();
        let file = self.cold_file.as_ref().expect("cold_file must be open");
        // SAFETY: The file is exclusively owned by this buffer. The mutable
        // mmap has been flushed above, so all writes are visible.
        let mmap = unsafe { Mmap::map(file)? };
        self.cold_mmap = Some(mmap);
        self.mmap_stale = false;
        Ok(())
    }

    /// Read record at `idx` from the cold mmap into `batch`.
    fn read_cold_record_into(&self, idx: usize, batch: &mut SampledBatch) {
        let mmap = self.cold_mmap.as_ref().expect("cold_mmap must exist");
        let record_size = self.record_byte_size();
        let offset = HEADER_SIZE + idx * record_size;
        let data = &mmap[offset..offset + record_size];

        let obs_bytes = self.obs_dim * 4;
        let act_bytes = self.act_dim * 4;

        // Parse obs.
        for i in 0..self.obs_dim {
            let start = i * 4;
            let val = f32::from_le_bytes([
                data[start],
                data[start + 1],
                data[start + 2],
                data[start + 3],
            ]);
            batch.observations.push(val);
        }

        // Parse next_obs.
        let next_obs_base = obs_bytes;
        for i in 0..self.obs_dim {
            let start = next_obs_base + i * 4;
            let val = f32::from_le_bytes([
                data[start],
                data[start + 1],
                data[start + 2],
                data[start + 3],
            ]);
            batch.next_observations.push(val);
        }

        // Parse actions.
        let act_base = obs_bytes * 2;
        for i in 0..self.act_dim {
            let start = act_base + i * 4;
            let val = f32::from_le_bytes([
                data[start],
                data[start + 1],
                data[start + 2],
                data[start + 3],
            ]);
            batch.actions.push(val);
        }

        // Parse reward.
        let reward_base = obs_bytes * 2 + act_bytes;
        let reward = f32::from_le_bytes([
            data[reward_base],
            data[reward_base + 1],
            data[reward_base + 2],
            data[reward_base + 3],
        ]);
        batch.rewards.push(reward);

        // Parse terminated, truncated.
        batch.terminated.push(data[reward_base + 4] != 0);
        batch.truncated.push(data[reward_base + 5] != 0);
    }
}

impl Drop for MmapReplayBuffer {
    fn drop(&mut self) {
        // Best-effort cleanup.
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{sample_record, sample_record_multidim};

    fn temp_cold_path() -> (tempfile::NamedTempFile, PathBuf) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Close the named temp file so MmapReplayBuffer can create/truncate it.
        // We keep the PathBuf; the file will be cleaned up by the buffer or test.
        (tmp, path)
    }

    #[test]
    fn test_mmap_buffer_new_is_empty() {
        let (_tmp, path) = temp_cold_path();
        let buf = MmapReplayBuffer::new(100, 1000, 4, 1, path).unwrap();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_mmap_buffer_push_within_hot_capacity() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(100, 1000, 4, 1, path).unwrap();
        for _ in 0..50 {
            buf.push(sample_record(4)).unwrap();
        }
        assert_eq!(buf.len(), 50);
        // No cold records should exist.
        assert_eq!(buf.cold_count, 0);
    }

    #[test]
    fn test_mmap_buffer_push_exceeds_hot_spills_to_cold() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(10, 100, 4, 1, path).unwrap();

        // Push 15 records: 10 fit in hot, then pushes 11-15 spill records to cold.
        for i in 0..15 {
            let mut rec = sample_record(4);
            rec.reward = i as f32;
            buf.push(rec).unwrap();
        }

        // Hot should be full (10), cold should have 5 spilled records.
        assert_eq!(buf.hot.len(), 10);
        assert_eq!(buf.cold_count, 5);
        assert_eq!(buf.len(), 15);
    }

    #[test]
    fn test_mmap_buffer_sample_from_hot_only() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(100, 1000, 4, 1, path).unwrap();
        for i in 0..50 {
            let mut rec = sample_record(4);
            rec.reward = i as f32;
            buf.push(rec).unwrap();
        }
        let batch = buf.sample(10, 42).unwrap();
        assert_eq!(batch.batch_size, 10);
        assert_eq!(batch.observations.len(), 10 * 4);
        assert_eq!(batch.rewards.len(), 10);
    }

    #[test]
    fn test_mmap_buffer_sample_from_hot_and_cold() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(10, 100, 4, 1, path).unwrap();

        // Push 20 records with distinct rewards.
        for i in 0..20 {
            let mut rec = sample_record(4);
            rec.reward = i as f32;
            buf.push(rec).unwrap();
        }

        assert_eq!(buf.cold_count, 10);
        assert_eq!(buf.hot.len(), 10);

        // Sample a large batch -- should draw from both hot and cold.
        let batch = buf.sample(20, 99).unwrap();
        assert_eq!(batch.batch_size, 20);
        assert_eq!(batch.rewards.len(), 20);

        // Verify we see some rewards from the cold range [0..10)
        // and some from the hot range [10..20).
        let has_cold = batch.rewards.iter().any(|&r| r < 10.0);
        let has_hot = batch.rewards.iter().any(|&r| r >= 10.0);
        assert!(has_cold, "expected some samples from cold storage");
        assert!(has_hot, "expected some samples from hot storage");
    }

    #[test]
    fn test_mmap_buffer_total_count() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(5, 50, 4, 1, path).unwrap();
        for _ in 0..30 {
            buf.push(sample_record(4)).unwrap();
        }
        // hot: 5, cold: 25
        assert_eq!(buf.len(), 30);
        assert_eq!(buf.hot.len(), 5);
        assert_eq!(buf.cold_count, 25);
    }

    #[test]
    fn test_mmap_buffer_deterministic_sampling() {
        let (_tmp, path1) = temp_cold_path();
        let (_tmp2, path2) = temp_cold_path();
        let mut buf1 = MmapReplayBuffer::new(10, 100, 4, 1, path1).unwrap();
        let mut buf2 = MmapReplayBuffer::new(10, 100, 4, 1, path2).unwrap();

        for i in 0..20 {
            let mut rec = sample_record(4);
            rec.reward = i as f32;
            buf1.push(rec.clone()).unwrap();
            buf2.push(rec).unwrap();
        }

        let b1 = buf1.sample(15, 42).unwrap();
        let b2 = buf2.sample(15, 42).unwrap();

        assert_eq!(b1.observations, b2.observations);
        assert_eq!(b1.rewards, b2.rewards);
        assert_eq!(b1.terminated, b2.terminated);
    }

    #[test]
    fn test_mmap_buffer_cleanup_removes_file() {
        let (_tmp, path) = temp_cold_path();
        let cold_path = path.clone();
        let mut buf = MmapReplayBuffer::new(5, 50, 4, 1, path).unwrap();

        // Push enough to create the cold file.
        for _ in 0..10 {
            buf.push(sample_record(4)).unwrap();
        }
        assert!(cold_path.exists(), "cold file should exist after spill");

        buf.close().unwrap();
        assert!(
            !cold_path.exists(),
            "cold file should be removed after close()"
        );
    }

    #[test]
    fn test_mmap_buffer_large_obs_dim() {
        let (_tmp, path) = temp_cold_path();
        let obs_dim = 28224; // Atari-scale (84*84*4)
        let act_dim = 1;
        let mut buf = MmapReplayBuffer::new(5, 20, obs_dim, act_dim, path).unwrap();

        for i in 0..10 {
            let mut rec = sample_record_multidim(obs_dim, act_dim);
            rec.reward = i as f32;
            // Set first obs element to distinguish records.
            rec.obs[0] = i as f32;
            buf.push(rec).unwrap();
        }

        assert_eq!(buf.len(), 10);
        assert_eq!(buf.hot.len(), 5);
        assert_eq!(buf.cold_count, 5);

        // Sample and verify obs_dim is preserved.
        let batch = buf.sample(5, 42).unwrap();
        assert_eq!(batch.observations.len(), 5 * obs_dim);
        assert_eq!(batch.obs_dim, obs_dim);
    }

    // ---- push_batch tests ----

    #[test]
    fn test_push_batch_fills_hot_correctly() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(100, 1000, 4, 1, path).unwrap();

        let n = 10;
        let obs: Vec<f32> = (0..n * 4).map(|i| i as f32).collect();
        let next_obs: Vec<f32> = (0..n * 4).map(|i| i as f32 + 100.0).collect();
        let actions: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let rewards: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let terminated = vec![0.0f32; n];
        let truncated = vec![0.0f32; n];

        buf.push_batch(&obs, &next_obs, &actions, &rewards, &terminated, &truncated)
            .unwrap();

        assert_eq!(buf.len(), 10);
        assert_eq!(buf.cold_count, 0);

        // Sample all and verify rewards round-trip.
        let batch = buf.sample(10, 42).unwrap();
        assert_eq!(batch.batch_size, 10);
        assert_eq!(batch.rewards.len(), 10);
    }

    #[test]
    fn test_push_batch_triggers_spill_to_cold() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(5, 100, 2, 1, path).unwrap();

        let n = 12;
        let obs: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
        let next_obs: Vec<f32> = (0..n * 2).map(|i| i as f32 + 100.0).collect();
        let actions: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let rewards: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let terminated = vec![0.0f32; n];
        let truncated = vec![0.0f32; n];

        buf.push_batch(&obs, &next_obs, &actions, &rewards, &terminated, &truncated)
            .unwrap();

        assert_eq!(buf.hot.len(), 5);
        assert_eq!(buf.cold_count, 7);
        assert_eq!(buf.len(), 12);
    }

    #[test]
    fn test_push_batch_shape_mismatch() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(100, 1000, 4, 1, path).unwrap();

        // Wrong obs length.
        let result = buf.push_batch(
            &[1.0, 2.0, 3.0], // 3 instead of 4
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_push_batch_terminated_truncated_flags() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(100, 1000, 2, 1, path).unwrap();

        let obs = vec![1.0, 2.0, 3.0, 4.0]; // 2 records, obs_dim=2
        let next_obs = vec![5.0, 6.0, 7.0, 8.0];
        let actions = vec![0.0, 1.0];
        let rewards = vec![1.0, 2.0];
        let terminated = vec![1.0, 0.0]; // first terminated
        let truncated = vec![0.0, 1.0]; // second truncated

        buf.push_batch(&obs, &next_obs, &actions, &rewards, &terminated, &truncated)
            .unwrap();

        let batch = buf.sample(2, 42).unwrap();
        // Both records should be present; verify bools are meaningful.
        assert_eq!(batch.terminated.len(), 2);
        assert_eq!(batch.truncated.len(), 2);
    }

    // ---- cold ring-buffer eviction tests ----

    #[test]
    fn test_cold_ring_buffer_eviction_overwrites_oldest() {
        let (_tmp, path) = temp_cold_path();
        // hot_capacity=3, total_capacity=6 => cold_capacity=3
        let mut buf = MmapReplayBuffer::new(3, 6, 2, 1, path).unwrap();

        // Push 10 records. After 6, cold is full (3 hot + 3 cold).
        // Records 7-10 should evict oldest cold entries via ring-buffer.
        for i in 0..10 {
            let rec = ExperienceRecord {
                obs: vec![i as f32; 2],
                next_obs: vec![i as f32 + 100.0; 2],
                action: vec![i as f32 * 0.1],
                reward: i as f32,
                terminated: false,
                truncated: false,
            };
            buf.push(rec).unwrap();
        }

        // Total should be capped at total_capacity=6.
        assert_eq!(buf.len(), 6);
        assert_eq!(buf.hot.len(), 3);
        assert_eq!(buf.cold_count, 3);

        // Sample all 6 records. The cold ring-buffer should contain records
        // with rewards from the spill sequence, not the very oldest ones.
        let batch = buf.sample(6, 42).unwrap();
        assert_eq!(batch.batch_size, 6);
        assert_eq!(batch.rewards.len(), 6);

        // Rewards 0, 1, 2 were the earliest cold entries but should have been
        // overwritten. Hot has the last 3 pushed (7, 8, 9).
        // Cold should have the 3 most recent spills.
        for &r in &batch.rewards {
            assert!(
                r >= 4.0,
                "expected only recent records (reward >= 4.0), got {r}"
            );
        }
    }

    #[test]
    fn test_cold_eviction_cold_write_pos_wraps() {
        let (_tmp, path) = temp_cold_path();
        // hot_capacity=2, total_capacity=4 => cold_capacity=2
        let mut buf = MmapReplayBuffer::new(2, 4, 1, 1, path).unwrap();

        // Push 8 records to force multiple cold wrap-arounds.
        for i in 0..8 {
            let rec = ExperienceRecord {
                obs: vec![i as f32],
                next_obs: vec![i as f32 + 10.0],
                action: vec![0.0],
                reward: i as f32,
                terminated: false,
                truncated: false,
            };
            buf.push(rec).unwrap();
        }

        assert_eq!(buf.len(), 4);
        assert_eq!(buf.cold_count, 2);
        // cold_write_pos should have wrapped: 6 spills into capacity 2 => pos 0
        assert_eq!(buf.cold_write_pos, 0);
    }

    #[test]
    fn test_sampling_after_cold_eviction_returns_valid_data() {
        let (_tmp, path) = temp_cold_path();
        // hot=5, total=8, cold_capacity=3
        let mut buf = MmapReplayBuffer::new(5, 8, 3, 1, path).unwrap();

        // Push 20 records to cause many evictions.
        for i in 0..20 {
            let rec = ExperienceRecord {
                obs: vec![i as f32; 3],
                next_obs: vec![i as f32 + 0.5; 3],
                action: vec![i as f32 * 0.01],
                reward: i as f32,
                terminated: i % 3 == 0,
                truncated: i % 5 == 0,
            };
            buf.push(rec).unwrap();
        }

        assert_eq!(buf.len(), 8);

        // Sampling should not panic and data should be internally consistent.
        let batch = buf.sample(8, 123).unwrap();
        assert_eq!(batch.batch_size, 8);
        assert_eq!(batch.observations.len(), 8 * 3);
        assert_eq!(batch.next_observations.len(), 8 * 3);
        assert_eq!(batch.actions.len(), 8);
        assert_eq!(batch.rewards.len(), 8);

        // Each sampled obs should match its reward (obs was [r; 3]).
        for i in 0..8 {
            let r = batch.rewards[i];
            let obs_slice = &batch.observations[i * 3..(i + 1) * 3];
            for &v in obs_slice {
                assert!((v - r).abs() < 1e-6, "obs {v} should match reward {r}");
            }
        }
    }

    #[test]
    fn test_push_batch_then_sample_consistency() {
        let (_tmp, path) = temp_cold_path();
        let mut buf = MmapReplayBuffer::new(5, 15, 2, 1, path).unwrap();

        // Push 3 batches of 5 to fill past hot and into cold.
        for batch_idx in 0..3 {
            let base = batch_idx as f32 * 5.0;
            let n = 5;
            let obs: Vec<f32> = (0..n)
                .flat_map(|i| {
                    let v = base + i as f32;
                    vec![v, v + 0.1]
                })
                .collect();
            let next_obs: Vec<f32> = (0..n)
                .flat_map(|i| {
                    let v = base + i as f32 + 100.0;
                    vec![v, v + 0.1]
                })
                .collect();
            let actions: Vec<f32> = (0..n).map(|i| (base + i as f32) * 0.01).collect();
            let rewards: Vec<f32> = (0..n).map(|i| base + i as f32).collect();
            let terminated = vec![0.0f32; n];
            let truncated = vec![0.0f32; n];
            buf.push_batch(&obs, &next_obs, &actions, &rewards, &terminated, &truncated)
                .unwrap();
        }

        assert_eq!(buf.len(), 15);
        assert_eq!(buf.hot.len(), 5);
        assert_eq!(buf.cold_count, 10);

        // Sample and check structural validity.
        let batch = buf.sample(15, 7).unwrap();
        assert_eq!(batch.batch_size, 15);
        assert_eq!(batch.observations.len(), 15 * 2);

        // Each observation pair should match: obs[0] and obs[1] differ by 0.1.
        for i in 0..15 {
            let o0 = batch.observations[i * 2];
            let o1 = batch.observations[i * 2 + 1];
            assert!(
                (o1 - o0 - 0.1).abs() < 1e-5,
                "obs pair mismatch at sample {i}: {o0}, {o1}"
            );
        }
    }
}
