use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use memmap2::Mmap;
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
/// - `cold`: memory-mapped file for overflow data
/// - Sampling prefers hot data but can reach into cold
pub struct MmapReplayBuffer {
    hot: ReplayBuffer,
    cold_path: PathBuf,
    cold_file: Option<File>,
    cold_mmap: Option<Mmap>,
    cold_count: usize,
    obs_dim: usize,
    act_dim: usize,
    hot_capacity: usize,
    total_capacity: usize,
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

        Ok(Self {
            hot: ReplayBuffer::new(hot_capacity, obs_dim, act_dim),
            cold_path,
            cold_file: None,
            cold_mmap: None,
            cold_count: 0,
            obs_dim,
            act_dim,
            hot_capacity,
            total_capacity,
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
            let cold_capacity = self.total_capacity - self.hot_capacity;
            if self.cold_count < cold_capacity {
                // Read the record at the current write position (the one about
                // to be overwritten) from the hot buffer.
                let oldest = self.read_oldest_hot_record();
                self.append_to_cold(&oldest)?;
            }
            // If cold is also full, the oldest cold data is effectively lost
            // (ring-buffer semantics for total_capacity). We could implement
            // cold ring-buffer rewriting but that adds complexity; for now the
            // cold file is append-only up to cold_capacity.
        }

        self.hot.push(record)?;
        Ok(())
    }

    /// Sample `batch_size` records uniformly from hot + cold storage.
    /// Uses ChaCha8Rng seeded with `seed` for deterministic reproducibility.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
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
        // Drop the mmap first so the file can be removed.
        self.cold_mmap.take();
        self.cold_file.take();
        if self.cold_path.exists() {
            std::fs::remove_file(&self.cold_path)?;
        }
        self.cold_count = 0;
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

    /// Append a record to the cold file and re-mmap it.
    fn append_to_cold(&mut self, record: &ExperienceRecord) -> Result<(), RloxError> {
        self.ensure_cold_file()?;

        // Serialize record.
        let record_size = self.record_byte_size();
        let mut buf = Vec::with_capacity(record_size);
        for &v in &record.obs {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &record.next_obs {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &record.action {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&record.reward.to_le_bytes());
        buf.push(record.terminated as u8);
        buf.push(record.truncated as u8);

        let file = self.cold_file.as_mut().expect("cold_file must be open");
        file.write_all(&buf)?;
        file.flush()?;

        self.cold_count += 1;

        // Update header with new record count.
        self.write_cold_header()?;

        // Re-mmap the file.
        self.remap_cold()?;

        Ok(())
    }

    /// Ensure the cold file is open and has a header written.
    fn ensure_cold_file(&mut self) -> Result<&mut File, RloxError> {
        if self.cold_file.is_none() {
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(true)
                .open(&self.cold_path)?;
            self.cold_file = Some(file);
            // Write initial header.
            self.write_cold_header()?;
        }
        Ok(self.cold_file.as_mut().expect("just created"))
    }

    /// Write/overwrite the 24-byte header.
    fn write_cold_header(&mut self) -> Result<(), RloxError> {
        use std::io::Seek;
        let file = self.cold_file.as_mut().expect("cold_file must be open");
        file.seek(std::io::SeekFrom::Start(0))?;
        file.write_all(&(self.obs_dim as u64).to_le_bytes())?;
        file.write_all(&(self.act_dim as u64).to_le_bytes())?;
        file.write_all(&(self.cold_count as u64).to_le_bytes())?;
        file.flush()?;
        // Seek back to end for future appends.
        file.seek(std::io::SeekFrom::End(0))?;
        Ok(())
    }

    /// Re-create the mmap over the cold file.
    fn remap_cold(&mut self) -> Result<(), RloxError> {
        // Drop old mmap first.
        self.cold_mmap.take();
        let file = self.cold_file.as_ref().expect("cold_file must be open");
        // SAFETY: The file is exclusively owned by this buffer. We only
        // write via `append_to_cold` which flushes before we remap. No
        // concurrent writers exist.
        let mmap = unsafe { Mmap::map(file)? };
        self.cold_mmap = Some(mmap);
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
}
