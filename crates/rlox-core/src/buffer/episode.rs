//! Episode boundary tracking for ring buffers.
//!
//! Provides [`EpisodeTracker`] which maintains metadata about episode boundaries
//! within a ring buffer, enabling sequence sampling and HER-style relabeling.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::RloxError;

/// Trait for episode-aware buffer components.
///
/// Anything that needs to track episode boundaries in a ring buffer
/// implements this trait. Used by SequenceReplayBuffer and HERBuffer.
pub trait EpisodeAware {
    /// Notify that a transition was pushed at `write_pos` with `done` flag.
    fn notify_push(&mut self, write_pos: usize, done: bool);

    /// Invalidate any episodes that overlap with the overwritten region.
    fn invalidate_overwritten(&mut self, write_pos: usize, count: usize);

    /// Number of complete (terminated/truncated) episodes currently tracked.
    fn num_complete_episodes(&self) -> usize;
}

/// Metadata for a single episode within the ring buffer.
#[derive(Debug, Clone, Copy)]
pub struct EpisodeMeta {
    /// Starting position in the ring buffer.
    pub start: usize,
    /// Number of transitions in this episode.
    pub length: usize,
    /// Whether this episode is complete (reached done=true).
    /// Incomplete episodes are still being built (the current in-progress
    /// episode). They are not returned by `eligible_episodes`.
    pub complete: bool,
}

/// A contiguous window of transitions within an episode,
/// suitable for sequence sampling.
#[derive(Debug, Clone, Copy)]
pub struct EpisodeWindow {
    /// Index of the episode this window belongs to.
    pub episode_idx: usize,
    /// Starting position in the ring buffer.
    pub ring_start: usize,
    /// Number of transitions in this window.
    pub length: usize,
}

/// Tracks episode boundaries within a ring buffer.
///
/// Maintains a list of [`EpisodeMeta`] entries, invalidating episodes
/// whose data has been overwritten by the ring buffer's write pointer.
/// Provides efficient sampling of contiguous windows for sequence models.
#[derive(Debug)]
pub struct EpisodeTracker {
    ring_capacity: usize,
    episodes: Vec<EpisodeMeta>,
    /// Start position of the episode currently being built.
    current_episode_start: Option<usize>,
    current_episode_length: usize,
}

impl EpisodeTracker {
    /// Create a new episode tracker for a ring buffer with the given capacity.
    pub fn new(ring_capacity: usize) -> Self {
        Self {
            ring_capacity,
            episodes: Vec::new(),
            current_episode_start: None,
            current_episode_length: 0,
        }
    }

    /// Record a push at `write_pos`. If `done`, the current episode is finalized.
    #[inline]
    pub fn notify_push(&mut self, write_pos: usize, done: bool) {
        if self.current_episode_start.is_none() {
            self.current_episode_start = Some(write_pos);
            self.current_episode_length = 0;
        }

        self.current_episode_length += 1;

        if done {
            self.episodes.push(EpisodeMeta {
                start: self.current_episode_start.take().unwrap_or(write_pos),
                length: self.current_episode_length,
                complete: true,
            });
            self.current_episode_start = None;
            self.current_episode_length = 0;
        }
    }

    /// Remove any episodes whose transitions have been overwritten.
    ///
    /// Called when the ring buffer wraps. An episode is invalidated if any
    /// of its positions overlap with the region `[write_pos, write_pos + count)`
    /// modulo the ring capacity.
    #[inline]
    pub fn invalidate_overwritten(&mut self, write_pos: usize, count: usize) {
        self.episodes.retain(|ep| {
            !ring_range_overlaps(ep.start, ep.length, write_pos, count, self.ring_capacity)
        });

        // Also invalidate current in-progress episode if it overlaps
        if let Some(start) = self.current_episode_start {
            if ring_range_overlaps(
                start,
                self.current_episode_length,
                write_pos,
                count,
                self.ring_capacity,
            ) {
                self.current_episode_start = None;
                self.current_episode_length = 0;
            }
        }
    }

    /// Number of complete episodes currently tracked.
    #[inline]
    pub fn num_complete_episodes(&self) -> usize {
        self.episodes.iter().filter(|ep| ep.complete).count()
    }

    /// All currently tracked episodes (complete and in-progress).
    pub fn episodes(&self) -> &[EpisodeMeta] {
        &self.episodes
    }

    /// Indices of episodes long enough for a given sequence length.
    pub fn eligible_episodes(&self, min_length: usize) -> Vec<usize> {
        self.episodes
            .iter()
            .enumerate()
            .filter(|(_, ep)| ep.complete && ep.length >= min_length)
            .map(|(i, _)| i)
            .collect()
    }

    /// Sample `batch_size` windows of `seq_len` consecutive transitions,
    /// each entirely within a single complete episode.
    ///
    /// Uses ChaCha8Rng seeded with `seed`.
    pub fn sample_windows(
        &self,
        batch_size: usize,
        seq_len: usize,
        seed: u64,
    ) -> Result<Vec<EpisodeWindow>, RloxError> {
        let eligible = self.eligible_episodes(seq_len);
        if eligible.is_empty() {
            return Err(RloxError::BufferError(format!(
                "no episodes with length >= {seq_len}"
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut windows = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let ep_idx = eligible[rng.random_range(0..eligible.len())];
            let ep = &self.episodes[ep_idx];

            // Random start offset within the episode
            let max_offset = ep.length - seq_len;
            let offset = if max_offset == 0 {
                0
            } else {
                rng.random_range(0..=max_offset)
            };

            let ring_start = (ep.start + offset) % self.ring_capacity;

            windows.push(EpisodeWindow {
                episode_idx: ep_idx,
                ring_start,
                length: seq_len,
            });
        }

        Ok(windows)
    }
}

impl EpisodeAware for EpisodeTracker {
    fn notify_push(&mut self, write_pos: usize, done: bool) {
        EpisodeTracker::notify_push(self, write_pos, done);
    }

    fn invalidate_overwritten(&mut self, write_pos: usize, count: usize) {
        EpisodeTracker::invalidate_overwritten(self, write_pos, count);
    }

    fn num_complete_episodes(&self) -> usize {
        EpisodeTracker::num_complete_episodes(self)
    }
}

/// Check if two ring-buffer ranges overlap in O(1) using modular arithmetic.
///
/// Range A: `[a_start, a_start + a_len)` mod `cap`
/// Range B: `[b_start, b_start + b_len)` mod `cap`
///
/// Two circular ranges overlap iff the start of either range falls
/// within the other range.
#[inline]
fn ring_range_overlaps(
    a_start: usize,
    a_len: usize,
    b_start: usize,
    b_len: usize,
    cap: usize,
) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a_in_b = (a_start + cap - b_start) % cap < b_len;
    let b_in_a = (b_start + cap - a_start) % cap < a_len;
    a_in_b || b_in_a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker_is_empty() {
        let tracker = EpisodeTracker::new(100);
        assert_eq!(tracker.num_complete_episodes(), 0);
        assert!(tracker.episodes().is_empty());
    }

    #[test]
    fn test_single_episode_tracked() {
        let mut tracker = EpisodeTracker::new(100);
        for i in 0..5 {
            tracker.notify_push(i, i == 4); // done at position 4
        }
        assert_eq!(tracker.num_complete_episodes(), 1);
        assert_eq!(tracker.episodes()[0].length, 5);
        assert_eq!(tracker.episodes()[0].start, 0);
        assert!(tracker.episodes()[0].complete);
    }

    #[test]
    fn test_multiple_episodes() {
        let mut tracker = EpisodeTracker::new(100);
        let mut pos = 0;
        // Episode 1: length 3
        for _ in 0..3 {
            tracker.notify_push(pos, pos == 2);
            pos += 1;
        }
        // Episode 2: length 5
        for _ in 0..5 {
            tracker.notify_push(pos, pos == 7);
            pos += 1;
        }
        // Episode 3: length 2
        for _ in 0..2 {
            tracker.notify_push(pos, pos == 9);
            pos += 1;
        }
        assert_eq!(tracker.num_complete_episodes(), 3);
        assert_eq!(tracker.episodes()[0].length, 3);
        assert_eq!(tracker.episodes()[1].length, 5);
        assert_eq!(tracker.episodes()[2].length, 2);
    }

    #[test]
    fn test_incomplete_episode_not_counted() {
        let mut tracker = EpisodeTracker::new(100);
        for i in 0..5 {
            tracker.notify_push(i, false);
        }
        assert_eq!(tracker.num_complete_episodes(), 0);
    }

    #[test]
    fn test_invalidate_removes_overwritten() {
        let mut tracker = EpisodeTracker::new(10);
        // Episode at positions 0..5
        for i in 0..5 {
            tracker.notify_push(i, i == 4);
        }
        assert_eq!(tracker.num_complete_episodes(), 1);

        // Overwrite positions 0..3
        tracker.invalidate_overwritten(0, 3);
        assert_eq!(tracker.num_complete_episodes(), 0);
    }

    #[test]
    fn test_sample_windows_within_episode() {
        let mut tracker = EpisodeTracker::new(100);
        // Two episodes of length 10
        for i in 0..10 {
            tracker.notify_push(i, i == 9);
        }
        for i in 10..20 {
            tracker.notify_push(i, i == 19);
        }
        assert_eq!(tracker.num_complete_episodes(), 2);

        let windows = tracker.sample_windows(5, 5, 42).unwrap();
        assert_eq!(windows.len(), 5);
        for w in &windows {
            assert_eq!(w.length, 5);
            // Verify window is within its episode
            let ep = &tracker.episodes()[w.episode_idx];
            let ep_end = ep.start + ep.length;
            assert!(
                w.ring_start >= ep.start && w.ring_start + w.length <= ep_end,
                "window [{}, {}) not within episode [{}, {})",
                w.ring_start,
                w.ring_start + w.length,
                ep.start,
                ep_end
            );
        }
    }

    #[test]
    fn test_sample_windows_deterministic() {
        let mut tracker = EpisodeTracker::new(100);
        for i in 0..10 {
            tracker.notify_push(i, i == 9);
        }
        let w1 = tracker.sample_windows(5, 3, 42).unwrap();
        let w2 = tracker.sample_windows(5, 3, 42).unwrap();
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert_eq!(a.ring_start, b.ring_start);
            assert_eq!(a.episode_idx, b.episode_idx);
            assert_eq!(a.length, b.length);
        }
    }

    #[test]
    fn test_sample_windows_rejects_too_long_seq() {
        let mut tracker = EpisodeTracker::new(100);
        for i in 0..3 {
            tracker.notify_push(i, i == 2);
        }
        let result = tracker.sample_windows(1, 5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_eligible_episodes_filters_short() {
        let mut tracker = EpisodeTracker::new(100);
        let mut pos = 0;
        // Episode 0: length 2
        for _ in 0..2 {
            tracker.notify_push(pos, pos == 1);
            pos += 1;
        }
        // Episode 1: length 5
        for _ in 0..5 {
            tracker.notify_push(pos, pos == 6);
            pos += 1;
        }
        // Episode 2: length 3
        for _ in 0..3 {
            tracker.notify_push(pos, pos == 9);
            pos += 1;
        }
        // Episode 3: length 8
        for _ in 0..8 {
            tracker.notify_push(pos, pos == 17);
            pos += 1;
        }
        let eligible = tracker.eligible_episodes(4);
        assert_eq!(eligible, vec![1, 3]);
    }

    #[test]
    fn test_invalidate_partial_episode() {
        let mut tracker = EpisodeTracker::new(10);
        // Episode at positions 0..5
        for i in 0..5 {
            tracker.notify_push(i, i == 4);
        }
        // Overwrite position 2 (middle of episode)
        tracker.invalidate_overwritten(2, 1);
        assert_eq!(
            tracker.num_complete_episodes(),
            0,
            "partially overwritten episode should be removed"
        );
    }

    #[test]
    fn test_consecutive_dones() {
        let mut tracker = EpisodeTracker::new(100);
        tracker.notify_push(0, true); // Episode of length 1
        tracker.notify_push(1, true); // Another episode of length 1
        assert_eq!(tracker.num_complete_episodes(), 2);
        assert_eq!(tracker.episodes()[0].length, 1);
        assert_eq!(tracker.episodes()[1].length, 1);
    }

    #[test]
    fn test_empty_tracker_sample_windows_errors() {
        let tracker = EpisodeTracker::new(100);
        let result = tracker.sample_windows(1, 1, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_transition_episode() {
        let mut tracker = EpisodeTracker::new(100);
        tracker.notify_push(0, true);
        assert_eq!(tracker.num_complete_episodes(), 1);
        let windows = tracker.sample_windows(1, 1, 42).unwrap();
        assert_eq!(windows[0].ring_start, 0);
        assert_eq!(windows[0].length, 1);
    }

    #[test]
    fn test_trait_object_safety() {
        let tracker: Box<dyn EpisodeAware> = Box::new(EpisodeTracker::new(100));
        assert_eq!(tracker.num_complete_episodes(), 0);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_episode_count_matches_dones(
                n in 1usize..200,
                done_rate in 0.05f64..0.5,
            ) {
                let mut tracker = EpisodeTracker::new(n * 2); // large capacity, no wrapping
                let mut expected_complete = 0;
                for i in 0..n {
                    let done = ((i as f64 + 1.0) * done_rate) as usize
                        > (i as f64 * done_rate) as usize;
                    tracker.notify_push(i, done);
                    if done {
                        expected_complete += 1;
                    }
                }
                prop_assert_eq!(
                    tracker.num_complete_episodes(),
                    expected_complete,
                    "expected {} complete episodes", expected_complete
                );
            }

            #[test]
            fn prop_window_within_bounds(
                ep_len in 5usize..50,
                seq_len in 1usize..5,
                batch_size in 1usize..10,
            ) {
                let cap = ep_len * 3;
                let mut tracker = EpisodeTracker::new(cap);
                for i in 0..ep_len {
                    tracker.notify_push(i, i == ep_len - 1);
                }
                let windows = tracker.sample_windows(batch_size, seq_len, 42).unwrap();
                for w in &windows {
                    prop_assert!(
                        w.ring_start + w.length <= cap,
                        "window [{}, {}) exceeds capacity {cap}",
                        w.ring_start,
                        w.ring_start + w.length
                    );
                }
            }

            #[test]
            fn prop_no_cross_episode_windows(
                n_episodes in 2usize..10,
                ep_len in 5usize..20,
                seq_len in 1usize..5,
            ) {
                let cap = n_episodes * ep_len * 2;
                let mut tracker = EpisodeTracker::new(cap);
                let mut pos = 0;
                for _ in 0..n_episodes {
                    for j in 0..ep_len {
                        tracker.notify_push(pos, j == ep_len - 1);
                        pos += 1;
                    }
                }
                let windows = tracker.sample_windows(n_episodes * 2, seq_len, 42).unwrap();
                for w in &windows {
                    let ep = &tracker.episodes()[w.episode_idx];
                    let ep_end = ep.start + ep.length;
                    prop_assert!(
                        w.ring_start >= ep.start && w.ring_start + w.length <= ep_end,
                        "window crosses episode boundary"
                    );
                }
            }

            #[test]
            fn prop_invalidation_never_returns_overwritten(
                cap in 10usize..100,
                n_pushes in 1usize..300,
            ) {
                let mut tracker = EpisodeTracker::new(cap);
                for (write_pos, i) in (0..n_pushes).enumerate() {
                    let done = i % 7 == 6; // episodes of ~7 steps
                    if write_pos >= cap {
                        // Wrapping: invalidate the position about to be overwritten
                        tracker.invalidate_overwritten(write_pos % cap, 1);
                    }
                    tracker.notify_push(write_pos % cap, done);
                }
                // All remaining episodes should have valid start positions
                for ep in tracker.episodes() {
                    prop_assert!(
                        ep.start < cap,
                        "episode start {} >= capacity {cap}",
                        ep.start
                    );
                }
            }
        }
    }
}
