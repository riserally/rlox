use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::RloxError;

/// Metadata attached to each transition for provenance tracking.
///
/// Fixed-size serialization: 48 bytes (little-endian).
/// Layout: env_id(4) + episode_id(8) + step_in_episode(4) + global_step(8)
///       + policy_version(8) + reward_model_version(8) + timestamp_ns(8) = 48
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionMeta {
    pub env_id: u32,
    pub episode_id: u64,
    pub step_in_episode: u32,
    pub global_step: u64,
    pub policy_version: u64,
    pub reward_model_version: u64,
    pub timestamp_ns: u64,
}

const SERIALIZED_SIZE: usize = 48;

impl TransitionMeta {
    /// Create a new `TransitionMeta`, auto-filling `timestamp_ns` from the
    /// system clock and defaulting `reward_model_version` to 0.
    pub fn new(
        env_id: u32,
        episode_id: u64,
        step_in_episode: u32,
        global_step: u64,
        policy_version: u64,
    ) -> Self {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_nanos() as u64;
        Self {
            env_id,
            episode_id,
            step_in_episode,
            global_step,
            policy_version,
            reward_model_version: 0,
            timestamp_ns,
        }
    }

    /// Serialize to a fixed 48-byte little-endian representation.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(SERIALIZED_SIZE);
        buf.extend_from_slice(&self.env_id.to_le_bytes());
        buf.extend_from_slice(&self.episode_id.to_le_bytes());
        buf.extend_from_slice(&self.step_in_episode.to_le_bytes());
        buf.extend_from_slice(&self.global_step.to_le_bytes());
        buf.extend_from_slice(&self.policy_version.to_le_bytes());
        buf.extend_from_slice(&self.reward_model_version.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }

    /// Deserialize from a 48-byte little-endian buffer.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, RloxError> {
        if bytes.len() != SERIALIZED_SIZE {
            return Err(RloxError::BufferError(format!(
                "TransitionMeta requires exactly {SERIALIZED_SIZE} bytes, got {}",
                bytes.len()
            )));
        }

        let env_id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let episode_id = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let step_in_episode = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let global_step = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let policy_version = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let reward_model_version = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let timestamp_ns = u64::from_le_bytes(bytes[40..48].try_into().unwrap());

        Ok(Self {
            env_id,
            episode_id,
            step_in_episode,
            global_step,
            policy_version,
            reward_model_version,
            timestamp_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta() -> TransitionMeta {
        TransitionMeta {
            env_id: 42,
            episode_id: 1000,
            step_in_episode: 7,
            global_step: 50000,
            policy_version: 3,
            reward_model_version: 1,
            timestamp_ns: 1_700_000_000_000_000_000,
        }
    }

    #[test]
    fn serialize_is_48_bytes() {
        let meta = sample_meta();
        let bytes = meta.serialize();
        assert_eq!(bytes.len(), 48);
    }

    #[test]
    fn roundtrip() {
        let meta = sample_meta();
        let bytes = meta.serialize();
        let restored = TransitionMeta::deserialize(&bytes).unwrap();
        assert_eq!(meta, restored);
    }

    #[test]
    fn deserialize_wrong_length_errors() {
        let result = TransitionMeta::deserialize(&[0u8; 47]);
        assert!(result.is_err());
        let result = TransitionMeta::deserialize(&[0u8; 49]);
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_zeros() {
        let meta = TransitionMeta {
            env_id: 0,
            episode_id: 0,
            step_in_episode: 0,
            global_step: 0,
            policy_version: 0,
            reward_model_version: 0,
            timestamp_ns: 0,
        };
        let bytes = meta.serialize();
        let restored = TransitionMeta::deserialize(&bytes).unwrap();
        assert_eq!(meta, restored);
    }

    #[test]
    fn roundtrip_max_values() {
        let meta = TransitionMeta {
            env_id: u32::MAX,
            episode_id: u64::MAX,
            step_in_episode: u32::MAX,
            global_step: u64::MAX,
            policy_version: u64::MAX,
            reward_model_version: u64::MAX,
            timestamp_ns: u64::MAX,
        };
        let bytes = meta.serialize();
        let restored = TransitionMeta::deserialize(&bytes).unwrap();
        assert_eq!(meta, restored);
    }

    #[test]
    fn test_transition_meta_roundtrip() {
        let meta = TransitionMeta::new(7, 100, 5, 99999, 2);
        let bytes = meta.serialize();
        let restored = TransitionMeta::deserialize(&bytes).unwrap();
        assert_eq!(meta, restored);
    }

    #[test]
    fn test_transition_meta_timestamp_nonzero() {
        let meta = TransitionMeta::new(0, 0, 0, 0, 0);
        assert!(meta.timestamp_ns > 0, "auto-filled timestamp must be non-zero");
    }

    #[test]
    fn test_transition_meta_serialize_size() {
        let meta = TransitionMeta::new(1, 2, 3, 4, 5);
        assert_eq!(meta.serialize().len(), 48);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn roundtrip_arbitrary(
                env_id: u32,
                episode_id: u64,
                step_in_episode: u32,
                global_step: u64,
                policy_version: u64,
                reward_model_version: u64,
                timestamp_ns: u64,
            ) {
                let meta = TransitionMeta {
                    env_id,
                    episode_id,
                    step_in_episode,
                    global_step,
                    policy_version,
                    reward_model_version,
                    timestamp_ns,
                };
                let bytes = meta.serialize();
                prop_assert_eq!(bytes.len(), 48);
                let restored = TransitionMeta::deserialize(&bytes).unwrap();
                prop_assert_eq!(meta, restored);
            }
        }
    }
}
