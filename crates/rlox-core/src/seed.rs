use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

/// Derive a per-environment seed from a master seed and an index.
///
/// Uses a simple but effective mixing strategy: XOR the master seed with
/// a hash derived from the index, ensuring each env gets a unique but
/// deterministic stream.
pub fn derive_seed(master: u64, index: usize) -> u64 {
    // Use a ChaCha8 round to mix master + index into a new seed.
    // This avoids trivial correlations between adjacent indices.
    let combined = master.wrapping_add(index as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut rng = ChaCha8Rng::seed_from_u64(combined);
    rand::Rng::random::<u64>(&mut rng)
}

/// Create a ChaCha8Rng from a seed value.
pub fn rng_from_seed(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_seed_is_deterministic() {
        let s1 = derive_seed(42, 0);
        let s2 = derive_seed(42, 0);
        assert_eq!(s1, s2);
    }

    #[test]
    fn derive_seed_differs_by_index() {
        let s0 = derive_seed(42, 0);
        let s1 = derive_seed(42, 1);
        let s2 = derive_seed(42, 2);
        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
        assert_ne!(s0, s2);
    }

    #[test]
    fn derive_seed_differs_by_master() {
        let s1 = derive_seed(1, 0);
        let s2 = derive_seed(2, 0);
        assert_ne!(s1, s2);
    }
}
