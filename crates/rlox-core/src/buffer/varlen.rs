/// Packed variable-length sequence storage.
///
/// Uses the Arrow ListArray pattern: a flat contiguous data array
/// plus an offsets array that marks sequence boundaries.
pub struct VarLenStore {
    data: Vec<u32>,
    offsets: Vec<u64>,
}

impl Default for VarLenStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VarLenStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Append a variable-length sequence.
    pub fn push(&mut self, sequence: &[u32]) {
        self.data.extend_from_slice(sequence);
        self.offsets.push(self.data.len() as u64);
    }

    /// Number of sequences stored.
    pub fn num_sequences(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Total number of elements across all sequences.
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    /// Get the i-th sequence as a slice.
    ///
    /// # Panics
    /// Panics if `index >= num_sequences()`.
    pub fn get(&self, index: usize) -> &[u32] {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        &self.data[start..end]
    }

    /// Length of the i-th sequence.
    pub fn sequence_len(&self, index: usize) -> usize {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        end - start
    }

    /// Raw flat data array.
    pub fn flat_data(&self) -> &[u32] {
        &self.data
    }

    /// Raw offsets array.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_store() {
        let store = VarLenStore::new();
        assert_eq!(store.num_sequences(), 0);
        assert_eq!(store.total_elements(), 0);
    }

    #[test]
    fn push_and_retrieve() {
        let mut store = VarLenStore::new();
        store.push(&[10, 20, 30]);
        store.push(&[40, 50]);
        assert_eq!(store.num_sequences(), 2);
        assert_eq!(store.get(0), &[10, 20, 30]);
        assert_eq!(store.get(1), &[40, 50]);
    }

    #[test]
    fn total_elements_correct() {
        let mut store = VarLenStore::new();
        store.push(&[1, 2, 3]);
        store.push(&[4, 5]);
        assert_eq!(store.total_elements(), 5);
    }

    #[test]
    fn no_padding_waste() {
        let mut store = VarLenStore::new();
        store.push(&[1, 2, 3]);
        store.push(&[4, 5]);
        assert_eq!(store.flat_data(), &[1, 2, 3, 4, 5]);
        assert_eq!(store.offsets(), &[0, 3, 5]);
    }

    #[test]
    fn sequence_len() {
        let mut store = VarLenStore::new();
        store.push(&[1, 2, 3]);
        store.push(&[4]);
        assert_eq!(store.sequence_len(0), 3);
        assert_eq!(store.sequence_len(1), 1);
    }

    #[test]
    fn default_is_empty() {
        let store = VarLenStore::default();
        assert_eq!(store.num_sequences(), 0);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;
        use proptest::collection::vec;

        proptest! {
            #[test]
            fn varlen_total_equals_sum_of_lengths(sequences in vec(vec(0u32..1000, 1..100), 1..50)) {
                let mut store = VarLenStore::new();
                let expected: usize = sequences.iter().map(|s| s.len()).sum();
                for seq in &sequences {
                    store.push(seq);
                }
                prop_assert_eq!(store.total_elements(), expected);
            }

            #[test]
            fn varlen_roundtrip(sequences in vec(vec(0u32..1000, 1..100), 1..50)) {
                let mut store = VarLenStore::new();
                for seq in &sequences {
                    store.push(seq);
                }
                for (i, seq) in sequences.iter().enumerate() {
                    prop_assert_eq!(store.get(i), seq.as_slice());
                }
            }

            #[test]
            fn varlen_num_sequences_matches(sequences in vec(vec(0u32..1000, 1..50), 1..100)) {
                let mut store = VarLenStore::new();
                for seq in &sequences {
                    store.push(seq);
                }
                prop_assert_eq!(store.num_sequences(), sequences.len());
            }
        }
    }
}
