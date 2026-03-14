use crate::error::RloxError;

/// A single packed batch containing multiple sequences concatenated together.
#[derive(Debug, Clone)]
pub struct PackedBatch {
    /// Token IDs, concatenated sequences padded to fill the bin.
    pub input_ids: Vec<u32>,
    /// Attention mask: 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<u32>,
    /// Position IDs: per-sequence positions starting from 0.
    pub position_ids: Vec<u32>,
    /// Start indices of each sequence within this batch.
    pub sequence_starts: Vec<usize>,
}

/// Pack variable-length sequences into fixed-size bins using first-fit-decreasing.
///
/// Sorts sequences by length (longest first), then greedily assigns each
/// to the first bin that has enough remaining capacity. Creates a new bin
/// if no existing bin can fit the sequence.
///
/// Returns `Err` if any single sequence exceeds `max_length`.
pub fn pack_sequences(sequences: &[&[u32]], max_length: usize) -> Result<Vec<PackedBatch>, RloxError> {
    if sequences.is_empty() {
        return Ok(Vec::new());
    }

    // Validate: no sequence exceeds max_length
    for (i, seq) in sequences.iter().enumerate() {
        if seq.len() > max_length {
            return Err(RloxError::BufferError(format!(
                "sequence {} has length {} which exceeds max_length {}",
                i,
                seq.len(),
                max_length
            )));
        }
    }

    // Sort by length descending (first-fit-decreasing)
    let mut indexed: Vec<(usize, &[u32])> = sequences.iter().enumerate().map(|(i, s)| (i, *s)).collect();
    indexed.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    // Bins: track remaining capacity and accumulated sequences
    struct Bin {
        tokens: Vec<u32>,
        attention_mask: Vec<u32>,
        position_ids: Vec<u32>,
        sequence_starts: Vec<usize>,
        used: usize,
    }

    let mut bins: Vec<Bin> = Vec::new();

    for (_orig_idx, seq) in &indexed {
        let seq_len = seq.len();

        // First-fit: find first bin with enough room
        let mut placed = false;
        for bin in bins.iter_mut() {
            if bin.used + seq_len <= max_length {
                bin.sequence_starts.push(bin.used);
                bin.tokens.extend_from_slice(seq);
                bin.attention_mask.extend(std::iter::repeat(1u32).take(seq_len));
                for j in 0..seq_len {
                    bin.position_ids.push(j as u32);
                }
                bin.used += seq_len;
                placed = true;
                break;
            }
        }

        if !placed {
            let mut bin = Bin {
                tokens: Vec::with_capacity(max_length),
                attention_mask: Vec::with_capacity(max_length),
                position_ids: Vec::with_capacity(max_length),
                sequence_starts: Vec::new(),
                used: 0,
            };
            bin.sequence_starts.push(0);
            bin.tokens.extend_from_slice(seq);
            bin.attention_mask.extend(std::iter::repeat(1u32).take(seq_len));
            for j in 0..seq_len {
                bin.position_ids.push(j as u32);
            }
            bin.used = seq_len;
            bins.push(bin);
        }
    }

    // Convert bins to PackedBatch (pad to used length, not max_length — no wasteful padding)
    let result = bins
        .into_iter()
        .map(|bin| PackedBatch {
            input_ids: bin.tokens,
            attention_mask: bin.attention_mask,
            position_ids: bin.position_ids,
            sequence_starts: bin.sequence_starts,
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_sequences_one_bin_exact_fit() {
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 6).unwrap();
        assert_eq!(packed.len(), 1, "should produce 1 bin for total_len=max_len");
        assert_eq!(packed[0].input_ids.len(), 6);
    }

    #[test]
    fn pack_sequences_two_bins_when_overflow() {
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 6).unwrap();
        assert!(packed.len() >= 2, "should produce at least 2 bins");
        for bin in &packed {
            assert!(
                bin.input_ids.len() <= 6,
                "bin length {} exceeds max_length 6",
                bin.input_ids.len()
            );
        }
    }

    #[test]
    fn pack_sequences_all_sequences_present() {
        let seqs: Vec<Vec<u32>> = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![6, 7, 8, 9],
            vec![10],
            vec![11, 12],
        ];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 8).unwrap();

        let mut all_packed_tokens: Vec<u32> = packed
            .iter()
            .flat_map(|b| {
                b.input_ids
                    .iter()
                    .copied()
                    .zip(b.attention_mask.iter().copied())
                    .filter_map(|(t, m)| if m != 0 { Some(t) } else { None })
            })
            .collect();
        all_packed_tokens.sort_unstable();

        let mut all_input_tokens: Vec<u32> = seqs.iter().flatten().copied().collect();
        all_input_tokens.sort_unstable();

        assert_eq!(
            all_packed_tokens, all_input_tokens,
            "all input tokens must appear exactly once in output"
        );
    }

    #[test]
    fn pack_sequences_sequence_exceeds_max_length_returns_error() {
        let long_seq = vec![1u32; 100];
        let slices = vec![long_seq.as_slice()];
        let result = pack_sequences(&slices, 50);
        assert!(result.is_err(), "sequence longer than max_length must return Err");
    }

    #[test]
    fn pack_sequences_empty_input_returns_empty() {
        let result = pack_sequences(&[], 512).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn pack_sequences_single_sequence() {
        let seq = vec![10u32, 20, 30];
        let slices = vec![seq.as_slice()];
        let packed = pack_sequences(&slices, 512).unwrap();
        assert_eq!(packed.len(), 1);
        assert_eq!(&packed[0].input_ids[..3], &[10, 20, 30]);
    }

    #[test]
    fn pack_sequences_attention_mask_matches_input_ids_length() {
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 8).unwrap();
        for bin in &packed {
            assert_eq!(
                bin.input_ids.len(),
                bin.attention_mask.len(),
                "attention_mask must have same length as input_ids"
            );
        }
    }

    #[test]
    fn pack_sequences_position_ids_per_sequence_start_from_zero() {
        let seqs: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5]];
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = pack_sequences(&slices, 8).unwrap();

        for bin in &packed {
            for (k, &start) in bin.sequence_starts.iter().enumerate() {
                let end = if k + 1 < bin.sequence_starts.len() {
                    bin.sequence_starts[k + 1]
                } else {
                    bin.input_ids.len()
                };
                for (j, pos) in (start..end).enumerate() {
                    assert_eq!(
                        bin.position_ids[pos], j as u32,
                        "position_ids[{pos}] should be {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn pack_sequences_fill_rate_good_for_varied_lengths() {
        let lengths = [10, 50, 20, 80, 30, 60, 15, 100, 40, 25];
        let seqs: Vec<Vec<u32>> = lengths
            .iter()
            .enumerate()
            .map(|(i, &l)| (0..l as u32).map(|t| i as u32 * 1000 + t).collect())
            .collect();
        let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let max_len = 128;
        let packed = pack_sequences(&slices, max_len).unwrap();

        let total_tokens: usize = lengths.iter().sum();
        let total_capacity: usize = packed.iter().map(|b| b.input_ids.len()).sum();
        let fill_rate = total_tokens as f64 / total_capacity as f64;

        assert!(
            fill_rate > 0.5,
            "fill_rate {fill_rate:.2} is below 0.5 — bin packing too inefficient"
        );
    }

    mod proptests {
        use super::*;
        use proptest::collection::vec;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn pack_sequences_no_bin_exceeds_max_length(
                lengths in vec(1usize..100, 1..20),
                max_len in 10usize..200,
            ) {
                let seqs: Vec<Vec<u32>> = lengths
                    .iter()
                    .filter(|&&l| l <= max_len)
                    .map(|&l| (0..l as u32).collect())
                    .collect();
                if seqs.is_empty() {
                    return Ok(());
                }
                let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
                let packed = pack_sequences(&slices, max_len).unwrap();
                for bin in &packed {
                    prop_assert!(
                        bin.input_ids.len() <= max_len,
                        "bin length {} exceeds max_len {}",
                        bin.input_ids.len(),
                        max_len
                    );
                }
            }

            #[test]
            fn pack_sequences_all_tokens_present(
                lengths in vec(1usize..50, 1..15),
            ) {
                let max_len = 128;
                let seqs: Vec<Vec<u32>> = lengths
                    .iter()
                    .filter(|&&l| l <= max_len)
                    .enumerate()
                    .map(|(i, &l)| (0..l as u32).map(|t| i as u32 * 1000 + t).collect())
                    .collect();
                if seqs.is_empty() {
                    return Ok(());
                }
                let slices: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
                let packed = pack_sequences(&slices, max_len).unwrap();

                let mut packed_tokens: Vec<u32> = packed
                    .iter()
                    .flat_map(|b| {
                        b.input_ids
                            .iter()
                            .copied()
                            .zip(b.attention_mask.iter().copied())
                            .filter_map(|(t, m)| if m != 0 { Some(t) } else { None })
                    })
                    .collect();
                packed_tokens.sort_unstable();

                let mut input_tokens: Vec<u32> = seqs.iter().flatten().copied().collect();
                input_tokens.sort_unstable();

                prop_assert_eq!(packed_tokens, input_tokens);
            }
        }
    }
}
