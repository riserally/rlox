use crate::error::RloxError;

/// Handle for accessing a named extra column. O(1) via Vec index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnHandle(usize);

impl ColumnHandle {
    /// The raw index used for Vec-based O(1) lookups.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Storage for user-defined extra columns on a replay buffer.
///
/// Each column is a flat `Vec<f32>` with a fixed dimensionality, stored
/// contiguously for cache-friendly sampling. When no columns are registered,
/// this struct has zero overhead — no allocations occur.
pub struct ExtraColumns {
    names: Vec<String>,
    dims: Vec<usize>,
    data: Vec<Vec<f32>>,
    capacity: usize,
}

impl ExtraColumns {
    /// Create empty extra-column storage. No allocations until `register()`.
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            dims: Vec::new(),
            data: Vec::new(),
            capacity: 0,
        }
    }

    /// Register a new column. Returns a handle for O(1) access.
    ///
    /// Must be called before any data is pushed. If storage has already been
    /// allocated (via `allocate()`), the new column is pre-allocated too.
    pub fn register(&mut self, name: &str, dim: usize) -> ColumnHandle {
        let handle = ColumnHandle(self.names.len());
        self.names.push(name.to_owned());
        self.dims.push(dim);
        let col = if self.capacity > 0 {
            vec![0.0; self.capacity * dim]
        } else {
            Vec::new()
        };
        self.data.push(col);
        handle
    }

    /// Number of registered columns.
    pub fn num_columns(&self) -> usize {
        self.names.len()
    }

    /// Get column name and dim by handle.
    pub fn column_info(&self, handle: ColumnHandle) -> (&str, usize) {
        (&self.names[handle.0], self.dims[handle.0])
    }

    /// Pre-allocate storage for a given capacity.
    pub fn allocate(&mut self, capacity: usize) {
        self.capacity = capacity;
        for (col, &dim) in self.data.iter_mut().zip(self.dims.iter()) {
            col.resize(capacity * dim, 0.0);
        }
    }

    /// Write values for one column at a given buffer position.
    ///
    /// The `values` slice must have length equal to the column's dimensionality.
    pub fn push(
        &mut self,
        handle: ColumnHandle,
        pos: usize,
        values: &[f32],
    ) -> Result<(), RloxError> {
        let dim = self.dims[handle.0];
        if values.len() != dim {
            return Err(RloxError::ShapeMismatch {
                expected: format!("extra column '{}' dim={}", self.names[handle.0], dim),
                got: format!("values.len()={}", values.len()),
            });
        }
        let col = &mut self.data[handle.0];
        let start = pos * dim;
        if start + dim > col.len() {
            return Err(RloxError::BufferError(format!(
                "extra column position {} out of bounds (allocated for {})",
                pos,
                col.len() / dim
            )));
        }
        col[start..start + dim].copy_from_slice(values);
        Ok(())
    }

    /// Gather values for one column at the given sampled indices.
    ///
    /// Returns a flat `Vec<f32>` of length `indices.len() * dim`.
    pub fn sample(&self, handle: ColumnHandle, indices: &[usize]) -> Vec<f32> {
        let dim = self.dims[handle.0];
        let col = &self.data[handle.0];
        let mut out = Vec::with_capacity(indices.len() * dim);
        for &idx in indices {
            let start = idx * dim;
            out.extend_from_slice(&col[start..start + dim]);
        }
        out
    }

    /// Gather all columns at the given sampled indices.
    ///
    /// Returns `(column_name, flat_data)` pairs. Only called when
    /// `num_columns() > 0`.
    pub fn sample_all(&self, indices: &[usize]) -> Vec<(String, Vec<f32>)> {
        self.names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let handle = ColumnHandle(i);
                (name.clone(), self.sample(handle, indices))
            })
            .collect()
    }

    /// Clear all data (keep column registrations and capacity).
    pub fn clear(&mut self) {
        for col in &mut self.data {
            col.fill(0.0);
        }
    }
}

impl Default for ExtraColumns {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extra_columns_register_and_push() {
        let mut ec = ExtraColumns::new();
        let h = ec.register("log_prob", 1);
        ec.allocate(10);

        ec.push(h, 0, &[0.5]).unwrap();
        ec.push(h, 1, &[-0.3]).unwrap();

        assert_eq!(ec.num_columns(), 1);
        let (name, dim) = ec.column_info(h);
        assert_eq!(name, "log_prob");
        assert_eq!(dim, 1);
    }

    #[test]
    fn test_extra_columns_sample_roundtrip() {
        let mut ec = ExtraColumns::new();
        let h = ec.register("values", 2);
        ec.allocate(5);

        for i in 0..5 {
            let v = i as f32;
            ec.push(h, i, &[v, v * 10.0]).unwrap();
        }

        let sampled = ec.sample(h, &[0, 2, 4]);
        assert_eq!(sampled, vec![0.0, 0.0, 2.0, 20.0, 4.0, 40.0]);
    }

    #[test]
    fn test_extra_columns_zero_overhead_when_empty() {
        let ec = ExtraColumns::new();
        assert_eq!(ec.num_columns(), 0);
        // No allocations at all
        assert!(ec.names.is_empty());
        assert!(ec.dims.is_empty());
        assert!(ec.data.is_empty());
    }

    #[test]
    fn test_extra_columns_multiple_columns() {
        let mut ec = ExtraColumns::new();
        let h1 = ec.register("log_prob", 1);
        let h2 = ec.register("action_mean", 3);
        ec.allocate(4);

        ec.push(h1, 0, &[0.1]).unwrap();
        ec.push(h2, 0, &[1.0, 2.0, 3.0]).unwrap();
        ec.push(h1, 1, &[0.2]).unwrap();
        ec.push(h2, 1, &[4.0, 5.0, 6.0]).unwrap();

        let s1 = ec.sample(h1, &[0, 1]);
        assert_eq!(s1, vec![0.1, 0.2]);

        let s2 = ec.sample(h2, &[0, 1]);
        assert_eq!(s2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_extra_columns_dim_mismatch_errors() {
        let mut ec = ExtraColumns::new();
        let h = ec.register("test", 2);
        ec.allocate(4);

        let result = ec.push(h, 0, &[1.0]); // dim 1 but column expects 2
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dim=2"),
            "error should mention dim, got: {err}"
        );

        let result = ec.push(h, 0, &[1.0, 2.0, 3.0]); // dim 3 but expects 2
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_columns_out_of_bounds_errors() {
        let mut ec = ExtraColumns::new();
        let h = ec.register("test", 1);
        ec.allocate(2);

        ec.push(h, 0, &[1.0]).unwrap();
        ec.push(h, 1, &[2.0]).unwrap();
        let result = ec.push(h, 2, &[3.0]); // pos 2 but capacity is 2
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_columns_sample_all() {
        let mut ec = ExtraColumns::new();
        let h1 = ec.register("alpha", 1);
        let h2 = ec.register("beta", 2);
        ec.allocate(3);

        ec.push(h1, 0, &[0.1]).unwrap();
        ec.push(h1, 1, &[0.2]).unwrap();
        ec.push(h2, 0, &[1.0, 2.0]).unwrap();
        ec.push(h2, 1, &[3.0, 4.0]).unwrap();

        let all = ec.sample_all(&[0, 1]);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].0, "alpha");
        assert_eq!(all[0].1, vec![0.1, 0.2]);
        assert_eq!(all[1].0, "beta");
        assert_eq!(all[1].1, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_extra_columns_clear_preserves_registrations() {
        let mut ec = ExtraColumns::new();
        let h = ec.register("test", 1);
        ec.allocate(3);
        ec.push(h, 0, &[42.0]).unwrap();

        ec.clear();
        assert_eq!(ec.num_columns(), 1);
        // Data should be zeroed
        let sampled = ec.sample(h, &[0]);
        assert_eq!(sampled, vec![0.0]);
    }
}
