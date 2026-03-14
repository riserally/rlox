/// Backend-agnostic tensor data container.
///
/// This is the "lingua franca" across the trait boundary between
/// rlox-core (raw buffers/slices) and NN backends. Data is stored
/// as flat f32 in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data length {} must match shape product {:?} = {}",
            data.len(),
            shape,
            shape.iter().product::<usize>()
        );
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![1.0; len],
            shape,
        }
    }

    pub fn from_f64(data: &[f64], shape: Vec<usize>) -> Self {
        Self::new(data.iter().map(|&x| x as f32).collect(), shape)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = TensorData::zeros(vec![2, 3]);
        assert_eq!(t.data.len(), 6);
        assert!(t.data.iter().all(|&x| x == 0.0));
        assert_eq!(t.shape, vec![2, 3]);
    }

    #[test]
    fn test_ones() {
        let t = TensorData::ones(vec![4]);
        assert_eq!(t.data.len(), 4);
        assert!(t.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_f64() {
        let vals = vec![1.0_f64, 2.0, 3.0];
        let t = TensorData::from_f64(&vals, vec![3]);
        assert_eq!(t.data, vec![1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn test_numel_ndim() {
        let t = TensorData::zeros(vec![2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.ndim(), 3);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_empty() {
        let t = TensorData::zeros(vec![0]);
        assert!(t.is_empty());
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_shape_mismatch_panics_in_debug() {
        TensorData::new(vec![1.0, 2.0], vec![3]);
    }
}
