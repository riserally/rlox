use burn::prelude::*;
use rlox_nn::TensorData;

/// Convert TensorData to a Burn Tensor<B, 1>.
pub fn to_tensor_1d<B: Backend>(data: &TensorData, device: &B::Device) -> Tensor<B, 1> {
    let td = burn::tensor::TensorData::new(data.data.clone(), burn::tensor::Shape::new([data.data.len()]));
    Tensor::from_data(td, device)
}

/// Convert TensorData to a Burn Tensor<B, 2>.
pub fn to_tensor_2d<B: Backend>(data: &TensorData, device: &B::Device) -> Tensor<B, 2> {
    assert_eq!(data.shape.len(), 2, "expected 2D shape, got {:?}", data.shape);
    let td = burn::tensor::TensorData::new(
        data.data.clone(),
        burn::tensor::Shape::new([data.shape[0], data.shape[1]]),
    );
    Tensor::from_data(td, device)
}

/// Convert a Burn Tensor<B, 1> to TensorData.
pub fn from_tensor_1d<B: Backend>(tensor: Tensor<B, 1>) -> TensorData {
    let td = tensor.into_data();
    let data: Vec<f32> = td.to_vec().expect("f32 conversion");
    let len = data.len();
    TensorData::new(data, vec![len])
}

/// Convert a Burn Tensor<B, 2> to TensorData.
pub fn from_tensor_2d<B: Backend>(tensor: Tensor<B, 2>) -> TensorData {
    let shape = tensor.shape().dims;
    let td = tensor.into_data();
    let data: Vec<f32> = td.to_vec().expect("f32 conversion");
    TensorData::new(data, vec![shape[0], shape[1]])
}

/// Convert TensorData to a Burn Int Tensor<B, 1> (for action indices).
pub fn to_int_tensor_1d<B: Backend>(data: &TensorData, device: &B::Device) -> Tensor<B, 1, Int> {
    let ints: Vec<i32> = data.data.iter().map(|&x| x as i32).collect();
    let td = burn::tensor::TensorData::new(ints, burn::tensor::Shape::new([data.data.len()]));
    Tensor::from_data(td, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;
    type TestDevice = <TestBackend as Backend>::Device;

    fn device() -> TestDevice {
        Default::default()
    }

    #[test]
    fn test_roundtrip_1d() {
        let original = TensorData::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor: Tensor<TestBackend, 1> = to_tensor_1d(&original, &device());
        let result = from_tensor_1d(tensor);
        assert_eq!(result.data, original.data);
        assert_eq!(result.shape, vec![3]);
    }

    #[test]
    fn test_roundtrip_2d() {
        let original = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor: Tensor<TestBackend, 2> = to_tensor_2d(&original, &device());
        let result = from_tensor_2d(tensor);
        assert_eq!(result.data, original.data);
        assert_eq!(result.shape, vec![2, 3]);
    }

    #[test]
    fn test_int_tensor() {
        let data = TensorData::new(vec![0.0, 1.0, 2.0], vec![3]);
        let _tensor: Tensor<TestBackend, 1, Int> = to_int_tensor_1d(&data, &device());
    }
}
