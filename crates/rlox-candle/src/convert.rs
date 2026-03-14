use candle_core::{Device, Tensor};
use rlox_nn::TensorData;

pub fn to_tensor_1d(data: &TensorData, device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(data.data.clone(), data.data.len(), device)
}

pub fn to_tensor_2d(data: &TensorData, device: &Device) -> candle_core::Result<Tensor> {
    assert_eq!(data.shape.len(), 2);
    Tensor::from_vec(data.data.clone(), (data.shape[0], data.shape[1]), device)
}

pub fn from_tensor_1d(tensor: &Tensor) -> candle_core::Result<TensorData> {
    let data: Vec<f32> = tensor.to_vec1()?;
    let len = data.len();
    Ok(TensorData::new(data, vec![len]))
}

pub fn from_tensor_2d(tensor: &Tensor) -> candle_core::Result<TensorData> {
    let dims = tensor.dims();
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    Ok(TensorData::new(data, vec![dims[0], dims[1]]))
}

pub fn to_int_tensor_1d(data: &TensorData, device: &Device) -> candle_core::Result<Tensor> {
    let ints: Vec<u32> = data.data.iter().map(|&x| x as u32).collect();
    let len = ints.len();
    Tensor::from_vec(ints, len, device)
}

fn candle_err(e: candle_core::Error) -> rlox_nn::NNError {
    rlox_nn::NNError::Backend(e.to_string())
}

pub trait IntoNNError<T> {
    fn nn_err(self) -> Result<T, rlox_nn::NNError>;
}

impl<T> IntoNNError<T> for candle_core::Result<T> {
    fn nn_err(self) -> Result<T, rlox_nn::NNError> {
        self.map_err(candle_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_1d() {
        let original = TensorData::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor = to_tensor_1d(&original, &Device::Cpu).unwrap();
        let result = from_tensor_1d(&tensor).unwrap();
        assert_eq!(result.data, original.data);
    }

    #[test]
    fn test_roundtrip_2d() {
        let original = TensorData::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let tensor = to_tensor_2d(&original, &Device::Cpu).unwrap();
        let result = from_tensor_2d(&tensor).unwrap();
        assert_eq!(result.data, original.data);
        assert_eq!(result.shape, vec![2, 2]);
    }
}
