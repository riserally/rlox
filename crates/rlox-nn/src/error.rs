use thiserror::Error;

#[derive(Debug, Error)]
pub enum NNError {
    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Device error: {0}")]
    Device(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Not supported: {0}")]
    NotSupported(String),
}
