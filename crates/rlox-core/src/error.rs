use thiserror::Error;

#[derive(Debug, Error)]
pub enum RloxError {
    #[error("Invalid action: {0}")]
    InvalidAction(String),

    #[error("Environment error: {0}")]
    EnvError(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Buffer error: {0}")]
    BufferError(String),
}
