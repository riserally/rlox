use thiserror::Error;

#[derive(Debug, Error)]
pub enum GrpcError {
    #[error("gRPC transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("gRPC status: {0}")]
    Status(#[from] tonic::Status),

    #[error("mixed action types in batch (all must be Discrete or all Continuous)")]
    MixedActionTypes,
}
