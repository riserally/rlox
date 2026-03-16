pub mod client;
pub mod error;
pub mod server;

/// Generated protobuf / gRPC types for the environment service.
pub mod proto {
    tonic::include_proto!("rlox.env");
}

// Re-exports for convenience.
pub use client::RemoteEnvClient;
pub use error::GrpcError;
pub use server::EnvWorker;
