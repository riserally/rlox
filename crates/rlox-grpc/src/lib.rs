#![allow(clippy::result_large_err)]
//! rlox-grpc: gRPC distributed environment workers for rlox.
//!
//! Enables remote environment execution over the network using Tonic gRPC.
//! Environment stepping can be distributed across multiple machines for
//! large-scale training.
//!
//! Key types:
//! - [`server::EnvWorker`]: gRPC server running an rlox environment.
//! - [`client::RemoteEnvClient`]: Async client connecting to remote workers.
//! - [`proto`]: Generated protobuf types for the `rlox.env` service.

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
