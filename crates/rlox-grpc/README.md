# rlox-grpc

gRPC distributed environment workers for rlox.

Enables remote environment execution over the network using Tonic gRPC, allowing environment stepping to be distributed across multiple machines.

## Key Types

- `EnvWorker` -- gRPC server that runs an rlox environment and serves `step`/`reset` RPCs
- `RemoteEnvClient` -- async client that connects to a remote `EnvWorker`
- `GrpcError` -- error type for gRPC operations
- `proto` -- generated protobuf types for the `rlox.env` service

## Usage

### Server (environment worker)

```rust
use rlox_grpc::EnvWorker;

// Start a gRPC environment worker on port 50051
// let worker = EnvWorker::new("CartPole-v1");
// worker.serve("[::1]:50051").await?;
```

### Client (training process)

```rust
use rlox_grpc::RemoteEnvClient;

// Connect to a remote environment worker
// let mut client = RemoteEnvClient::connect("http://[::1]:50051").await?;
// let obs = client.reset(None).await?;
// let transition = client.step(&action).await?;
```

## Dependencies

- `tonic` 0.13 / `prost` 0.13 for gRPC
- `tokio` for async runtime
- `rlox-core` for environment types

## Part of rlox

This crate enables distributed training for [rlox](https://github.com/riserally/rlox). See the main project for algorithms, Python bindings, and documentation.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE).
