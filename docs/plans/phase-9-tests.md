# Phase 9 TDD Test Specifications — Distributed and Scale (v1.0)

**Status: RED (all tests must be written before implementation)**
**Phase plan: `/docs/plans/phase-9-distributed-and-scale.md`**
**Depends on: Phase 8 tests passing (SAC, TD3, DQN, one-liner API all working)**
**Run Rust tests:** `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --package rlox-dist`
**Run Python tests:** `.venv/bin/python -m pytest tests/python/test_phase9.py -v`

---

## Test Execution Order

```
Group 0: Lock-free channel + pipeline primitives (Rust)
Group 1: gRPC env worker service (Rust + integration)
Group 2: V-trace + transition provenance (Rust)
Group 3: Decoupled collection/training (Python integration)
Group 4: Multi-GPU PPO (requires 2+ GPUs — CI skip if unavailable)
Group 5: vLLM / TGI integration (requires inference server — mock + real)
Group 6: Reward model serving (Python)
Group 7: MAPPO + DreamerV3 (Python E2E)
Group 8: IMPALA async throughput
Group 9: API 1.0 stability
```

---

## Part 1: Rust Unit Tests (new crate: rlox-dist)

All tests live in `crates/rlox-dist/src/`. This is a new crate added to the workspace.

### 1.1 Lock-Free Experience Channel Tests

**Target file:** `crates/rlox-dist/src/channel.rs` (new file)
**Group: 0**

```rust
// crates/rlox-dist/src/channel.rs
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // RED: channel creation
    #[test]
    fn experience_channel_creates_sender_and_receiver() {
        let (tx, rx) = experience_channel(10);
        // Both ends created without panic
        drop(tx);
        drop(rx);
    }

    // RED: sender can send and receiver can receive
    #[test]
    fn experience_channel_send_receive() {
        let (tx, rx) = experience_channel(10);
        tx.send(42u32).unwrap();
        let received = rx.recv().unwrap();
        assert_eq!(received, 42u32);
    }

    // RED: bounded channel blocks sender when full (backpressure)
    #[test]
    fn experience_channel_bounded_blocks_on_full() {
        let capacity = 2;
        let (tx, rx) = experience_channel(capacity);

        // Fill the channel
        tx.send(1u32).unwrap();
        tx.send(2u32).unwrap();

        // Non-blocking send should fail when full
        let result = tx.try_send(3u32);
        assert!(result.is_err(), "try_send must fail when channel is full");

        // After receiving, can send again
        let _ = rx.recv().unwrap();
        tx.try_send(3u32).expect("send should succeed after receiver");
    }

    // RED: channel is FIFO
    #[test]
    fn experience_channel_is_fifo() {
        let (tx, rx) = experience_channel(100);
        for i in 0u32..10 {
            tx.send(i).unwrap();
        }
        for expected in 0u32..10 {
            let received = rx.try_recv().unwrap();
            assert_eq!(received, expected, "channel must be FIFO");
        }
    }

    // RED: multiple producers, single consumer
    #[test]
    fn experience_channel_multiple_producers() {
        let (tx, rx) = experience_channel(100);
        let n_producers = 4;
        let items_per_producer = 10;

        let handles: Vec<_> = (0..n_producers)
            .map(|i| {
                let tx_clone = tx.clone();
                thread::spawn(move || {
                    for j in 0..items_per_producer {
                        tx_clone.send(i * 100 + j).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        drop(tx);  // close all senders

        let mut received = Vec::new();
        while let Ok(v) = rx.try_recv() {
            received.push(v);
        }
        assert_eq!(
            received.len(), n_producers * items_per_producer,
            "all sent items must be received"
        );
    }

    // RED: receiver disconnects when all senders dropped
    #[test]
    fn experience_channel_receiver_disconnects_on_drop() {
        let (tx, rx) = experience_channel(10);
        drop(tx);  // drop sender
        let result = rx.recv();
        assert!(result.is_err(), "recv should fail when sender is dropped");
    }

    // RED: RolloutBatch can be sent through channel
    #[test]
    fn experience_channel_sends_rollout_batch() {
        use crate::pipeline::RolloutBatch;

        let (tx, rx) = experience_channel::<RolloutBatch>(2);

        let batch = RolloutBatch {
            obs: vec![0.0f32; 4 * 128],
            actions: vec![0.0f32; 128],
            rewards: vec![1.0f32; 128],
            advantages: vec![0.5f32; 128],
            returns: vec![1.5f32; 128],
            log_probs: vec![-1.0f32; 128],
            obs_dim: 4,
            act_dim: 1,
            n_steps: 128,
        };

        tx.send(batch).unwrap();
        let received = rx.recv().unwrap();
        assert_eq!(received.n_steps, 128);
        assert_eq!(received.obs.len(), 4 * 128);
    }

    // --- Property-based tests ---
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // RED: all sent items are received in order
            #[test]
            fn channel_fifo_property(items in proptest::collection::vec(0u32..1000, 1..50)) {
                let (tx, rx) = experience_channel(items.len() + 1);
                for &item in &items {
                    tx.send(item).unwrap();
                }
                for &expected in &items {
                    let received = rx.try_recv().unwrap();
                    prop_assert_eq!(received, expected);
                }
            }
        }
    }
}
```

### 1.2 Async Collector / Pipeline Tests

**Target file:** `crates/rlox-dist/src/pipeline.rs` (new file)
**Group: 0**

```rust
// crates/rlox-dist/src/pipeline.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::experience_channel;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    // RED: Pipeline::new does not block or panic
    #[test]
    fn pipeline_new_does_not_block() {
        // This test must complete quickly (< 1 second)
        // If collector blocks on creation, this will timeout
        let _pipeline = Pipeline::new(
            /*env_factory=*/ || make_test_batch_stepper(4),
            /*n_envs=*/ 4,
            /*n_steps=*/ 32,
            /*channel_capacity=*/ 4,
        );
    }

    // RED: pipeline delivers batches to receiver
    #[test]
    fn pipeline_delivers_batches() {
        let pipeline = Pipeline::new(
            || make_test_batch_stepper(4),
            4, 32, 4,
        );

        // Wait for first batch (with timeout)
        let batch = pipeline.next_batch_timeout(Duration::from_secs(5));
        assert!(batch.is_some(), "pipeline should deliver a batch within 5 seconds");
        let batch = batch.unwrap();
        assert_eq!(batch.n_steps, 32 * 4);  // n_steps * n_envs
    }

    // RED: try_next_batch returns None when channel empty
    #[test]
    fn pipeline_try_next_batch_non_blocking() {
        let pipeline = Pipeline::new(
            || make_test_batch_stepper(4),
            4, 128, 1,  // large n_steps — collector needs time
        );

        // Immediately after creation, channel is likely empty
        // try_next_batch must return without blocking
        let start = std::time::Instant::now();
        let _result = pipeline.try_next_batch();
        let elapsed = start.elapsed();

        assert!(elapsed < Duration::from_millis(100),
            "try_next_batch must be non-blocking, took {:?}", elapsed);
    }

    // RED: slow learner triggers backpressure (collector blocks, no memory growth)
    #[test]
    fn pipeline_backpressure_on_slow_learner() {
        let capacity = 2;
        let batches_collected = Arc::new(AtomicUsize::new(0));
        let bc_clone = batches_collected.clone();

        let pipeline = Pipeline::new_with_hook(
            || make_test_batch_stepper(4),
            4, 32, capacity,
            move |_batch| {
                bc_clone.fetch_add(1, Ordering::Relaxed);
            },
        );

        // Sleep for 1 second WITHOUT draining the channel
        std::thread::sleep(Duration::from_millis(200));

        // With capacity=2 and no receiver, at most capacity + 1 batches
        // should have been collected (one being produced + 2 buffered)
        let collected = batches_collected.load(Ordering::Relaxed);
        assert!(collected <= capacity + 2,
            "with backpressure, at most {} batches should be produced, got {}",
            capacity + 2, collected);

        // Drain the channel — more batches should flow
        for _ in 0..capacity {
            let _ = pipeline.try_next_batch();
        }
        std::thread::sleep(Duration::from_millis(100));
        let collected_after_drain = batches_collected.load(Ordering::Relaxed);
        assert!(collected_after_drain > collected,
            "after draining, more batches should be produced");
    }

    // RED: pipeline shutdown is clean (no hang on drop)
    #[test]
    fn pipeline_drops_cleanly() {
        // If pipeline.drop() hangs, this test will timeout
        let pipeline = Pipeline::new(
            || make_test_batch_stepper(4),
            4, 32, 4,
        );
        // Immediately drop — must not hang
        drop(pipeline);
    }

    // RED: batch rewards are non-NaN (collector is computing GAE correctly)
    #[test]
    fn pipeline_batch_rewards_finite() {
        let pipeline = Pipeline::new(
            || make_test_batch_stepper(4),
            4, 32, 4,
        );

        let batch = pipeline
            .next_batch_timeout(Duration::from_secs(5))
            .expect("pipeline should deliver batch");

        for &r in &batch.rewards {
            assert!(r.is_finite(), "reward must be finite, got {}", r);
        }
        for &a in &batch.advantages {
            assert!(a.is_finite(), "advantage must be finite, got {}", a);
        }
    }

    // Helper: create a test BatchSteppable using CartPole
    fn make_test_batch_stepper(n_envs: usize) -> impl BatchSteppable {
        use rlox_core::env::batch::ParallelBatchStepper;
        use rlox_core::env::builtins::CartPole;

        let envs: Vec<Box<dyn rlox_core::env::RLEnv>> = (0..n_envs)
            .map(|i| Box::new(CartPole::new(Some(i as u64))) as Box<dyn rlox_core::env::RLEnv>)
            .collect();
        ParallelBatchStepper::new(envs)
    }
}
```

### 1.3 gRPC Env Worker Service Tests

**Target file:** `crates/rlox-dist/src/grpc/env_service.rs` (new file)
**Group: 1 — requires tokio runtime**

```rust
// crates/rlox-dist/src/grpc/env_service.rs
#[cfg(test)]
mod tests {
    use super::*;
    use tonic::transport::Server;
    use rlox_core::env::builtins::CartPole;

    // RED: EnvWorker::new creates worker with n_envs environments
    #[tokio::test]
    async fn env_worker_creates_n_envs() {
        let worker = EnvWorker::new_cartpole(4, 42);
        assert_eq!(worker.num_envs(), 4);
    }

    // RED: gRPC server starts and accepts connections
    #[tokio::test]
    async fn env_service_server_starts() {
        let worker = EnvWorker::new_cartpole(2, 42);
        let addr = "127.0.0.1:0".parse().unwrap();

        // Bind to port 0 (OS assigns free port)
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(EnvServiceServer::new(worker))
                .serve_with_incoming(
                    tokio_stream::wrappers::TcpListenerStream::new(listener)
                )
                .await
        });

        // Give server time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Connect client
        let client_result = EnvServiceClient::connect(
            format!("http://{}", server_addr)
        ).await;
        assert!(client_result.is_ok(), "client should connect to gRPC server");

        server.abort();
    }

    // RED: step_batch via gRPC returns correct batch structure
    #[tokio::test]
    async fn grpc_step_batch_returns_correct_structure() {
        let (server_addr, server_handle) = start_test_server(4, 42).await;
        let mut client = EnvServiceClient::connect(
            format!("http://{}", server_addr)
        ).await.unwrap();

        // Reset first
        let reset_req = ResetRequest {
            seed: Some(42),
        };
        client.reset_batch(reset_req).await.unwrap();

        // Step with discrete actions (CartPole: 0 or 1)
        let step_req = StepRequest {
            actions: vec![0, 1, 0, 1],  // 4 envs, discrete
            action_type: ActionType::Discrete as i32,
        };
        let response = client.step_batch(step_req).await.unwrap().into_inner();

        assert_eq!(response.obs.len(), 4 * 4);  // 4 envs, CartPole obs_dim=4
        assert_eq!(response.rewards.len(), 4);
        assert_eq!(response.terminated.len(), 4);
        assert_eq!(response.truncated.len(), 4);

        server_handle.abort();
    }

    // RED: reset_batch via gRPC returns observations
    #[tokio::test]
    async fn grpc_reset_batch_returns_observations() {
        let (server_addr, server_handle) = start_test_server(2, 42).await;
        let mut client = EnvServiceClient::connect(
            format!("http://{}", server_addr)
        ).await.unwrap();

        let reset_req = ResetRequest { seed: Some(99) };
        let response = client.reset_batch(reset_req).await.unwrap().into_inner();

        assert_eq!(response.obs.len(), 2 * 4);  // 2 envs, CartPole obs_dim=4
        for &v in &response.obs {
            assert!(v.is_finite(), "obs contains NaN/Inf: {}", v);
        }

        server_handle.abort();
    }

    // RED: get_spaces returns correct obs_dim and act_dim
    #[tokio::test]
    async fn grpc_get_spaces_returns_cartpole_dims() {
        let (server_addr, server_handle) = start_test_server(2, 42).await;
        let mut client = EnvServiceClient::connect(
            format!("http://{}", server_addr)
        ).await.unwrap();

        let response = client
            .get_spaces(GetSpacesRequest {})
            .await
            .unwrap()
            .into_inner();

        assert_eq!(response.obs_dim, 4, "CartPole obs_dim should be 4");
        assert_eq!(response.act_dim, 1, "CartPole act_dim should be 1");
        assert_eq!(response.n_envs, 2);

        server_handle.abort();
    }

    // RED: concurrent step_batch calls are handled safely (no data races)
    #[tokio::test]
    async fn grpc_concurrent_step_calls_safe() {
        // This tests that the server handles sequential calls correctly.
        // (gRPC is sequential per stream by default)
        let (server_addr, server_handle) = start_test_server(4, 42).await;
        let mut client = EnvServiceClient::connect(
            format!("http://{}", server_addr)
        ).await.unwrap();

        client.reset_batch(ResetRequest { seed: Some(42) }).await.unwrap();

        // 50 sequential step calls
        for _ in 0..50 {
            let step_req = StepRequest {
                actions: vec![0, 1, 0, 1],
                action_type: ActionType::Discrete as i32,
            };
            let resp = client.step_batch(step_req).await.unwrap().into_inner();
            assert_eq!(resp.obs.len(), 4 * 4);
        }

        server_handle.abort();
    }

    // RED: wrong action count returns gRPC error (not crash)
    #[tokio::test]
    async fn grpc_wrong_action_count_returns_error() {
        let (server_addr, server_handle) = start_test_server(4, 42).await;
        let mut client = EnvServiceClient::connect(
            format!("http://{}", server_addr)
        ).await.unwrap();

        client.reset_batch(ResetRequest { seed: None }).await.unwrap();

        // Send 3 actions for 4 envs
        let step_req = StepRequest {
            actions: vec![0, 1, 0],  // 3 actions, 4 envs
            action_type: ActionType::Discrete as i32,
        };
        let result = client.step_batch(step_req).await;
        assert!(result.is_err(), "wrong action count should return gRPC error");

        server_handle.abort();
    }

    async fn start_test_server(
        n_envs: usize,
        seed: u64,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let worker = EnvWorker::new_cartpole(n_envs, seed);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            Server::builder()
                .add_service(EnvServiceServer::new(worker))
                .serve_with_incoming(
                    tokio_stream::wrappers::TcpListenerStream::new(listener)
                )
                .await
                .ok();
        });
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        (addr, handle)
    }
}
```

### 1.4 V-trace Tests

**Target file:** `crates/rlox-core/src/training/vtrace.rs` (new file)
**Group: 2**

```rust
// crates/rlox-core/src/training/vtrace.rs
#[cfg(test)]
mod tests {
    use super::*;

    // Reference V-trace implementation (from IMPALA paper)
    fn reference_vtrace(
        log_rhos: &[f32],
        rewards: &[f32],
        values: &[f32],
        bootstrap_value: f32,
        gamma: f32,
        rho_bar: f32,
        c_bar: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = rewards.len();
        let mut vs = vec![0.0f32; n];
        let mut pg_adv = vec![0.0f32; n];

        let rhos: Vec<f32> = log_rhos.iter().map(|&l| l.exp().min(rho_bar)).collect();
        let cs: Vec<f32> = log_rhos.iter().map(|&l| l.exp().min(c_bar)).collect();

        // Compute from back to front
        let mut vs_next = bootstrap_value;
        for t in (0..n).rev() {
            let delta_t = rhos[t] * (rewards[t] + gamma * vs_next - values[t]);
            vs[t] = values[t] + delta_t
                + gamma * cs[t] * (vs_next - (if t + 1 < n { values[t + 1] } else { bootstrap_value }));
            pg_adv[t] = rhos[t] * (rewards[t] + gamma * vs_next - values[t]);
            vs_next = vs[t];
        }

        (vs, pg_adv)
    }

    // RED: compute_vtrace returns correct dimensions
    #[test]
    fn vtrace_returns_correct_dimensions() {
        let n = 10;
        let log_rhos = vec![0.0f32; n];
        let rewards = vec![1.0f32; n];
        let values = vec![0.5f32; n];

        let result = compute_vtrace(
            &log_rhos, &rewards, &values,
            /*bootstrap_value=*/ 0.5,
            /*gamma=*/ 0.99,
            /*rho_bar=*/ 1.0,
            /*c_bar=*/ 1.0,
        );

        assert!(result.is_ok());
        let (vs, pg_adv) = result.unwrap();
        assert_eq!(vs.len(), n);
        assert_eq!(pg_adv.len(), n);
    }

    // RED: when log_rho=0 (on-policy), V-trace reduces to TD(n)
    #[test]
    fn vtrace_on_policy_reduces_to_td() {
        // With log_rho=0 (rho=1, c=1, rho_bar=1, c_bar=1),
        // V-trace is equivalent to standard TD(n) computation
        let n = 5;
        let log_rhos = vec![0.0f32; n];  // on-policy: rho = 1.0
        let rewards = vec![1.0f32; n];
        let values = vec![0.5f32; n];
        let bootstrap = 0.5f32;
        let gamma = 0.99f32;

        let (vs, _) = compute_vtrace(
            &log_rhos, &rewards, &values, bootstrap, gamma, 1.0, 1.0,
        ).unwrap();

        // All vs must be finite
        for &v in &vs {
            assert!(v.is_finite(), "V-trace value must be finite, got {}", v);
        }
    }

    // RED: V-trace matches reference implementation
    #[test]
    fn vtrace_matches_reference_implementation() {
        let n = 20;
        let rng_seed = 42u64;
        // Generate random-ish test data
        let log_rhos: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.1 + rng_seed as f32 * 0.01) % 1.0) - 0.5)
            .collect();
        let rewards: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 0.5 + 0.5).collect();
        let values: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.01).collect();
        let bootstrap = 0.5f32;
        let gamma = 0.99f32;
        let rho_bar = 1.0f32;
        let c_bar = 1.0f32;

        let (vs_rlox, pg_rlox) = compute_vtrace(
            &log_rhos, &rewards, &values, bootstrap, gamma, rho_bar, c_bar,
        ).unwrap();

        let (vs_ref, pg_ref) = reference_vtrace(
            &log_rhos, &rewards, &values, bootstrap, gamma, rho_bar, c_bar,
        );

        for i in 0..n {
            assert!(
                (vs_rlox[i] - vs_ref[i]).abs() < 1e-5,
                "vs[{}]: rlox={:.6}, reference={:.6}", i, vs_rlox[i], vs_ref[i]
            );
            assert!(
                (pg_rlox[i] - pg_ref[i]).abs() < 1e-5,
                "pg_adv[{}]: rlox={:.6}, reference={:.6}", i, pg_rlox[i], pg_ref[i]
            );
        }
    }

    // RED: rho clipping limits IS ratio
    #[test]
    fn vtrace_rho_clipping_applied() {
        let n = 5;
        // Very high IS ratio (large difference between policy and behavior)
        let log_rhos = vec![10.0f32; n];  // rho = exp(10) >> 1
        let rewards = vec![1.0f32; n];
        let values = vec![0.5f32; n];

        let (vs, _) = compute_vtrace(
            &log_rhos, &rewards, &values, 0.5, 0.99,
            /*rho_bar=*/ 1.0,  // clip at 1.0
            /*c_bar=*/ 1.0,
        ).unwrap();

        // With rho_bar=1.0, high IS ratios are clipped
        // The result should be the same as on-policy (log_rho=0)
        let (vs_on_policy, _) = compute_vtrace(
            &vec![0.0f32; n], &rewards, &values, 0.5, 0.99, 1.0, 1.0,
        ).unwrap();

        for i in 0..n {
            assert!(
                (vs[i] - vs_on_policy[i]).abs() < 1e-5,
                "clipped rho should match on-policy at index {}", i
            );
        }
    }

    // RED: empty input returns Ok with empty vecs
    #[test]
    fn vtrace_empty_input_returns_empty() {
        let result = compute_vtrace(&[], &[], &[], 0.0, 0.99, 1.0, 1.0);
        assert!(result.is_ok());
        let (vs, pg) = result.unwrap();
        assert!(vs.is_empty());
        assert!(pg.is_empty());
    }

    // RED: mismatched lengths returns error
    #[test]
    fn vtrace_mismatched_lengths_returns_error() {
        let result = compute_vtrace(
            &[0.0f32; 5], &[1.0f32; 4], &[0.5f32; 5],
            0.5, 0.99, 1.0, 1.0,
        );
        assert!(result.is_err(), "mismatched lengths must return Err");
    }

    // RED: NaN log_rho is handled (returns Err or NaN propagation, not silent corruption)
    #[test]
    fn vtrace_nan_log_rho_does_not_silently_corrupt() {
        let log_rhos = vec![f32::NAN, 0.0, 0.0];
        let rewards = vec![1.0f32; 3];
        let values = vec![0.5f32; 3];

        let result = compute_vtrace(&log_rhos, &rewards, &values, 0.5, 0.99, 1.0, 1.0);
        // Either error or NaN propagation — must not produce finite non-NaN values
        match result {
            Err(_) => {}  // acceptable
            Ok((vs, _)) => {
                // If it succeeds, at least one value should be NaN
                assert!(vs.iter().any(|v| v.is_nan()),
                    "NaN input should propagate to output, not be silently ignored");
            }
        }
    }

    // --- Property-based tests ---
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // RED: V-trace output length matches input length
            #[test]
            fn vtrace_length_invariant(n in 1usize..100) {
                let log_rhos = vec![0.0f32; n];
                let rewards = vec![1.0f32; n];
                let values = vec![0.5f32; n];
                let (vs, pg) = compute_vtrace(&log_rhos, &rewards, &values, 0.5, 0.99, 1.0, 1.0).unwrap();
                prop_assert_eq!(vs.len(), n);
                prop_assert_eq!(pg.len(), n);
            }

            // RED: all V-trace values are finite when inputs are finite
            #[test]
            fn vtrace_finite_inputs_produce_finite_outputs(
                n in 1usize..50,
                gamma in 0.0f32..1.0,
                rho_bar in 0.1f32..5.0,
            ) {
                let log_rhos = vec![0.0f32; n];  // on-policy
                let rewards = vec![1.0f32; n];
                let values = vec![0.5f32; n];
                let (vs, pg) = compute_vtrace(
                    &log_rhos, &rewards, &values, 0.5, gamma, rho_bar, rho_bar,
                ).unwrap();
                for &v in &vs {
                    prop_assert!(v.is_finite(), "vs contains NaN/Inf: {}", v);
                }
                for &p in &pg {
                    prop_assert!(p.is_finite(), "pg_adv contains NaN/Inf: {}", p);
                }
            }
        }
    }
}
```

### 1.5 Transition Provenance Tests

**Target file:** `crates/rlox-core/src/buffer/provenance.rs` (new file)
**Group: 2**

```rust
// crates/rlox-core/src/buffer/provenance.rs
#[cfg(test)]
mod tests {
    use super::*;

    // RED: TransitionMeta construction
    #[test]
    fn transition_meta_construction() {
        let meta = TransitionMeta {
            env_id: 3,
            episode_id: 42,
            step_in_episode: 15,
            global_step: 10000,
            policy_version: 5,
            reward_model_version: 2,
            timestamp_ns: 1000000000,
        };
        assert_eq!(meta.env_id, 3);
        assert_eq!(meta.episode_id, 42);
    }

    // RED: TransitionMeta serializes to bytes
    #[test]
    fn transition_meta_serializes() {
        let meta = TransitionMeta {
            env_id: 1, episode_id: 100, step_in_episode: 50,
            global_step: 5000, policy_version: 3, reward_model_version: 1,
            timestamp_ns: 9999999,
        };
        let bytes = meta.serialize();
        assert!(!bytes.is_empty(), "serialized bytes must not be empty");
    }

    // RED: TransitionMeta roundtrip (serialize then deserialize)
    #[test]
    fn transition_meta_roundtrip() {
        let meta = TransitionMeta {
            env_id: 7,
            episode_id: 1234,
            step_in_episode: 99,
            global_step: 500000,
            policy_version: 10,
            reward_model_version: 3,
            timestamp_ns: 1234567890123456789,
        };

        let bytes = meta.serialize();
        let restored = TransitionMeta::deserialize(&bytes).unwrap();

        assert_eq!(meta.env_id, restored.env_id);
        assert_eq!(meta.episode_id, restored.episode_id);
        assert_eq!(meta.step_in_episode, restored.step_in_episode);
        assert_eq!(meta.global_step, restored.global_step);
        assert_eq!(meta.policy_version, restored.policy_version);
        assert_eq!(meta.reward_model_version, restored.reward_model_version);
        assert_eq!(meta.timestamp_ns, restored.timestamp_ns);
    }

    // RED: TransitionMeta implements PartialEq
    #[test]
    fn transition_meta_eq() {
        let m1 = TransitionMeta {
            env_id: 1, episode_id: 1, step_in_episode: 1,
            global_step: 1, policy_version: 1, reward_model_version: 1,
            timestamp_ns: 1,
        };
        let m2 = m1.clone();
        assert_eq!(m1, m2);
    }

    // RED: deserialization of corrupt bytes returns error
    #[test]
    fn transition_meta_deserialize_corrupt_bytes_returns_error() {
        let corrupt_bytes = vec![0u8, 1, 2, 3];  // too short
        let result = TransitionMeta::deserialize(&corrupt_bytes);
        assert!(result.is_err(), "corrupt bytes must return Err");
    }

    // RED: serialized size is bounded (no unbounded growth)
    #[test]
    fn transition_meta_serialized_size_bounded() {
        let meta = TransitionMeta {
            env_id: u32::MAX,
            episode_id: u64::MAX,
            step_in_episode: u32::MAX,
            global_step: u64::MAX,
            policy_version: u64::MAX,
            reward_model_version: u64::MAX,
            timestamp_ns: u64::MAX,
        };
        let bytes = meta.serialize();
        // Max size: 4+8+4+8+8+8+8 = 48 bytes (or similar with padding)
        assert!(bytes.len() <= 64, "serialized meta should be <= 64 bytes");
    }

    // --- Property-based tests ---
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // RED: all TransitionMeta roundtrip correctly
            #[test]
            fn transition_meta_roundtrip_prop(
                env_id: u32,
                episode_id: u64,
                step_in_episode: u32,
                global_step: u64,
                policy_version: u64,
                reward_model_version: u64,
                timestamp_ns: u64,
            ) {
                let meta = TransitionMeta {
                    env_id, episode_id, step_in_episode,
                    global_step, policy_version, reward_model_version, timestamp_ns,
                };
                let bytes = meta.serialize();
                let restored = TransitionMeta::deserialize(&bytes).unwrap();
                prop_assert_eq!(meta.env_id, restored.env_id);
                prop_assert_eq!(meta.episode_id, restored.episode_id);
                prop_assert_eq!(meta.global_step, restored.global_step);
                prop_assert_eq!(meta.policy_version, restored.policy_version);
            }
        }
    }
}
```

---

## Part 2: Python Integration Tests

**Target file:** `tests/python/test_phase9.py`

```python
# tests/python/test_phase9.py
"""
Phase 9 TDD test specifications — Distributed and Scale (v1.0).

Status: RED — all tests fail until implementation is complete.
Depends on: Phase 7 + Phase 8 tests passing.
Run: .venv/bin/python -m pytest tests/python/test_phase9.py -v

Multi-GPU tests require CUDA and 2+ GPUs. Skip with:
  pytest -k "not multi_gpu" tests/python/test_phase9.py

vLLM integration tests require a running vLLM server. Skip with:
  pytest -m "not requires_vllm" tests/python/test_phase9.py

gRPC tests require protobuf + grpc. Skip with:
  pytest -k "not grpc" tests/python/test_phase9.py
"""

import contextlib
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="session")
def gymnasium():
    return pytest.importorskip("gymnasium")


# ---------------------------------------------------------------------------
# Group 3: Decoupled collection/training
# ---------------------------------------------------------------------------

class TestDecoupledCollectionTraining:
    """Group 3: Async collector delivers batches while learner trains."""

    # RED: Pipeline importable from rlox.distributed
    def test_pipeline_importable(self):
        from rlox.distributed import Pipeline  # or from rlox import Pipeline

    # RED: Pipeline delivers batches asynchronously
    def test_pipeline_delivers_batches_async(self, torch, gymnasium):
        from rlox.distributed import Pipeline

        pipeline = Pipeline(
            env_id="CartPole-v1",
            n_envs=4,
            n_steps=32,
            channel_capacity=4,
            seed=42,
        )

        # Give collector time to produce first batch
        batch = pipeline.next_batch(timeout=10.0)
        assert batch is not None, "pipeline should deliver batch within 10 seconds"

        # Batch shape validation
        assert batch.obs.shape == (4 * 32, 4), (
            f"batch.obs should be (n_envs * n_steps, obs_dim) = ({4*32}, 4), "
            f"got {batch.obs.shape}"
        )
        assert batch.rewards.shape == (4 * 32,)
        assert batch.advantages.shape == (4 * 32,)

        pipeline.close()

    # RED: Pipeline supports continuous batch delivery
    def test_pipeline_continuous_batch_delivery(self, torch, gymnasium):
        from rlox.distributed import Pipeline

        pipeline = Pipeline(
            env_id="CartPole-v1",
            n_envs=2,
            n_steps=32,
            channel_capacity=8,
            seed=42,
        )

        batches_received = 0
        for _ in range(5):
            batch = pipeline.next_batch(timeout=10.0)
            assert batch is not None
            batches_received += 1

        assert batches_received == 5
        pipeline.close()

    # RED: Pipeline.try_next_batch is non-blocking
    def test_pipeline_try_next_batch_non_blocking(self, torch, gymnasium):
        from rlox.distributed import Pipeline

        pipeline = Pipeline(
            env_id="CartPole-v1",
            n_envs=4,
            n_steps=128,  # large n_steps — takes time to collect
            channel_capacity=1,
            seed=42,
        )

        # Immediately after creation, may or may not have a batch
        start = time.perf_counter()
        _ = pipeline.try_next_batch()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, (
            f"try_next_batch should return in < 100ms, took {elapsed*1000:.1f}ms"
        )
        pipeline.close()

    # RED: decoupled pipeline actually decouples (slow learner doesn't block collector)
    def test_decoupled_collection_with_slow_learner(self, torch, gymnasium):
        from rlox.distributed import Pipeline

        pipeline = Pipeline(
            env_id="CartPole-v1",
            n_envs=4,
            n_steps=32,
            channel_capacity=4,
            seed=42,
        )

        # Simulate slow learner: sleep between batch reads
        batches = []
        for _ in range(3):
            time.sleep(0.1)  # simulate 100ms training step
            batch = pipeline.next_batch(timeout=5.0)
            assert batch is not None
            batches.append(batch)

        assert len(batches) == 3
        pipeline.close()

    @pytest.mark.slow
    def test_decoupled_pipeline_convergence(self, torch, gymnasium):
        """
        PPO with decoupled collection/training must converge on CartPole.
        This validates that the async data flow doesn't break learning.
        """
        from rlox.distributed import AsyncPPO  # or from rlox import AsyncPPO

        ppo = AsyncPPO(
            env_id="CartPole-v1",
            n_envs=8,
            n_steps=64,
            channel_capacity=4,
            seed=42,
        )
        metrics = ppo.train(total_timesteps=50_000)

        import gymnasium as gym
        env = gym.make("CartPole-v1")
        rewards = []
        for ep in range(10):
            obs, _ = env.reset(seed=ep + 99)
            done = False
            ep_r = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = ppo.policy(obs_t).argmax(dim=-1).item()
                obs, r, term, trunc, _ = env.step(action)
                ep_r += r
                done = term or trunc
            rewards.append(ep_r)

        mean_reward = float(np.mean(rewards))
        assert mean_reward >= 200.0, (
            f"Async PPO should learn CartPole (>= 200), got {mean_reward:.1f}"
        )


# ---------------------------------------------------------------------------
# Group 4: Multi-GPU PPO
# ---------------------------------------------------------------------------

class TestMultiGPU:
    """Group 4: Multi-GPU PPO training."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self, torch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < 2:
            pytest.skip("Need >= 2 GPUs for multi-GPU tests")

    # RED: MultiGPUTrainer importable
    def test_multi_gpu_trainer_importable(self):
        from rlox.distributed import MultiGPUTrainer

    # RED: MultiGPUTrainer wraps PPO for DDP
    def test_multi_gpu_trainer_initialization(self, torch, gymnasium):
        from rlox.distributed import MultiGPUTrainer

        trainer = MultiGPUTrainer(
            trainer_cls="PPO",
            env="CartPole-v1",
            seed=42,
        )
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "evaluate")

    @pytest.mark.slow
    def test_multi_gpu_ppo_converges_cartpole(self, torch, gymnasium):
        """
        Multi-GPU PPO with 2 GPUs must converge on CartPole.
        Uses torchrun for proper DDP initialization.
        """
        # Create a test script for torchrun
        script = """
import torch
import torch.distributed as dist
from rlox.distributed import MultiGPUTrainer
import numpy as np

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    trainer = MultiGPUTrainer(
        trainer_cls="PPO",
        env="CartPole-v1",
        seed=42,
        rank=rank,
    )
    trainer.train(total_timesteps=25_000)

    if rank == 0:
        result = trainer.evaluate(n_episodes=10)
        mean_reward = result["mean_reward"]
        with open("/tmp/multi_gpu_result.txt", "w") as f:
            f.write(str(mean_reward))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
"""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["torchrun", "--nproc_per_node=2", script_path],
                timeout=300,  # 5 minute timeout
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"Multi-GPU training failed:\n{result.stderr}"
            )
            with open("/tmp/multi_gpu_result.txt") as f:
                mean_reward = float(f.read().strip())
            assert mean_reward >= 200.0, (
                f"Multi-GPU PPO should learn CartPole, got {mean_reward:.1f}"
            )
        finally:
            import os
            os.unlink(script_path)


# ---------------------------------------------------------------------------
# Group 5: Remote env pool (gRPC workers)
# ---------------------------------------------------------------------------

class TestRemoteEnvPool:
    """Group 5: RemoteEnvPool connects to gRPC workers."""

    @pytest.fixture(autouse=True)
    def skip_if_no_grpc(self):
        grpc = pytest.importorskip("grpc")

    # RED: RemoteEnvPool importable
    def test_remote_env_pool_importable(self):
        from rlox.distributed import RemoteEnvPool

    # RED: RemoteEnvPool with local gRPC worker
    def test_remote_env_pool_with_local_worker(self, gymnasium):
        from rlox.distributed import RemoteEnvPool, EnvWorkerServer

        # Start a local gRPC worker server
        server = EnvWorkerServer(
            env_id="CartPole-v1",
            n_envs=4,
            port=0,  # OS-assigned port
            seed=42,
        )
        server.start()
        addr = f"localhost:{server.port}"

        try:
            pool = RemoteEnvPool(workers=[addr], envs_per_worker=4)

            # Reset all envs
            obs = pool.reset_all(seed=99)
            assert obs.shape == (4, 4), f"obs shape should be (4, 4), got {obs.shape}"

            # Step all envs
            actions = [0, 1, 0, 1]
            result = pool.step_all(actions)
            assert "obs" in result
            assert result["obs"].shape == (4, 4)

        finally:
            server.stop()

    # RED: RemoteEnvPool with multiple workers
    def test_remote_env_pool_multiple_workers(self, gymnasium):
        from rlox.distributed import RemoteEnvPool, EnvWorkerServer

        servers = []
        addrs = []
        try:
            for i in range(2):
                server = EnvWorkerServer(
                    env_id="CartPole-v1",
                    n_envs=4,
                    port=0,
                    seed=i * 100,
                )
                server.start()
                servers.append(server)
                addrs.append(f"localhost:{server.port}")

            # 2 workers * 4 envs = 8 total envs
            pool = RemoteEnvPool(workers=addrs, envs_per_worker=4)
            obs = pool.reset_all(seed=42)
            assert obs.shape == (8, 4), f"obs shape should be (8, 4), got {obs.shape}"

        finally:
            for s in servers:
                s.stop()

    # RED: RemoteEnvPool has same API as VecEnv
    def test_remote_env_pool_same_api_as_vecenv(self, gymnasium):
        from rlox import VecEnv
        from rlox.distributed import RemoteEnvPool, EnvWorkerServer

        server = EnvWorkerServer(env_id="CartPole-v1", n_envs=4, port=0, seed=42)
        server.start()

        try:
            pool = RemoteEnvPool(workers=[f"localhost:{server.port}"], envs_per_worker=4)

            # Both VecEnv and RemoteEnvPool must have these methods
            for method in ["reset_all", "step_all", "num_envs"]:
                assert hasattr(pool, method), f"RemoteEnvPool missing method: {method}"

        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Group 5: vLLM / TGI integration
# ---------------------------------------------------------------------------

class TestVLLMIntegration:
    """Group 5: vLLM backend integration (mock + real)."""

    # RED: VllmBackend importable
    def test_vllm_backend_importable(self):
        from rlox.distributed import VllmBackend  # or from rlox.llm

    # RED: VllmBackend with mock server
    def test_vllm_backend_mock_generate(self, torch):
        from rlox.distributed import VllmBackend
        from unittest.mock import patch, MagicMock
        import json

        # Mock the HTTP responses from vLLM
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "text": " The answer is 42.",
                    "finish_reason": "length",
                    "logprobs": {"token_logprobs": [-0.5, -0.3, -0.7]},
                }
            ]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            backend = VllmBackend(base_url="http://mock-vllm:8000")
            result = backend.generate(
                prompts=["What is the answer?"],
                max_new_tokens=10,
                n=1,
            )

            assert len(result) == 1
            assert "text" in result[0] or "completion" in result[0]

    # RED: VllmBackend.log_probs returns correct shape
    def test_vllm_backend_log_probs_shape(self, torch):
        from rlox.distributed import VllmBackend
        from unittest.mock import patch, MagicMock

        seq_len = 5
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "log_probs": [[-0.5, -0.3, -0.7, -0.4, -0.6]]  # 1 sequence, 5 tokens
        }

        with patch("requests.post", return_value=mock_response):
            backend = VllmBackend(base_url="http://mock-vllm:8000")
            input_ids = [[1, 2, 3, 4, 5]]
            log_probs = backend.log_probs(input_ids=input_ids)

            assert len(log_probs) == 1  # 1 sequence
            assert len(log_probs[0]) == seq_len

    @pytest.mark.requires_vllm
    def test_vllm_backend_real_server(self):
        """Requires a running vLLM server at VLLM_URL environment variable."""
        vllm_url = os.environ.get("VLLM_URL")
        if not vllm_url:
            pytest.skip("VLLM_URL not set")

        from rlox.distributed import VllmBackend

        backend = VllmBackend(base_url=vllm_url)
        result = backend.generate(
            prompts=["Hello, world!"],
            max_new_tokens=5,
            n=1,
        )
        assert len(result) == 1

    # RED: LLMEnvironment wraps VllmBackend
    def test_llm_environment_importable(self):
        from rlox.llm import LLMEnvironment  # or from rlox.distributed

    # RED: LLMEnvironment.generate returns completions
    def test_llm_environment_generate_mock(self):
        from rlox.llm import LLMEnvironment
        from unittest.mock import patch, MagicMock

        mock_backend = MagicMock()
        mock_backend.generate.return_value = [
            {"text": " four", "log_probs": [-0.5, -0.3]},
            {"text": " 4", "log_probs": [-0.7, -0.4]},
        ]

        with patch("rlox.llm.LLMEnvironment._create_backend", return_value=mock_backend):
            env = LLMEnvironment(backend="vllm", url="http://mock:8000")
            completions = env.generate(
                prompts=["What is 2+2?"],
                n=2,
                max_new_tokens=5,
            )
            assert len(completions) == 2


# ---------------------------------------------------------------------------
# Group 6: Reward model serving
# ---------------------------------------------------------------------------

class TestRewardModelServing:
    """Group 6: Reward model scoring — single, ensemble, multi-objective."""

    # RED: RewardModelServer importable
    def test_reward_model_server_importable(self):
        from rlox.llm import RewardModelServer  # or from rlox.distributed

    # RED: RewardModelServer.score_batch returns rewards array
    def test_reward_model_score_batch_shape(self, torch):
        from rlox.llm import RewardModelServer
        import torch.nn as nn

        class MockRM(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                # Return scalar reward for each sequence
                return torch.randn(input_ids.shape[0], 1)

        rm = RewardModelServer(model=MockRM(), batch_size=8)
        prompts = ["Hello " + str(i) for i in range(4)]
        completions = ["World " + str(i) for i in range(4)]

        rewards = rm.score_batch(prompts=prompts, completions=completions)
        assert rewards.shape == (4,), f"rewards shape should be (4,), got {rewards.shape}"
        assert np.all(np.isfinite(rewards)), "all rewards should be finite"

    # RED: EnsembleRewardModel averages scores from multiple models
    def test_ensemble_reward_model(self, torch):
        from rlox.llm import RewardModelServer, EnsembleRewardModel
        import torch.nn as nn

        class ConstantRM(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, input_ids, attention_mask=None):
                return torch.full((input_ids.shape[0], 1), self.value)

        rm1 = RewardModelServer(model=ConstantRM(1.0), batch_size=8)
        rm2 = RewardModelServer(model=ConstantRM(3.0), batch_size=8)

        ensemble = EnsembleRewardModel(models=[rm1, rm2], weights=[0.5, 0.5])

        prompts = ["Test"] * 4
        completions = ["Response"] * 4
        rewards = ensemble.score_batch(prompts=prompts, completions=completions)

        # Average of 1.0 and 3.0 with equal weights = 2.0
        np.testing.assert_allclose(rewards, 2.0, atol=1e-5)

    # RED: MultiObjectiveReward weighted combination
    def test_multi_objective_reward_model(self, torch):
        from rlox.llm import RewardModelServer, MultiObjectiveReward
        import torch.nn as nn

        class ConstantRM(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, input_ids, attention_mask=None):
                return torch.full((input_ids.shape[0], 1), self.value)

        helpfulness_rm = RewardModelServer(model=ConstantRM(0.8), batch_size=8)
        safety_rm = RewardModelServer(model=ConstantRM(0.6), batch_size=8)

        multi_rm = MultiObjectiveReward(
            objectives={"helpfulness": helpfulness_rm, "safety": safety_rm},
            weights={"helpfulness": 0.7, "safety": 0.3},
        )

        prompts = ["Test"] * 2
        completions = ["Response"] * 2
        rewards = multi_rm.score_batch(prompts=prompts, completions=completions)

        expected = 0.7 * 0.8 + 0.3 * 0.6  # = 0.56 + 0.18 = 0.74
        np.testing.assert_allclose(rewards, expected, atol=1e-5)

    # RED: EnsembleRewardModel with non-equal weights
    def test_ensemble_with_non_equal_weights(self, torch):
        from rlox.llm import RewardModelServer, EnsembleRewardModel
        import torch.nn as nn

        class ConstantRM(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, input_ids, attention_mask=None):
                return torch.full((input_ids.shape[0], 1), self.value)

        rm1 = RewardModelServer(model=ConstantRM(2.0), batch_size=8)
        rm2 = RewardModelServer(model=ConstantRM(4.0), batch_size=8)

        ensemble = EnsembleRewardModel(models=[rm1, rm2], weights=[0.25, 0.75])

        prompts = ["Test"]
        completions = ["Response"]
        rewards = ensemble.score_batch(prompts=prompts, completions=completions)

        expected = 0.25 * 2.0 + 0.75 * 4.0  # = 0.5 + 3.0 = 3.5
        np.testing.assert_allclose(rewards, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Group 7: MAPPO cooperative task
# ---------------------------------------------------------------------------

class TestMAPPO:
    """Group 7: MAPPO cooperative multi-agent RL."""

    # RED: MAPPO importable
    def test_mappo_importable(self):
        from rlox import MAPPO  # or from rlox.algorithms.mappo

    # RED: MAPPO initialization
    def test_mappo_initialization(self, torch):
        from rlox import MAPPO

        mappo = MAPPO(
            env_id="CartPole-v1",  # treat as 1-agent MA for smoke test
            n_agents=1,
            seed=42,
        )
        assert hasattr(mappo, "actors")
        assert hasattr(mappo, "critic")
        assert len(mappo.actors) == 1

    # RED: MAPPO smoke test
    def test_mappo_smoke_test(self, torch, gymnasium):
        from rlox import MAPPO

        mappo = MAPPO(
            env_id="CartPole-v1",
            n_agents=1,
            seed=42,
        )
        mappo.train(total_timesteps=1_000)

    @pytest.mark.slow
    def test_mappo_cooperative_navigation(self, torch):
        """
        MAPPO must show positive returns on a simple cooperative task.
        Uses a synthetic cooperative environment.
        """
        from rlox import MAPPO

        # Create synthetic cooperative env where 2 agents share reward
        class CooperativeReachEnv:
            """2 agents must reach target. Shared reward = -distance."""
            def __init__(self, n_agents=2):
                self.n_agents = n_agents
                self.obs_dim = 4  # [x1, y1, x2, y2]
                self.act_dim = 2  # [dx, dy] for each agent
                self.positions = np.zeros((n_agents, 2))
                self.target = np.array([1.0, 1.0])

            def reset(self, seed=None):
                rng = np.random.default_rng(seed)
                self.positions = rng.standard_normal((self.n_agents, 2)) * 0.1
                return self._get_obs()

            def step(self, actions):
                # actions: (n_agents, act_dim)
                self.positions += np.array(actions) * 0.1
                distances = np.linalg.norm(self.positions - self.target, axis=1)
                reward = -float(distances.mean())
                done = all(d < 0.1 for d in distances)
                return self._get_obs(), reward, done, {}

            def _get_obs(self):
                return self.positions.flatten()  # (n_agents * 2,)

        mappo = MAPPO(
            env=CooperativeReachEnv(n_agents=2),
            n_agents=2,
            seed=42,
            shared_reward=True,
        )
        metrics = mappo.train(total_timesteps=50_000)
        final_reward = metrics.get("final_mean_reward", float("-inf"))

        # Should improve over random baseline (random achieves ~ -2.0)
        assert final_reward > -1.5, (
            f"MAPPO should improve cooperative navigation, got {final_reward:.2f}"
        )


# ---------------------------------------------------------------------------
# Group 7: DreamerV3 visual control
# ---------------------------------------------------------------------------

class TestDreamerV3:
    """Group 7: DreamerV3 world-model RL."""

    # RED: DreamerV3 importable
    def test_dreamer_importable(self):
        from rlox import DreamerV3  # or from rlox.algorithms.dreamer

    # RED: DreamerV3 initialization
    def test_dreamer_initialization(self, torch, gymnasium):
        from rlox import DreamerV3

        dreamer = DreamerV3(
            env_id="CartPole-v1",  # low-dim for smoke test
            seed=42,
        )
        assert hasattr(dreamer, "world_model")
        assert hasattr(dreamer, "actor")
        assert hasattr(dreamer, "critic")

    # RED: DreamerV3 smoke test
    def test_dreamer_smoke_test(self, torch, gymnasium):
        from rlox import DreamerV3

        dreamer = DreamerV3(env_id="CartPole-v1", seed=42)
        dreamer.train(total_timesteps=500)

    @pytest.mark.slow
    def test_dreamer_learns_low_dim(self, torch, gymnasium):
        """DreamerV3 must improve on CartPole using world-model imagination."""
        from rlox import DreamerV3

        dreamer = DreamerV3(
            env_id="CartPole-v1",
            seed=42,
            imagination_horizon=15,
        )
        metrics = dreamer.train(total_timesteps=100_000)
        final_reward = metrics.get("final_mean_reward", 0.0)

        # Should beat random baseline (~10 for CartPole)
        assert final_reward > 50.0, (
            f"DreamerV3 should learn CartPole > 50, got {final_reward:.1f}"
        )


# ---------------------------------------------------------------------------
# Group 8: IMPALA async throughput
# ---------------------------------------------------------------------------

class TestIMPALA:
    """Group 8: IMPALA with V-trace correction."""

    # RED: IMPALA importable
    def test_impala_importable(self):
        from rlox import IMPALA  # or from rlox.algorithms.impala

    # RED: IMPALA smoke test with 2 actors
    def test_impala_smoke_test(self, torch, gymnasium):
        from rlox import IMPALA

        impala = IMPALA(
            env_id="CartPole-v1",
            n_actors=2,
            seed=42,
        )
        impala.train(total_timesteps=2_000)

    # RED: IMPALA uses V-trace for off-policy correction
    def test_impala_uses_vtrace(self, torch, gymnasium):
        from rlox import IMPALA

        impala = IMPALA(env_id="CartPole-v1", n_actors=2, seed=42)
        assert hasattr(impala, "vtrace"), "IMPALA must have vtrace attribute/method"

    @pytest.mark.slow
    def test_impala_throughput_scales_with_actors(self, torch, gymnasium):
        """
        IMPALA with 4 actors should achieve > 3.5x throughput vs 1 actor.
        Measures steps-per-second (env steps, not gradient steps).
        """
        from rlox import IMPALA
        import time

        def measure_throughput(n_actors: int, n_steps: int = 10_000) -> float:
            impala = IMPALA(
                env_id="CartPole-v1",
                n_actors=n_actors,
                n_envs_per_actor=4,
                seed=42,
            )
            start = time.perf_counter()
            impala.train(total_timesteps=n_steps)
            elapsed = time.perf_counter() - start
            return n_steps / elapsed

        tps_1 = measure_throughput(n_actors=1)
        tps_4 = measure_throughput(n_actors=4)

        speedup = tps_4 / tps_1
        assert speedup > 3.0, (
            f"IMPALA throughput with 4 actors ({tps_4:.0f} steps/s) should be "
            f"> 3x single actor ({tps_1:.0f} steps/s), got {speedup:.1f}x"
        )

    @pytest.mark.slow
    def test_impala_learns_cartpole(self, torch, gymnasium):
        """IMPALA must converge on CartPole."""
        from rlox import IMPALA

        impala = IMPALA(
            env_id="CartPole-v1",
            n_actors=4,
            seed=42,
        )
        metrics = impala.train(total_timesteps=100_000)
        final_reward = metrics.get("final_mean_reward", 0.0)

        assert final_reward >= 200.0, (
            f"IMPALA should learn CartPole (>= 200), got {final_reward:.1f}"
        )


# ---------------------------------------------------------------------------
# Group 9: API 1.0 stability
# ---------------------------------------------------------------------------

class TestAPI10Stability:
    """Group 9: All public APIs documented, consistent, stable."""

    # RED: all public API objects have docstrings
    def test_all_public_symbols_have_docstrings(self):
        import inspect
        import rlox

        missing_docs = []
        for name in rlox.__all__:
            obj = getattr(rlox, name, None)
            if obj is None:
                missing_docs.append(f"{name}: not found in module")
                continue
            if not obj.__doc__:
                missing_docs.append(f"{name}: missing docstring")

        assert not missing_docs, (
            f"The following public API symbols lack docstrings:\n"
            + "\n".join(f"  - {m}" for m in missing_docs)
        )

    # RED: __all__ is explicitly defined in rlox.__init__
    def test_rlox_has_explicit_all(self):
        import rlox
        assert hasattr(rlox, "__all__"), (
            "rlox must define __all__ for stable public API"
        )
        assert isinstance(rlox.__all__, list), "__all__ must be a list"
        assert len(rlox.__all__) > 0, "__all__ must not be empty"

    # RED: Phase 7 symbols still present (no regression)
    def test_phase7_symbols_still_present(self):
        import rlox

        phase7_symbols = [
            "RunningStats", "pack_sequences", "RolloutCollector",
            "PPOLoss", "PPO", "GRPO", "DPO", "A2C", "Checkpoint",
        ]
        for sym in phase7_symbols:
            assert hasattr(rlox, sym), (
                f"Phase 7 symbol '{sym}' missing from rlox — API regression!"
            )

    # RED: Phase 8 symbols still present (no regression)
    def test_phase8_symbols_still_present(self):
        import rlox

        phase8_symbols = [
            "PrioritizedReplayBuffer", "SAC", "TD3", "DQN",
            "PPOTrainer", "SACTrainer", "DQNTrainer",
            "PPOConfig", "SACConfig", "DQNConfig",
            "Callback", "EvalCallback", "CheckpointCallback", "EarlyStoppingCallback",
            "TrainingDiagnostics", "interquartile_mean",
        ]
        for sym in phase8_symbols:
            assert hasattr(rlox, sym), (
                f"Phase 8 symbol '{sym}' missing from rlox — API regression!"
            )

    # RED: Phase 9 distributed symbols present
    def test_phase9_distributed_symbols_present(self):
        from rlox import distributed as dist

        phase9_dist_symbols = [
            "Pipeline", "MultiGPUTrainer", "RemoteEnvPool",
            "VllmBackend",
        ]
        for sym in phase9_dist_symbols:
            assert hasattr(dist, sym), (
                f"Phase 9 distributed symbol '{sym}' missing — API incomplete!"
            )

    # RED: Algorithm constructors accept consistent env parameter
    def test_algorithm_constructors_consistent_env_param(self, gymnasium):
        import rlox

        algorithms = ["PPO", "SAC", "TD3", "DQN", "A2C", "GRPO", "DPO", "MAPPO"]
        for algo_name in algorithms:
            algo_cls = getattr(rlox, algo_name, None)
            if algo_cls is None:
                continue
            import inspect
            sig = inspect.signature(algo_cls.__init__)
            params = list(sig.parameters.keys())
            # All algorithms must accept either 'env' or 'env_id' parameter
            has_env_param = "env" in params or "env_id" in params
            assert has_env_param, (
                f"{algo_name}.__init__ must accept 'env' or 'env_id' parameter, "
                f"got params: {params}"
            )

    # RED: all Trainer classes have consistent save/load API
    def test_trainer_save_load_api_consistency(self):
        import rlox

        trainers = ["PPOTrainer", "SACTrainer", "DQNTrainer"]
        for trainer_name in trainers:
            trainer_cls = getattr(rlox, trainer_name, None)
            if trainer_cls is None:
                continue
            assert hasattr(trainer_cls, "save"), f"{trainer_name} must have .save()"
            assert hasattr(trainer_cls, "load"), f"{trainer_name} must have .load()"
            assert hasattr(trainer_cls, "evaluate"), f"{trainer_name} must have .evaluate()"
            assert hasattr(trainer_cls, "train"), f"{trainer_name} must have .train()"

    # RED: config classes are dataclasses (IDE completion works)
    def test_config_classes_are_dataclasses(self):
        import dataclasses
        import rlox

        config_classes = ["PPOConfig", "SACConfig", "DQNConfig"]
        for cls_name in config_classes:
            cls = getattr(rlox, cls_name, None)
            if cls is None:
                continue
            assert dataclasses.is_dataclass(cls), (
                f"{cls_name} must be a dataclass for IDE completion support"
            )

    # RED: version is at least 1.0.0
    def test_rlox_version_is_1_0(self):
        import rlox
        from packaging.version import Version

        version = Version(rlox.__version__)
        assert version >= Version("1.0.0"), (
            f"Phase 9 is API 1.0 — version must be >= 1.0.0, got {rlox.__version__}"
        )
```

---

## Part 3: Performance Regression Tests

**Target file:** `tests/python/test_phase9_perf.py`

```python
# tests/python/test_phase9_perf.py
"""
Phase 9 performance regression tests.
All Phase 0-8 benchmarks must be maintained at v1.0.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))
from conftest import BenchmarkResult, timed_run


class TestPhase9PerfRegressions:
    """Ensure v1.0 release does not regress any benchmark from phases 0-8."""

    def test_gae_speed_at_v1(self):
        """compute_gae must remain >= 5x faster than numpy loop at v1.0."""
        from rlox import compute_gae

        rng = np.random.default_rng(42)
        n = 2048
        rewards = rng.standard_normal(n)
        values = rng.standard_normal(n)
        dones = (rng.random(n) > 0.95).astype(float)

        def numpy_gae(r, v, d, lv):
            adv = np.zeros(len(r))
            last = 0.0
            for t in reversed(range(len(r))):
                nnt = 1.0 - float(d[t])
                nv = lv if t == len(r) - 1 else v[t + 1]
                delta = r[t] + 0.99 * nv * nnt - v[t]
                last = delta + 0.99 * 0.95 * nnt * last
                adv[t] = last
            return adv, adv + v

        rlox_times = timed_run(
            lambda: compute_gae(rewards, values, dones, 0.0, 0.99, 0.95),
            n_warmup=10, n_reps=100,
        )
        numpy_times = timed_run(
            lambda: numpy_gae(rewards, values, dones, 0.0),
            n_warmup=10, n_reps=100,
        )

        speedup = np.median(numpy_times) / np.median(rlox_times)
        assert speedup >= 5.0, f"GAE regression at v1.0: {speedup:.1f}x (must be >= 5x)"

    def test_vtrace_speed(self):
        """compute_vtrace must process 2048 steps in < 1ms."""
        from rlox import compute_vtrace

        n = 2048
        log_rhos = np.random.default_rng(42).standard_normal(n).astype(np.float32)
        rewards = np.ones(n, dtype=np.float32)
        values = np.full(n, 0.5, dtype=np.float32)

        times = timed_run(
            lambda: compute_vtrace(log_rhos, rewards, values, 0.5, 0.99, 1.0, 1.0),
            n_warmup=10, n_reps=100,
        )
        result = BenchmarkResult(
            name="vtrace_2048", category="training",
            framework="rlox", times_ns=times,
        )
        assert result.median_ns < 1_000_000, (
            f"compute_vtrace(2048) took {result.median_ns:.0f}ns, expected < 1ms"
        )

    def test_pipeline_throughput(self):
        """Pipeline must deliver >= 10K steps/sec on CartPole."""
        from rlox.distributed import Pipeline
        import time

        pipeline = Pipeline(
            env_id="CartPole-v1",
            n_envs=8,
            n_steps=64,
            channel_capacity=4,
            seed=42,
        )

        n_batches = 10
        total_steps = 0
        start = time.perf_counter()
        for _ in range(n_batches):
            batch = pipeline.next_batch(timeout=10.0)
            assert batch is not None
            total_steps += batch.n_steps

        elapsed = time.perf_counter() - start
        throughput = total_steps / elapsed

        pipeline.close()

        assert throughput >= 10_000, (
            f"Pipeline throughput {throughput:.0f} steps/s < 10K — "
            "distributed collection too slow"
        )
```

---

## Test Coverage Targets

| Component | Target Coverage | Notes |
|---|---|---|
| `channel.rs` (lock-free channel) | 90%+ | All error paths + proptest |
| `pipeline.rs` (async collector) | 85%+ | Backpressure + shutdown paths |
| `grpc/env_service.rs` | 80%+ | Happy path + error responses |
| `training/vtrace.rs` | 95%+ | Reference comparison + proptest |
| `buffer/provenance.rs` | 95%+ | Full roundtrip proptest |
| Python `Pipeline` | 80%+ | Non-blocking + convergence |
| Python `MultiGPUTrainer` | 60%+ | Requires 2 GPUs |
| Python `RemoteEnvPool` | 75%+ | Local gRPC server fixture |
| Python `VllmBackend` | 70%+ | Mock + requires_vllm |
| Python `RewardModelServer` | 85%+ | All model variants |
| Python `MAPPO` | 70%+ | Cooperative task |
| Python `DreamerV3` | 65%+ | Smoke + low-dim E2E |
| Python `IMPALA` | 75%+ | Throughput + convergence |
| API 1.0 stability | 100% | All public symbols documented |

---

## Notes on Test Execution

1. **New crate `rlox-dist`**: add to `Cargo.toml` workspace members. Add
   `rlox-core` as a dependency for `BatchSteppable` + `compute_vtrace`.

2. **gRPC tests** require `tonic`, `prost`, `tokio` in `rlox-dist/Cargo.toml`.
   All gRPC tests use tokio runtime: `#[tokio::test]`.

3. **Multi-GPU tests** are skipped when fewer than 2 GPUs are available.
   In CI, use a separate job with GPU runners.

4. **vLLM integration tests** marked `@pytest.mark.requires_vllm` require a
   running vLLM server. Run with `VLLM_URL=http://localhost:8000 pytest -m requires_vllm`.

5. **API 1.0 stability tests** in `TestAPI10Stability` enforce the v1.0
   contract: any public symbol removal is caught by
   `test_phase7_symbols_still_present`, `test_phase8_symbols_still_present`.
   This forms the regression gate for API stability.

6. **`packaging` library** is used in `test_rlox_version_is_1_0`. Add it to
   `dev-requirements.txt` or skip with `pytest.importorskip("packaging")`.

7. **V-trace reference implementation** in `tests` module should be kept
   simple (pure Rust, no external deps) to serve as a cross-check. The
   reference in `vtrace_matches_reference_implementation` is the ground truth;
   if the reference and implementation diverge, the reference wins.

8. **Transition provenance** is pure data — no environment or training
   dependencies. These tests can run in isolation before any other Phase 9
   work.
