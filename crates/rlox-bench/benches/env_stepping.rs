use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use rlox_core::env::builtins::CartPole;
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::spaces::Action;
use rlox_core::env::RLEnv;
use rlox_core::seed::derive_seed;

/// Helper: build a VecEnv with `n` CartPole sub-environments.
fn make_vec_env(n: usize, seed: u64) -> VecEnv {
    let envs: Vec<Box<dyn RLEnv>> = (0..n)
        .map(|i| {
            let s = derive_seed(seed, i);
            Box::new(CartPole::new(Some(s))) as Box<dyn RLEnv>
        })
        .collect();
    VecEnv::new(envs)
}

fn bench_cartpole_single_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("cartpole_single_step");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("step", |b| {
        let mut env = CartPole::new(Some(42));
        b.iter(|| match env.step(black_box(&Action::Discrete(1))) {
            Ok(_) => {}
            Err(_) => {
                let _ = env.reset(Some(42));
            }
        });
    });

    group.finish();
}

fn bench_vecenv_step_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("vecenv_step_all");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for n in [1, 4, 16, 64, 128, 256, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut venv = make_vec_env(n, 42);
            let actions: Vec<Action> = (0..n).map(|i| Action::Discrete((i % 2) as u32)).collect();

            b.iter(|| {
                let result = venv.step_all(black_box(&actions));
                if result.is_err() {
                    let _ = venv.reset_all(Some(42));
                }
            });
        });
    }
    group.finish();
}

fn bench_vecenv_reset_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("vecenv_reset_all");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for n in [1, 4, 16, 64, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut venv = make_vec_env(n, 42);

            b.iter(|| {
                let _ = black_box(venv.reset_all(Some(42)));
            });
        });
    }
    group.finish();
}

fn bench_scaling_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_efficiency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Baseline: single CartPole step throughput
    group.bench_function(BenchmarkId::new("cartpole_step", 1), |b| {
        let mut env = CartPole::new(Some(42));
        b.iter(|| match env.step(black_box(&Action::Discrete(1))) {
            Ok(_) => {}
            Err(_) => {
                let _ = env.reset(Some(42));
            }
        });
    });

    // VecEnv throughput for various N
    for n in [1, 4, 16, 64, 128, 256, 512, 1024] {
        group.bench_function(BenchmarkId::new("vecenv_step", n), |b| {
            let mut venv = make_vec_env(n, 42);
            let actions: Vec<Action> = (0..n).map(|i| Action::Discrete((i % 2) as u32)).collect();

            b.iter(|| {
                let result = venv.step_all(black_box(&actions));
                if result.is_err() {
                    let _ = venv.reset_all(Some(42));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cartpole_single_step,
    bench_vecenv_step_all,
    bench_vecenv_reset_all,
    bench_scaling_efficiency,
);
criterion_main!(benches);
