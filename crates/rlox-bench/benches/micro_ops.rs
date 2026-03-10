use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use rlox_core::env::spaces::Observation;

fn bench_observation_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("observation_clone");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for obs_dim in [4, 17, 28224] {
        group.bench_with_input(
            BenchmarkId::new("obs_dim", obs_dim),
            &obs_dim,
            |b, &dim| {
                let obs = Observation(vec![0.42_f32; dim]);
                b.iter(|| black_box(obs.clone()));
            },
        );
    }

    group.finish();
}

fn bench_vec_push_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_push_f32");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("push_1M", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(1_000_000);
            for i in 0..1_000_000u32 {
                v.push(i as f32);
            }
            black_box(v);
        });
    });

    group.finish();
}

fn bench_vec_extend_from_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_extend_from_slice");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for obs_dim in [4, 17, 28224] {
        group.bench_with_input(
            BenchmarkId::new("obs_dim", obs_dim),
            &obs_dim,
            |b, &dim| {
                let slice = vec![0.42_f32; dim];
                // Simulate appending 256 observations (one batch) into a columnar buffer
                let num_appends = 256;
                b.iter(|| {
                    let mut buf = Vec::with_capacity(dim * num_appends);
                    for _ in 0..num_appends {
                        buf.extend_from_slice(black_box(&slice));
                    }
                    black_box(buf);
                });
            },
        );
    }

    group.finish();
}

fn bench_rand_chacha_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_chacha_gen");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("gen_f32_1M", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        b.iter(|| {
            let mut sum = 0.0_f32;
            for _ in 0..1_000_000 {
                sum += rng.random::<f32>();
            }
            black_box(sum);
        });
    });

    group.finish();
}

fn bench_par_iter_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_iter_overhead");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for n in [1, 4, 16, 64] {
        // Sequential baseline
        group.bench_function(BenchmarkId::new("seq_iter", n), |b| {
            let data: Vec<u64> = (0..n).collect();
            b.iter(|| {
                let sum: u64 = data.iter().map(|x| black_box(*x)).sum();
                black_box(sum);
            });
        });

        // Parallel with Rayon
        group.bench_function(BenchmarkId::new("par_iter", n), |b| {
            let data: Vec<u64> = (0..n).collect();
            b.iter(|| {
                let sum: u64 = data.par_iter().map(|x| black_box(*x)).sum();
                black_box(sum);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_observation_clone,
    bench_vec_push_f32,
    bench_vec_extend_from_slice,
    bench_rand_chacha_gen,
    bench_par_iter_overhead,
);
criterion_main!(benches);
