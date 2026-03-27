use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Burn backend types
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::prelude::Backend;

// Candle types
use candle_core::Device as CandleDevice;

// Shared trait types
use rlox_nn::{
    ActorCritic, ContinuousQFunction, DeterministicPolicy, PPOStepConfig, QFunction,
    StochasticPolicy, TensorData,
};

// Backend implementations
use rlox_burn::actor_critic::BurnActorCritic;
use rlox_burn::continuous_q::BurnTwinQ;
use rlox_burn::deterministic::BurnDeterministicPolicy;
use rlox_burn::dqn::BurnDQN;
use rlox_burn::stochastic::BurnStochasticPolicy;

use rlox_candle::actor_critic::CandleActorCritic;
use rlox_candle::continuous_q::CandleTwinQ;
use rlox_candle::deterministic::CandleDeterministicPolicy;
use rlox_candle::dqn::CandleDQN;
use rlox_candle::stochastic::CandleStochasticPolicy;

type B = Autodiff<NdArray>;
type BurnDevice = <NdArray as Backend>::Device;

fn burn_device() -> BurnDevice {
    Default::default()
}

fn random_obs(batch: usize, dim: usize) -> TensorData {
    let data: Vec<f32> = (0..batch * dim)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    TensorData::new(data, vec![batch, dim])
}

// ─── ActorCritic (PPO) inference ────────────────────────────

fn bench_actor_critic_act(c: &mut Criterion) {
    let mut group = c.benchmark_group("actor_critic_act");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let obs_dim = 4;
    let n_actions = 2;
    let hidden = 64;

    for batch_size in [1, 32, 256] {
        let obs = random_obs(batch_size, obs_dim);

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let ac = BurnActorCritic::<B>::new(
                obs_dim,
                n_actions,
                hidden,
                3e-4,
                burn_device().into(),
                42,
            );
            b.iter(|| ac.act(black_box(&obs)).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let ac =
                    CandleActorCritic::new(obs_dim, n_actions, hidden, 3e-4, CandleDevice::Cpu, 42)
                        .unwrap();
                b.iter(|| ac.act(black_box(&obs)).unwrap());
            },
        );
    }
    group.finish();
}

// ─── ActorCritic (PPO) training step ────────────────────────

fn bench_ppo_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppo_step");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    let obs_dim = 4;
    let n_actions = 2;
    let hidden = 64;
    let config = PPOStepConfig::default();

    for batch_size in [64, 256] {
        let obs = random_obs(batch_size, obs_dim);
        let actions = TensorData::new(vec![0.0; batch_size], vec![batch_size]);
        let old_lp = TensorData::new(vec![-0.7; batch_size], vec![batch_size]);
        let advantages = TensorData::new(vec![1.0; batch_size], vec![batch_size]);
        let returns = TensorData::new(vec![1.0; batch_size], vec![batch_size]);
        let old_values = TensorData::zeros(vec![batch_size]);

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let mut ac = BurnActorCritic::<B>::new(
                obs_dim,
                n_actions,
                hidden,
                3e-4,
                burn_device().into(),
                42,
            );
            b.iter(|| {
                ac.ppo_step(
                    black_box(&obs),
                    black_box(&actions),
                    black_box(&old_lp),
                    black_box(&advantages),
                    black_box(&returns),
                    black_box(&old_values),
                    &config,
                )
                .unwrap()
            });
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let mut ac =
                    CandleActorCritic::new(obs_dim, n_actions, hidden, 3e-4, CandleDevice::Cpu, 42)
                        .unwrap();
                b.iter(|| {
                    ac.ppo_step(
                        black_box(&obs),
                        black_box(&actions),
                        black_box(&old_lp),
                        black_box(&advantages),
                        black_box(&returns),
                        black_box(&old_values),
                        &config,
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

// ─── DQN inference + training step ──────────────────────────

fn bench_dqn_q_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("dqn_q_values");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let obs_dim = 4;
    let n_actions = 2;
    let hidden = 64;

    for batch_size in [1, 32, 256] {
        let obs = random_obs(batch_size, obs_dim);

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let dqn = BurnDQN::<B>::new(obs_dim, n_actions, hidden, 1e-4, burn_device().into());
            b.iter(|| dqn.q_values(black_box(&obs)).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let dqn =
                    CandleDQN::new(obs_dim, n_actions, hidden, 1e-4, CandleDevice::Cpu).unwrap();
                b.iter(|| dqn.q_values(black_box(&obs)).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_dqn_td_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("dqn_td_step");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    let obs_dim = 4;
    let n_actions = 2;
    let hidden = 64;

    for batch_size in [64, 256] {
        let obs = random_obs(batch_size, obs_dim);
        let actions = TensorData::new(vec![0.0; batch_size], vec![batch_size]);
        let targets = TensorData::new(vec![1.0; batch_size], vec![batch_size]);

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let mut dqn = BurnDQN::<B>::new(obs_dim, n_actions, hidden, 1e-4, burn_device().into());
            b.iter(|| {
                dqn.td_step(
                    black_box(&obs),
                    black_box(&actions),
                    black_box(&targets),
                    None,
                )
                .unwrap()
            });
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let mut dqn =
                    CandleDQN::new(obs_dim, n_actions, hidden, 1e-4, CandleDevice::Cpu).unwrap();
                b.iter(|| {
                    dqn.td_step(
                        black_box(&obs),
                        black_box(&actions),
                        black_box(&targets),
                        None,
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

// ─── SAC stochastic policy ──────────────────────────────────

fn bench_sac_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("sac_sample_actions");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let obs_dim = 17; // HalfCheetah-like
    let act_dim = 6;
    let hidden = 256;

    for batch_size in [1, 32, 256] {
        let obs = random_obs(batch_size, obs_dim);

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let policy = BurnStochasticPolicy::<B>::new(
                obs_dim,
                act_dim,
                hidden,
                3e-4,
                burn_device().into(),
                42,
            );
            b.iter(|| policy.sample_actions(black_box(&obs)).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let policy = CandleStochasticPolicy::new(
                    obs_dim,
                    act_dim,
                    hidden,
                    3e-4,
                    CandleDevice::Cpu,
                    42,
                )
                .unwrap();
                b.iter(|| policy.sample_actions(black_box(&obs)).unwrap());
            },
        );
    }
    group.finish();
}

// ─── TD3 deterministic policy ───────────────────────────────

fn bench_td3_act(c: &mut Criterion) {
    let mut group = c.benchmark_group("td3_act");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let obs_dim = 17;
    let act_dim = 6;
    let hidden = 256;

    for batch_size in [1, 32, 256] {
        let obs = random_obs(batch_size, obs_dim);

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let policy = BurnDeterministicPolicy::<B>::new(
                obs_dim,
                act_dim,
                hidden,
                1.0,
                1e-3,
                burn_device().into(),
            );
            b.iter(|| policy.act(black_box(&obs)).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let policy = CandleDeterministicPolicy::new(
                    obs_dim,
                    act_dim,
                    hidden,
                    1.0,
                    1e-3,
                    CandleDevice::Cpu,
                )
                .unwrap();
                b.iter(|| policy.act(black_box(&obs)).unwrap());
            },
        );
    }
    group.finish();
}

// ─── Twin-Q critic ──────────────────────────────────────────

fn bench_twin_q(c: &mut Criterion) {
    let mut group = c.benchmark_group("twin_q_values");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    let obs_dim = 17;
    let act_dim = 6;
    let hidden = 256;

    for batch_size in [1, 32, 256] {
        let obs = random_obs(batch_size, obs_dim);
        let actions = TensorData::new(
            (0..batch_size * act_dim)
                .map(|i| (i as f32 * 0.1).sin())
                .collect(),
            vec![batch_size, act_dim],
        );

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let critic = BurnTwinQ::<B>::new(obs_dim, act_dim, hidden, 3e-4, burn_device().into());
            b.iter(|| {
                critic
                    .twin_q_values(black_box(&obs), black_box(&actions))
                    .unwrap()
            });
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let critic =
                    CandleTwinQ::new(obs_dim, act_dim, hidden, 3e-4, CandleDevice::Cpu).unwrap();
                b.iter(|| {
                    critic
                        .twin_q_values(black_box(&obs), black_box(&actions))
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

// ─── Critic training step ───────────────────────────────────

fn bench_critic_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("critic_step");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    let obs_dim = 17;
    let act_dim = 6;
    let hidden = 256;

    for batch_size in [64, 256] {
        let obs = random_obs(batch_size, obs_dim);
        let actions = TensorData::new(
            (0..batch_size * act_dim)
                .map(|i| (i as f32 * 0.1).sin())
                .collect(),
            vec![batch_size, act_dim],
        );
        let targets = TensorData::new(vec![1.0; batch_size], vec![batch_size]);

        group.bench_with_input(BenchmarkId::new("burn", batch_size), &batch_size, |b, _| {
            let mut critic =
                BurnTwinQ::<B>::new(obs_dim, act_dim, hidden, 3e-4, burn_device().into());
            b.iter(|| {
                critic
                    .critic_step(black_box(&obs), black_box(&actions), black_box(&targets))
                    .unwrap()
            });
        });

        group.bench_with_input(
            BenchmarkId::new("candle", batch_size),
            &batch_size,
            |b, _| {
                let mut critic =
                    CandleTwinQ::new(obs_dim, act_dim, hidden, 3e-4, CandleDevice::Cpu).unwrap();
                b.iter(|| {
                    critic
                        .critic_step(black_box(&obs), black_box(&actions), black_box(&targets))
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_actor_critic_act,
    bench_ppo_step,
    bench_dqn_q_values,
    bench_dqn_td_step,
    bench_sac_sample,
    bench_td3_act,
    bench_twin_q,
    bench_critic_step,
);
criterion_main!(benches);
