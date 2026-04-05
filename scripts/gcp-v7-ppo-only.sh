#!/bin/bash
set -euo pipefail
exec > /var/log/rlox-v7.log 2>&1

echo "=== v7: Targeted rlox PPO re-run (5 experiments) ==="
echo "=== Both fixes: truncation bootstrap + .close() ==="

apt-get update && apt-get install -y docker.io

cd /tmp
git clone https://github.com/riserally/rlox.git
cd /tmp/rlox

# Build Docker image
cat > /tmp/Dockerfile.v7 << 'DOCKERFILE'
FROM python:3.12-slim

RUN apt-get update && apt-get install -y gcc g++ curl pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir maturin numpy gymnasium torch pyyaml "gymnasium[mujoco]"

COPY . /app
WORKDIR /app
RUN maturin develop --release
DOCKERFILE

docker build -t rlox-v7 -f /tmp/Dockerfile.v7 /tmp/rlox

echo "=== Running 5 rlox PPO experiments ==="
mkdir -p /tmp/rlox/results/convergence

docker run --rm -v /tmp/rlox/results:/app/results rlox-v7 python -c "
import json, time, numpy as np, torch, gymnasium as gym
from rlox import Trainer

EXPERIMENTS = [
    ('CartPole-v1', 100_000, {'n_envs': 8, 'n_steps': 128}),
    ('Acrobot-v1', 500_000, {'n_envs': 8, 'n_steps': 128}),
    ('HalfCheetah-v4', 1_000_000, {'n_envs': 8, 'n_steps': 2048, 'n_epochs': 10, 'batch_size': 64, 'learning_rate': 3e-4, 'normalize_obs': True, 'normalize_rewards': True}),
    ('Hopper-v4', 1_000_000, {'n_envs': 8, 'n_steps': 2048, 'n_epochs': 10, 'batch_size': 64, 'learning_rate': 3e-4, 'normalize_obs': True, 'normalize_rewards': True}),
    ('Walker2d-v4', 2_000_000, {'n_envs': 8, 'n_steps': 2048, 'n_epochs': 10, 'batch_size': 64, 'learning_rate': 3e-4, 'normalize_obs': True, 'normalize_rewards': True}),
]

for env_id, total_steps, config in EXPERIMENTS:
    print(f'\\n{\"=\"*60}')
    print(f'Running: rlox PPO on {env_id}, {total_steps} steps')
    print(f'{\"=\"*60}')

    t0 = time.time()
    trainer = Trainer('ppo', env=env_id, seed=42, config=config)
    metrics = trainer.train(total_timesteps=total_steps)
    elapsed = time.time() - t0
    sps = total_steps / elapsed

    # Evaluate
    eval_env = gym.make(env_id)
    vn = getattr(trainer.algo, 'vec_normalize', None)
    is_discrete = hasattr(eval_env.action_space, 'n')
    rewards = []
    for _ in range(30):
        obs, _ = eval_env.reset()
        ep_r = 0; done = False
        while not done:
            if vn: obs = vn.normalize_obs(obs.reshape(1,-1)).flatten()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if is_discrete:
                    action = trainer.algo.policy.actor(obs_t).argmax(-1).item()
                else:
                    action = trainer.algo.policy.actor(obs_t).squeeze(0).numpy()
            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += r; done = term or trunc
        rewards.append(ep_r)
    eval_env.close()

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    print(f'  Result: {mean_r:.1f} +/- {std_r:.1f}, SPS={sps:.0f}, wall={elapsed:.0f}s')

    # Save result
    result = {
        'framework': 'rlox',
        'algorithm': 'PPO',
        'environment': env_id,
        'seed': 42,
        'total_steps': total_steps,
        'config': {k: v for k, v in config.items()},
        'evaluations': [{'step': total_steps, 'mean_return': mean_r, 'std_return': std_r}],
        'wall_time_s': elapsed,
        'sps': sps,
    }
    fname = f'rlox_PPO_{env_id.replace(\"/\", \"_\")}_seed0.json'
    with open(f'/app/results/convergence/{fname}', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {fname}')

print('\\n=== All 5 experiments complete ===')
"

echo "=== Uploading results ==="
set +e
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
gcloud storage cp -r /tmp/rlox/results/convergence/ "gs://rkox-bench-results/convergence-v7-${TIMESTAMP}/" 2>&1 || true
gcloud storage cp /var/log/rlox-v7.log "gs://rkox-bench-results/convergence-v7-${TIMESTAMP}/" 2>&1 || true

echo "=== Done — shutting down ==="
shutdown -h now
