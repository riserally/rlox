#!/bin/bash
set -euo pipefail
exec > /var/log/rlox-hopper.log 2>&1

echo "=== Installing Docker ==="
apt-get update && apt-get install -y docker.io

echo "=== Cloning rlox ==="
cd /tmp
git clone https://github.com/riserally/rlox.git
cd /tmp/rlox

echo "=== Building Docker image ==="
cat > /tmp/Dockerfile.hopper << 'DOCKERFILE'
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

docker build -t rlox-hopper -f /tmp/Dockerfile.hopper /tmp/rlox

echo "=== Running PPO Hopper validation ==="
mkdir -p /tmp/rlox/results
docker run --rm -v /tmp/rlox/results:/app/results rlox-hopper python -c "
from rlox import Trainer
import json, time, numpy as np, torch, gymnasium as gym

print('PPO Hopper-v4 with VecNormalize - 1M steps')
t0 = time.time()
trainer = Trainer('ppo', env='Hopper-v4', seed=42, config={
    'n_envs': 8, 'n_steps': 2048, 'normalize_obs': True, 'normalize_rewards': True,
})
trainer.train(total_timesteps=1_000_000)
elapsed = time.time() - t0

env = gym.make('Hopper-v4')
vn = getattr(trainer.algo, 'vec_normalize', None)
rewards = []
for _ in range(30):
    obs, _ = env.reset()
    ep_r = 0; done = False
    while not done:
        if vn: obs = vn.normalize_obs(obs.reshape(1,-1)).flatten()
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): action = trainer.algo.policy.actor(obs_t).squeeze(0).numpy()
        obs, r, term, trunc, _ = env.step(action)
        ep_r += r; done = term or trunc
    rewards.append(ep_r)
env.close()

result = {'mean_return': float(np.mean(rewards)), 'std': float(np.std(rewards)),
          'wall_time': elapsed, 'sps': 1_000_000/elapsed}
print(f'Result: {result[\"mean_return\"]:.1f} +/- {result[\"std\"]:.1f} in {elapsed:.0f}s')
json.dump(result, open('/app/results/ppo_hopper_v6_validation.json', 'w'), indent=2)
"

echo "=== Uploading results ==="
set +e
gcloud storage cp /tmp/rlox/results/ppo_hopper_v6_validation.json gs://rkox-bench-results/convergence-v6-validation/ 2>&1 || true
gcloud storage cp /var/log/rlox-hopper.log gs://rkox-bench-results/convergence-v6-validation/ 2>&1 || true

echo "=== Done — shutting down ==="
shutdown -h now
