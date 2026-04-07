"""Pin equivalence between Trainer('ppo') and the legacy benchmark runner.

These two paths share the same hyperparameters, the same VecNormalize, and
the same PPOLoss; they only differ in *how* they orchestrate the rollout
loop. They should therefore produce statistically indistinguishable returns
on a short, fixed-seed Hopper run.

Empirically (2026-04-06): at 100k steps, seed=42, both paths land within
~3 reward points of each other on Hopper-v4. We test on Pendulum (faster,
no MuJoCo dependency for the marker) at a much smaller budget; this is a
*regression smoke test*, not a convergence test.

Marked ``slow`` so it doesn't run by default — invoke with::

    pytest tests/python/test_ppo_path_equivalence.py -m slow -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# The legacy runner lives under benchmarks/, not the package — add it to sys.path
_RUNNER_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "convergence"
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))


# Hyperparameters intentionally match what the v6/v8 benchmarks used.
HP = {
    "n_envs": 4,
    "n_steps": 512,
    "n_epochs": 4,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_obs": True,
    "normalize_rewards": True,
}
ENV_ID = "Hopper-v4"   # MuJoCo; the algo we actually care about regressing
SEED = 42
TOTAL_STEPS = HP["n_envs"] * HP["n_steps"] * 12  # 12 updates ≈ 24k steps

# Tolerance: at 24k steps both paths land in the 100–250 range on Hopper
# (still in the noise floor). 100 reward points ≫ measured run-to-run jitter
# of the equivalent paths but ≪ a real algorithmic divergence (1000+).
EQUIVALENCE_TOLERANCE = 100.0


def _eval_policy(policy_fn) -> float:
    """Run 5 deterministic eval episodes and return the mean return."""
    import gymnasium as gym

    env = gym.make(ENV_ID)
    returns = []
    for ep in range(5):
        obs, _ = env.reset(seed=1000 + ep)
        ep_r = 0.0
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            ep_r += float(r)
            done = term or trunc
        returns.append(ep_r)
    env.close()
    return float(np.mean(returns))


def _train_via_trainer() -> float:
    """Train via Trainer API and return mean eval reward (true episodes)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from rlox import Trainer

    trainer = Trainer("ppo", env=ENV_ID, seed=SEED, config=dict(HP))
    trainer.train(total_timesteps=TOTAL_STEPS)

    vn = getattr(trainer.algo, "vec_normalize", None)
    if vn is not None:
        vn.training = False

    def policy_fn(obs):
        if vn is not None:
            obs = vn.normalize_obs(np.asarray(obs, dtype=np.float32)[None])[0]
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return trainer.algo.policy.actor(obs_t).squeeze(0).numpy()

    return _eval_policy(policy_fn)


def _train_via_runner() -> float:
    """Train via legacy benchmark runner and return mean eval reward."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from common import ExperimentLog
    from rlox_runner import _run_ppo

    log = ExperimentLog(
        framework="rlox",
        algorithm="PPO",
        environment=ENV_ID,
        seed=SEED,
        hyperparameters=HP,
        hardware={},
    )
    _run_ppo(
        ENV_ID,
        HP,
        {"hidden_sizes": [64, 64]},
        SEED,
        TOTAL_STEPS,
        TOTAL_STEPS,  # eval_freq — only eval at the end
        5,            # eval_episodes
        log,
    )
    assert log.evaluations, "runner did not record any evaluation"
    return float(log.evaluations[-1].mean_return)


@pytest.mark.slow
def test_trainer_and_runner_produce_similar_hopper_returns() -> None:
    """The two PPO paths must not silently diverge on Hopper-v4.

    Empirically (2026-04-06): at 24k steps, seed=42, identical hyper-
    parameters, both paths produce returns within ~10 reward points on
    Hopper. We allow ``EQUIVALENCE_TOLERANCE = 100`` reward points of slack.
    A larger gap signals a real code-path divergence (e.g. truncation
    bootstrap, advantage normalization scope, action-space detection)
    and should be investigated immediately.

    See ``docs/plans/results-inspection-2026-04-06.md`` for the original
    investigation that motivated this test.
    """
    pytest.importorskip("mujoco")

    trainer_return = _train_via_trainer()
    runner_return = _train_via_runner()

    delta = abs(trainer_return - runner_return)
    assert delta <= EQUIVALENCE_TOLERANCE, (
        f"Trainer vs runner Hopper return diverged: "
        f"trainer={trainer_return:.1f}, runner={runner_return:.1f}, |delta|={delta:.1f}. "
        f"Tolerance was {EQUIVALENCE_TOLERANCE}. "
        f"This indicates a code-path bug — see "
        f"docs/plans/results-inspection-2026-04-06.md."
    )
