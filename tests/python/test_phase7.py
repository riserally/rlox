"""Phase 7 tests: Python Layer 1+ components, algorithms, ops."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Group 0: Bug-fix regressions (should already pass from Step 0)
# ---------------------------------------------------------------------------


class TestMultiDimActionBuffer:
    def test_push_and_observations(self):
        from rlox import ExperienceTable

        table = ExperienceTable(obs_dim=4, act_dim=2)
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        act = np.array([0.5, 0.6], dtype=np.float32)
        table.push(obs, act, 1.0, False, False)
        assert len(table) == 1
        np.testing.assert_array_equal(table.observations()[0], obs)


class TestTerminalObsPreservation:
    def test_terminal_obs_in_step(self):
        from rlox import VecEnv

        venv = VecEnv(n=4, seed=42)
        # Step many times to trigger at least one terminal
        terminal_seen = False
        for _ in range(500):
            result = venv.step_all([1, 1, 1, 1])
            for i, t in enumerate(result["terminal_obs"]):
                if t is not None:
                    terminal_seen = True
                    assert isinstance(t, np.ndarray)
                    assert t.shape == (4,)
            if terminal_seen:
                break
        assert terminal_seen, "Expected at least one terminal_obs in 500 steps"


class TestTokenKLReturnsResult:
    def test_mismatched_lengths_raises(self):
        from rlox import compute_token_kl

        a = np.array([0.1, 0.2], dtype=np.float64)
        b = np.array([0.1], dtype=np.float64)
        with pytest.raises(RuntimeError):
            compute_token_kl(a, b)


class TestConfigurableVecEnv:
    def test_env_id_cartpole(self):
        from rlox import VecEnv

        venv = VecEnv(n=2, seed=0, env_id="CartPole-v1")
        obs = venv.reset_all()
        assert obs.shape == (2, 4)


# ---------------------------------------------------------------------------
# Group 1: Primitives (should already pass from Step 1)
# ---------------------------------------------------------------------------


class TestRunningStats:
    def test_basic_stats(self):
        from rlox import RunningStats

        rs = RunningStats()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.update(v)
        assert rs.count() == 5
        assert abs(rs.mean() - 3.0) < 1e-10
        assert abs(rs.var() - 2.0) < 1e-10  # population variance = 2.0

    def test_batch_update(self):
        from rlox import RunningStats

        rs = RunningStats()
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        rs.batch_update(vals)
        assert rs.count() == 5
        assert abs(rs.mean() - 3.0) < 1e-10

    def test_normalize(self):
        from rlox import RunningStats

        rs = RunningStats()
        for v in [10.0, 20.0, 30.0]:
            rs.update(v)
        normed = rs.normalize(20.0)
        assert abs(normed) < 1e-10  # mean is 20, so normalized should be ~0

    def test_reset(self):
        from rlox import RunningStats

        rs = RunningStats()
        rs.update(5.0)
        rs.reset()
        assert rs.count() == 0


class TestSequencePacking:
    def test_basic_packing(self):
        from rlox import pack_sequences

        seqs = [
            np.array([1, 2, 3], dtype=np.uint32),
            np.array([4, 5], dtype=np.uint32),
            np.array([6], dtype=np.uint32),
        ]
        bins = pack_sequences(seqs, max_length=4)
        assert len(bins) >= 1
        for b in bins:
            assert "input_ids" in b
            assert "attention_mask" in b
            assert "position_ids" in b
            assert "sequence_starts" in b


class TestBatchAssembly:
    def test_rollout_batch_shapes(self):
        from rlox.batch import RolloutBatch

        n = 32
        obs_dim = 4
        batch = RolloutBatch(
            obs=torch.randn(n, obs_dim),
            actions=torch.randint(0, 2, (n,)),
            rewards=torch.randn(n),
            dones=torch.zeros(n),
            log_probs=torch.randn(n),
            values=torch.randn(n),
            advantages=torch.randn(n),
            returns=torch.randn(n),
        )
        assert batch.obs.shape == (n, obs_dim)
        assert batch.returns.shape == (n,)

    def test_minibatch_iteration(self):
        from rlox.batch import RolloutBatch

        n = 64
        batch = RolloutBatch(
            obs=torch.randn(n, 4),
            actions=torch.randint(0, 2, (n,)),
            rewards=torch.randn(n),
            dones=torch.zeros(n),
            log_probs=torch.randn(n),
            values=torch.randn(n),
            advantages=torch.randn(n),
            returns=torch.randn(n),
        )
        minibatches = list(batch.sample_minibatches(batch_size=16, shuffle=False))
        assert len(minibatches) == 4
        for mb in minibatches:
            assert mb.obs.shape == (16, 4)

    def test_minibatch_shuffle(self):
        from rlox.batch import RolloutBatch

        n = 64
        batch = RolloutBatch(
            obs=torch.arange(n, dtype=torch.float32).unsqueeze(1),
            actions=torch.zeros(n),
            rewards=torch.zeros(n),
            dones=torch.zeros(n),
            log_probs=torch.zeros(n),
            values=torch.zeros(n),
            advantages=torch.zeros(n),
            returns=torch.zeros(n),
        )
        # With shuffle, the order should differ (not guaranteed but extremely likely)
        torch.manual_seed(42)
        mbs = list(batch.sample_minibatches(batch_size=16, shuffle=True))
        all_obs = torch.cat([mb.obs for mb in mbs], dim=0).squeeze()
        # Check all indices are present (permutation)
        assert set(all_obs.long().tolist()) == set(range(n))

    def test_minibatch_drops_remainder(self):
        from rlox.batch import RolloutBatch

        n = 50
        batch = RolloutBatch(
            obs=torch.randn(n, 4),
            actions=torch.zeros(n),
            rewards=torch.zeros(n),
            dones=torch.zeros(n),
            log_probs=torch.zeros(n),
            values=torch.zeros(n),
            advantages=torch.zeros(n),
            returns=torch.zeros(n),
        )
        mbs = list(batch.sample_minibatches(batch_size=16, shuffle=False))
        assert len(mbs) == 3  # 48 / 16 = 3, drops 2


# ---------------------------------------------------------------------------
# Group 2: Collectors
# ---------------------------------------------------------------------------


class _DummyDiscretePolicy(nn.Module):
    """Minimal policy for testing the collector."""

    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.fc = nn.Linear(obs_dim, n_actions)
        self.value_head = nn.Linear(obs_dim, 1)

    def get_action_and_logprob(self, obs: torch.Tensor):
        logits = self.fc(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, obs: torch.Tensor):
        return self.value_head(obs).squeeze(-1)

    def get_logprob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.fc(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class TestRolloutCollector:
    def test_collect_shapes(self):
        from rlox.collectors import RolloutCollector

        n_envs = 4
        n_steps = 16
        collector = RolloutCollector(env_id="CartPole-v1", n_envs=n_envs, seed=42)
        policy = _DummyDiscretePolicy()
        batch = collector.collect(policy, n_steps=n_steps)

        total = n_envs * n_steps
        assert batch.obs.shape == (total, 4)
        assert batch.actions.shape == (total,)
        assert batch.rewards.shape == (total,)
        assert batch.dones.shape == (total,)
        assert batch.log_probs.shape == (total,)
        assert batch.values.shape == (total,)
        assert batch.advantages.shape == (total,)
        assert batch.returns.shape == (total,)

    def test_collect_advantages_nonzero(self):
        from rlox.collectors import RolloutCollector

        collector = RolloutCollector(env_id="CartPole-v1", n_envs=2, seed=0)
        policy = _DummyDiscretePolicy()
        batch = collector.collect(policy, n_steps=32)
        # Advantages should not all be zero
        assert batch.advantages.abs().sum().item() > 0

    def test_collect_returns_finite(self):
        from rlox.collectors import RolloutCollector

        collector = RolloutCollector(env_id="CartPole-v1", n_envs=2, seed=0)
        policy = _DummyDiscretePolicy()
        batch = collector.collect(policy, n_steps=32)
        assert torch.isfinite(batch.returns).all()
        assert torch.isfinite(batch.advantages).all()


# ---------------------------------------------------------------------------
# Group 3: Algorithms
# ---------------------------------------------------------------------------


class TestPPOLoss:
    def test_loss_computation(self):
        from rlox.losses import PPOLoss

        loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
        policy = _DummyDiscretePolicy()
        n = 32
        obs = torch.randn(n, 4)
        actions = torch.randint(0, 2, (n,))
        old_log_probs = torch.randn(n)
        advantages = torch.randn(n)
        returns = torch.randn(n)
        old_values = torch.randn(n)

        total_loss, metrics = loss_fn(
            policy, obs, actions, old_log_probs, advantages, returns, old_values
        )
        assert total_loss.shape == ()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics

    def test_zero_advantage_gives_near_zero_policy_loss(self):
        from rlox.losses import PPOLoss

        loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.0, ent_coef=0.0)
        policy = _DummyDiscretePolicy()
        n = 32
        obs = torch.randn(n, 4)
        actions = torch.randint(0, 2, (n,))
        # Get fresh log probs so ratio ≈ 1
        with torch.no_grad():
            log_probs, _ = policy.get_logprob_and_entropy(obs, actions)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)
        old_values = torch.zeros(n)

        total_loss, metrics = loss_fn(
            policy, obs, actions, log_probs, advantages, returns, old_values
        )
        # With zero advantages, policy loss should be ~0
        assert abs(metrics["policy_loss"]) < 1e-5

    def test_clipped_value_loss(self):
        from rlox.losses import PPOLoss

        loss_fn = PPOLoss(clip_eps=0.2, vf_coef=1.0, ent_coef=0.0, clip_vloss=True)
        policy = _DummyDiscretePolicy()
        n = 16
        obs = torch.randn(n, 4)
        actions = torch.randint(0, 2, (n,))
        with torch.no_grad():
            log_probs, _ = policy.get_logprob_and_entropy(obs, actions)
        advantages = torch.zeros(n)
        returns = torch.ones(n) * 10.0  # large returns
        old_values = torch.zeros(n)

        total_loss, metrics = loss_fn(
            policy, obs, actions, log_probs, advantages, returns, old_values
        )
        assert metrics["value_loss"] > 0


class TestPPOEndToEnd:
    def test_importable(self):
        from rlox.algorithms.ppo import PPO, PPOConfig

        assert PPOConfig is not None
        assert PPO is not None

    def test_smoke(self):
        from rlox.algorithms.ppo import PPO

        ppo = PPO(env_id="CartPole-v1", n_envs=2, seed=42, n_steps=16, n_epochs=1, batch_size=16)
        metrics = ppo.train(total_timesteps=64)
        assert "mean_reward" in metrics
        assert "policy_loss" in metrics

    @pytest.mark.slow
    def test_cartpole_solves(self):
        from rlox.algorithms.ppo import PPO

        ppo = PPO(env_id="CartPole-v1", n_envs=8, seed=42)
        metrics = ppo.train(total_timesteps=50_000)
        assert metrics["mean_reward"] > 100


class TestA2CEndToEnd:
    def test_importable(self):
        from rlox.algorithms.a2c import A2C

        assert A2C is not None

    def test_smoke(self):
        from rlox.algorithms.a2c import A2C

        a2c = A2C(env_id="CartPole-v1", n_envs=2, seed=42, n_steps=8)
        metrics = a2c.train(total_timesteps=32)
        assert "mean_reward" in metrics
        assert "policy_loss" in metrics


class _MockLanguageModel(nn.Module):
    """Mock LM for testing GRPO/DPO without a real transformer."""

    def __init__(self, vocab_size=100, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, seq_len, vocab_size)."""
        h = self.embed(input_ids)
        return self.head(h)

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 8) -> torch.Tensor:
        """Greedy generate max_new_tokens tokens."""
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class TestGRPOEndToEnd:
    def test_importable(self):
        from rlox.algorithms.grpo import GRPO

        assert GRPO is not None

    def test_smoke(self):
        from rlox.algorithms.grpo import GRPO

        model = _MockLanguageModel()
        ref_model = _MockLanguageModel()

        def reward_fn(completions, prompts):
            return [float(len(c)) for c in completions]

        grpo = GRPO(
            model=model,
            ref_model=ref_model,
            reward_fn=reward_fn,
            group_size=2,
            kl_coef=0.1,
        )
        prompts = torch.randint(0, 50, (4, 5))  # 4 prompts, length 5
        metrics = grpo.train_step(prompts)
        assert "loss" in metrics
        assert "mean_reward" in metrics
        assert "kl" in metrics


class TestDPOEndToEnd:
    def test_importable(self):
        from rlox.algorithms.dpo import DPO

        assert DPO is not None

    def test_smoke(self):
        from rlox.algorithms.dpo import DPO

        model = _MockLanguageModel()
        ref_model = _MockLanguageModel()

        dpo = DPO(model=model, ref_model=ref_model, beta=0.1)
        prompt = torch.randint(0, 50, (2, 5))
        chosen = torch.randint(0, 50, (2, 10))
        rejected = torch.randint(0, 50, (2, 10))

        loss, metrics = dpo.compute_loss(prompt, chosen, rejected)
        assert loss.shape == ()
        assert "loss" in metrics
        assert "chosen_reward" in metrics
        assert "rejected_reward" in metrics


# ---------------------------------------------------------------------------
# Group 4: Ops
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    def test_save_load_smoke(self, tmp_path):
        from rlox.checkpoint import Checkpoint

        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "ckpt.pt"

        Checkpoint.save(
            str(path), model=model, optimizer=optimizer, step=100, config={"lr": 1e-3}
        )
        data = Checkpoint.load(str(path))
        assert data["step"] == 100
        assert data["config"] == {"lr": 1e-3}
        assert "model_state_dict" in data
        assert "optimizer_state_dict" in data

    def test_missing_required_fields(self, tmp_path):
        from rlox.checkpoint import Checkpoint

        model = nn.Linear(4, 2)
        path = tmp_path / "ckpt.pt"
        with pytest.raises(TypeError):
            Checkpoint.save(str(path), model=model)  # missing optimizer, step, config


class TestLogging:
    def test_logger_callback_interface(self):
        from rlox.logging import LoggerCallback

        cb = LoggerCallback()
        # Should not raise
        cb.on_train_step(0, {"loss": 0.5})
        cb.on_rollout_end(0, {"reward": 1.0})
        cb.on_eval(0, {"accuracy": 0.9})

    def test_custom_callback_receives_metrics(self):
        from rlox.logging import LoggerCallback

        received = []

        class MyLogger(LoggerCallback):
            def on_train_step(self, step, metrics):
                received.append(("train", step, metrics))

        logger = MyLogger()
        logger.on_train_step(1, {"loss": 0.3})
        assert len(received) == 1
        assert received[0] == ("train", 1, {"loss": 0.3})


class TestWheelInstallation:
    def test_all_public_symbols_importable(self):
        import rlox

        # Core Rust primitives
        for name in [
            "CartPole", "VecEnv", "GymEnv", "ExperienceTable", "ReplayBuffer",
            "VarLenStore", "compute_gae", "compute_group_advantages",
            "compute_token_kl", "DPOPair", "RunningStats", "pack_sequences",
        ]:
            assert hasattr(rlox, name), f"Missing: {name}"

        # Python Layer 1
        for name in ["RolloutBatch", "RolloutCollector", "PPOLoss"]:
            assert hasattr(rlox, name), f"Missing: {name}"

        # Algorithms
        from rlox.algorithms import PPO, A2C, GRPO, DPO
        for cls in [PPO, A2C, GRPO, DPO]:
            assert cls is not None

        # Ops
        from rlox.checkpoint import Checkpoint
        from rlox.logging import LoggerCallback
        assert Checkpoint is not None
        assert LoggerCallback is not None


# ---------------------------------------------------------------------------
# Step 4: Reward/Obs Normalization
# ---------------------------------------------------------------------------


class TestRewardNormalization:
    def test_ppo_with_normalize_rewards(self):
        from rlox.algorithms.ppo import PPO

        ppo = PPO(
            env_id="CartPole-v1", n_envs=2, seed=42,
            n_steps=16, n_epochs=1, batch_size=16,
            normalize_rewards=True,
        )
        metrics = ppo.train(total_timesteps=64)
        assert "mean_reward" in metrics

    def test_ppo_with_normalize_obs(self):
        from rlox.algorithms.ppo import PPO

        ppo = PPO(
            env_id="CartPole-v1", n_envs=2, seed=42,
            n_steps=16, n_epochs=1, batch_size=16,
            normalize_obs=True,
        )
        metrics = ppo.train(total_timesteps=64)
        assert "mean_reward" in metrics
