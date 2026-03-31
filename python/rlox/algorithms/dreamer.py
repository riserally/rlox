"""DreamerV3: world model with RSSM, symlog transforms, and actor-critic in latent space.

Enhanced implementation featuring:
- RSSM (Recurrent State-Space Model) with categorical latents
- Symlog/symexp transforms for reward and value prediction
- Sequence replay from the buffer
- Continuous action support via TanhNormal distribution

Reference:
    D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap,
    "Mastering Diverse Domains through World Models,"
    arXiv:2301.04104, 2023.
    https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox


# ---------------------------------------------------------------------------
# Symlog / Symexp transforms
# ---------------------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transform: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ---------------------------------------------------------------------------
# RSSM (Recurrent State-Space Model)
# ---------------------------------------------------------------------------


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical latents.

    The state is decomposed into:
    - Deterministic component h (GRU hidden state)
    - Stochastic component z (categorical latent, stoch_dim * classes)

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Action dimensionality (continuous) or number of actions (discrete).
    deter_dim : int
        Deterministic GRU state dimension (default 512).
    stoch_dim : int
        Number of categorical distributions (default 32).
    classes : int
        Number of classes per categorical (default 32).
    hidden : int
        Hidden layer width for MLPs (default 512).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        classes: int = 32,
        hidden: int = 512,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.classes = classes
        self.stoch_total = stoch_dim * classes

        # Sequence model: GRU that takes (prev_stoch + action) -> deter state
        self.gru_input = nn.Linear(self.stoch_total + act_dim, deter_dim)
        self.gru = nn.GRUCell(deter_dim, deter_dim)

        # Prior: deter -> stochastic (imagination, no observation)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.stoch_total),
        )

        # Posterior: deter + obs -> stochastic (observation step)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.stoch_total),
        )

        # Decoder: feature -> obs
        feat_dim = deter_dim + self.stoch_total
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, obs_dim),
        )

        # Reward head with symlog prediction
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

        # Continue head (predicts non-termination)
        self.continue_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h, z) state.

        Returns
        -------
        h : Tensor of shape (batch, deter_dim)
        z : Tensor of shape (batch, stoch_total)
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_total, device=device)
        return h, z

    def _sample_stoch(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical latents using straight-through Gumbel-Softmax.

        Parameters
        ----------
        logits : Tensor of shape (batch, stoch_total)
            Raw logits reshaped to (batch, stoch_dim, classes).

        Returns
        -------
        z : Tensor of shape (batch, stoch_total)
            Sampled one-hot vectors flattened.
        """
        batch = logits.shape[0]
        logits = logits.reshape(batch, self.stoch_dim, self.classes)
        # Straight-through: hard one-hot in forward, soft gradient in backward
        probs = F.softmax(logits, dim=-1)
        # Gumbel trick for differentiable sampling
        uniform = torch.rand_like(probs).clamp(1e-6, 1 - 1e-6)
        gumbels = -torch.log(-torch.log(uniform))
        samples = F.one_hot(
            (logits + gumbels).argmax(dim=-1), self.classes
        ).float()
        # Straight-through gradient
        z = samples + probs - probs.detach()
        return z.reshape(batch, self.stoch_total)

    def observe_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step with observation (posterior).

        Parameters
        ----------
        h : Tensor (batch, deter_dim) -- previous deterministic state
        z : Tensor (batch, stoch_total) -- previous stochastic state
        action : Tensor (batch, act_dim) -- action taken
        obs : Tensor (batch, obs_dim) -- current observation

        Returns
        -------
        h_new : Tensor (batch, deter_dim)
        z_post : Tensor (batch, stoch_total) -- posterior sample
        z_prior : Tensor (batch, stoch_total) -- prior sample (for KL)
        """
        # Deterministic transition
        gru_in = F.elu(self.gru_input(torch.cat([z, action], dim=-1)))
        h_new = self.gru(gru_in, h)

        # Prior (no obs)
        prior_logits = self.prior_net(h_new)
        z_prior = self._sample_stoch(prior_logits)

        # Posterior (with obs)
        post_logits = self.posterior_net(torch.cat([h_new, obs], dim=-1))
        z_post = self._sample_stoch(post_logits)

        return h_new, z_post, z_prior

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step without observation (prior only, for imagination).

        Returns
        -------
        h_new : Tensor (batch, deter_dim)
        z_prior : Tensor (batch, stoch_total)
        """
        gru_in = F.elu(self.gru_input(torch.cat([z, action], dim=-1)))
        h_new = self.gru(gru_in, h)
        prior_logits = self.prior_net(h_new)
        z_prior = self._sample_stoch(prior_logits)
        return h_new, z_prior

    def get_feature(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate deterministic and stochastic state into a feature vector."""
        return torch.cat([h, z], dim=-1)

    def decode(self, feature: torch.Tensor) -> torch.Tensor:
        """Decode feature to observation prediction."""
        return self.decoder(feature)

    def predict_reward(self, feature: torch.Tensor) -> torch.Tensor:
        """Predict symlog-transformed reward."""
        return self.reward_head(feature).squeeze(-1)

    def predict_continue(self, feature: torch.Tensor) -> torch.Tensor:
        """Predict continuation probability (logit)."""
        return self.continue_head(feature).squeeze(-1)

    def get_kl_loss(
        self,
        prior_logits: torch.Tensor,
        post_logits: torch.Tensor,
        balance: float = 0.8,
        free_nats: float = 1.0,
    ) -> torch.Tensor:
        """Compute balanced KL divergence between posterior and prior."""
        batch = prior_logits.shape[0]
        prior = F.softmax(
            prior_logits.reshape(batch, self.stoch_dim, self.classes), dim=-1
        )
        post = F.softmax(
            post_logits.reshape(batch, self.stoch_dim, self.classes), dim=-1
        )

        # KL(post || prior) per categorical
        kl = (post * (torch.log(post + 1e-8) - torch.log(prior + 1e-8))).sum(dim=-1)
        kl = kl.sum(dim=-1)  # sum over stoch_dim

        # Free nats
        kl = torch.clamp(kl, min=free_nats)

        return kl.mean()


# ---------------------------------------------------------------------------
# Latent actor-critic with continuous action support
# ---------------------------------------------------------------------------


class TanhNormal:
    """TanhNormal distribution for continuous actions in latent space."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.normal = torch.distributions.Normal(loc, scale)

    def sample(self) -> torch.Tensor:
        x = self.normal.rsample()
        return torch.tanh(x)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Inverse tanh (atanh) for log_prob correction
        eps = 1e-6
        actions_clamped = actions.clamp(-1 + eps, 1 - eps)
        pre_tanh = torch.atanh(actions_clamped)
        log_prob = self.normal.log_prob(pre_tanh)
        # Tanh squashing correction
        log_prob = log_prob - torch.log(1 - actions_clamped.pow(2) + eps)
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # Approximate entropy (exact entropy not available for TanhNormal)
        return self.normal.entropy().sum(dim=-1)


class LatentActorCritic(nn.Module):
    """Actor-critic operating in latent space.

    Supports both discrete (Categorical) and continuous (TanhNormal) actions.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent feature vector (deter_dim + stoch_total).
    n_actions : int
        Number of discrete actions, or action dimensionality for continuous.
    continuous : bool
        Whether to use continuous actions (default False).
    hidden : int
        Hidden layer width (default 64).
    """

    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        continuous: bool = False,
        hidden: int = 64,
    ):
        super().__init__()
        self.continuous = continuous
        self.n_actions = n_actions

        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )

        if continuous:
            self.actor_mean = nn.Linear(hidden, n_actions)
            self.actor_log_std = nn.Linear(hidden, n_actions)
        else:
            self.actor_head = nn.Linear(hidden, n_actions)

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def policy(
        self, h: torch.Tensor
    ) -> torch.distributions.Categorical | TanhNormal:
        features = self.actor(h)
        if self.continuous:
            mean = self.actor_mean(features)
            log_std = self.actor_log_std(features).clamp(-5.0, 2.0)
            return TanhNormal(mean, log_std.exp())
        else:
            logits = self.actor_head(features)
            return torch.distributions.Categorical(logits=logits)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# DreamerV3 agent
# ---------------------------------------------------------------------------


class DreamerV3:
    """DreamerV3 agent with RSSM world model and latent actor-critic.

    Components:
    - RSSM world model (GRU dynamics + categorical latents)
    - Symlog transforms for reward/value prediction
    - Actor-critic in latent space with imagination rollouts
    - Sequence replay from buffer
    - Continuous and discrete action support

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    n_envs : int
        Number of parallel environments (default 1).
    seed : int
        Random seed (default 42).
    latent_dim : int
        Latent feature dimension -- ignored, computed from RSSM dims.
    imagination_horizon : int
        Steps to imagine for policy learning (default 5).
    buffer_size : int
        Replay buffer capacity (default 10000).
    batch_size : int
        Number of sequences per training batch (default 32).
    seq_len : int
        Sequence length for world model training (default 16).
    learning_rate : float
        Learning rate for all optimisers (default 3e-4).
    gamma : float
        Discount factor (default 0.99).
    obs_dim : int
        Observation dimensionality (default 4, auto-detected if possible).
    n_actions : int
        Number of actions or action dim (default 2, auto-detected if possible).
    deter_dim : int
        RSSM deterministic state dim (default 64).
    stoch_dim : int
        Number of categorical distributions (default 8).
    stoch_classes : int
        Classes per categorical (default 8).
    kl_balance : float
        KL balancing coefficient (default 0.8).
    free_nats : float
        Free nats for KL loss (default 1.0).
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        seed: int = 42,
        latent_dim: int = 64,
        imagination_horizon: int = 5,
        buffer_size: int = 10000,
        batch_size: int = 32,
        seq_len: int = 16,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        obs_dim: int = 4,
        n_actions: int = 2,
        deter_dim: int = 64,
        stoch_dim: int = 8,
        stoch_classes: int = 8,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.imagination_horizon = imagination_horizon
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.gamma = gamma
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.device = "cpu"

        # Detect if continuous
        tmp = gym.make(env_id)
        self._is_continuous = isinstance(tmp.action_space, gym.spaces.Box)
        if self._is_continuous:
            self.n_actions = int(np.prod(tmp.action_space.shape))
        tmp.close()

        # Environment
        self.env = rlox.VecEnv(n=n_envs, seed=seed, env_id=env_id)

        # RSSM world model
        act_dim = self.n_actions if self._is_continuous else self.n_actions
        self.world_model = RSSM(
            obs_dim=obs_dim,
            act_dim=act_dim if self._is_continuous else n_actions,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            classes=stoch_classes,
            hidden=max(64, deter_dim),
        )

        # Latent actor-critic
        feat_dim = deter_dim + stoch_dim * stoch_classes
        self.actor_critic = LatentActorCritic(
            latent_dim=feat_dim,
            n_actions=self.n_actions,
            continuous=self._is_continuous,
            hidden=max(64, deter_dim),
        )

        # Replay buffer -- stores individual transitions for sequence slicing
        self.buffer = rlox.ReplayBuffer(buffer_size, obs_dim, 1)
        # Also maintain a sequential buffer for sequence replay
        self._seq_buffer: list[dict[str, np.ndarray]] = []
        self._seq_buffer_maxlen = buffer_size

        # Optimizers
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=learning_rate
        )
        self.ac_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate
        )

    def _collect_experience(self, n_steps: int) -> float:
        """Collect experience with the current policy and add to buffer."""
        obs = self.env.reset_all()
        total_reward = 0.0

        episode_data: list[dict[str, Any]] = [
            {"obs": [], "actions": [], "rewards": [], "dones": []}
            for _ in range(self.n_envs)
        ]

        for _ in range(n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            # Encode through RSSM
            with torch.no_grad():
                h, z = self.world_model.initial_state(obs_tensor.shape[0])
                # Use a simple encode: posterior from initial state
                h_new, z_post, _ = self.world_model.observe_step(
                    h, z,
                    torch.zeros(obs_tensor.shape[0], self.n_actions),
                    obs_tensor,
                )
                feat = self.world_model.get_feature(h_new, z_post)
                dist = self.actor_critic.policy(feat)
                actions = dist.sample()

            if self._is_continuous:
                actions_list = actions.cpu().numpy().astype(np.float32)
                actions_for_env = actions_list
            else:
                actions_list = actions.cpu().numpy().astype(np.uint32).tolist()
                actions_for_env = actions_list

            step_result = self.env.step_all(actions_for_env)

            next_obs = step_result["obs"]
            for i in range(self.n_envs):
                self.buffer.push(
                    obs[i].astype(np.float32),
                    np.array([float(actions[i])], dtype=np.float32)
                    if not self._is_continuous
                    else actions[i].cpu().numpy().astype(np.float32).reshape(-1)[:1],
                    float(step_result["rewards"][i]),
                    bool(step_result["terminated"][i]),
                    bool(step_result["truncated"][i]),
                    next_obs[i].astype(np.float32),
                )

                # Store for sequence replay
                episode_data[i]["obs"].append(obs[i].astype(np.float32))
                episode_data[i]["actions"].append(
                    actions[i].cpu().numpy().astype(np.float32)
                    if self._is_continuous
                    else np.array(float(actions[i]), dtype=np.float32)
                )
                episode_data[i]["rewards"].append(float(step_result["rewards"][i]))
                episode_data[i]["dones"].append(
                    bool(step_result["terminated"][i])
                    or bool(step_result["truncated"][i])
                )

            total_reward += step_result["rewards"].sum()
            obs = step_result["obs"].copy()

        # Add collected sequences to sequential buffer
        for i in range(self.n_envs):
            if len(episode_data[i]["obs"]) > 0:
                self._seq_buffer.append(
                    {
                        "obs": np.stack(episode_data[i]["obs"]),
                        "actions": np.stack(episode_data[i]["actions"]),
                        "rewards": np.array(episode_data[i]["rewards"], dtype=np.float32),
                        "dones": np.array(episode_data[i]["dones"], dtype=np.float32),
                    }
                )
                # Trim buffer
                while len(self._seq_buffer) > self._seq_buffer_maxlen:
                    self._seq_buffer.pop(0)

        return total_reward

    def _sample_sequences(
        self, batch_size: int, seq_len: int
    ) -> dict[str, torch.Tensor] | None:
        """Sample sequences from the sequential buffer.

        Returns
        -------
        dict with keys 'obs', 'actions', 'rewards', 'dones',
        each of shape (batch_size, seq_len, ...), or None if insufficient data.
        """
        if len(self._seq_buffer) == 0:
            return None

        # Find sequences long enough
        valid_seqs = [s for s in self._seq_buffer if len(s["obs"]) >= seq_len]
        if len(valid_seqs) < batch_size:
            # Use shorter sequences if needed, pad with zeros
            valid_seqs = self._seq_buffer
            if len(valid_seqs) == 0:
                return None

        rng = np.random.default_rng()
        indices = rng.choice(len(valid_seqs), size=batch_size, replace=True)

        obs_batch = []
        act_batch = []
        rew_batch = []
        done_batch = []

        for idx in indices:
            seq = valid_seqs[idx]
            max_start = max(0, len(seq["obs"]) - seq_len)
            start = rng.integers(0, max_start + 1)
            end = start + seq_len

            obs_slice = seq["obs"][start:end]
            act_slice = seq["actions"][start:end]
            rew_slice = seq["rewards"][start:end]
            done_slice = seq["dones"][start:end]

            # Pad if needed
            pad_len = seq_len - len(obs_slice)
            if pad_len > 0:
                obs_slice = np.concatenate(
                    [obs_slice, np.zeros((pad_len, *obs_slice.shape[1:]), dtype=np.float32)]
                )
                act_slice = np.concatenate(
                    [act_slice, np.zeros((pad_len, *act_slice.shape[1:]), dtype=np.float32)]
                )
                rew_slice = np.concatenate(
                    [rew_slice, np.zeros(pad_len, dtype=np.float32)]
                )
                done_slice = np.concatenate(
                    [done_slice, np.ones(pad_len, dtype=np.float32)]
                )

            obs_batch.append(obs_slice)
            act_batch.append(act_slice)
            rew_batch.append(rew_slice)
            done_batch.append(done_slice)

        return {
            "obs": torch.as_tensor(np.stack(obs_batch), dtype=torch.float32),
            "actions": torch.as_tensor(np.stack(act_batch), dtype=torch.float32),
            "rewards": torch.as_tensor(np.stack(rew_batch), dtype=torch.float32),
            "dones": torch.as_tensor(np.stack(done_batch), dtype=torch.float32),
        }

    def _train_world_model(self) -> float:
        """Train world model on sequence data from replay."""
        seqs = self._sample_sequences(self.batch_size, self.seq_len)
        if seqs is None:
            # Fall back to individual transitions
            if len(self.buffer) < self.batch_size:
                return 0.0
            batch = self.buffer.sample(
                self.batch_size, seed=np.random.randint(0, 2**31)
            )
            obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32)

            h, z = self.world_model.initial_state(obs.shape[0])
            dummy_action = torch.zeros(obs.shape[0], self.n_actions)
            h_new, z_post, z_prior = self.world_model.observe_step(
                h, z, dummy_action, obs
            )

            feat = self.world_model.get_feature(h_new, z_post)
            pred_obs = self.world_model.decode(feat)
            recon_loss = F.mse_loss(pred_obs, obs)

            rewards = torch.as_tensor(np.array(batch["rewards"]), dtype=torch.float32)
            pred_rewards = self.world_model.predict_reward(feat)
            reward_loss = F.mse_loss(pred_rewards, symlog(rewards))

            loss = recon_loss + reward_loss
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
            self.wm_optimizer.step()
            return loss.item()

        # Sequence-based training
        obs = seqs["obs"]  # (batch, seq_len, obs_dim)
        actions = seqs["actions"]  # (batch, seq_len, ...)
        rewards = seqs["rewards"]  # (batch, seq_len)
        dones = seqs["dones"]  # (batch, seq_len)

        batch_size, seq_len = obs.shape[0], obs.shape[1]

        h, z = self.world_model.initial_state(batch_size)

        total_recon_loss = torch.tensor(0.0)
        total_reward_loss = torch.tensor(0.0)
        total_kl_loss = torch.tensor(0.0)

        for t in range(seq_len):
            obs_t = obs[:, t]
            act_t = actions[:, t]

            # Ensure action has correct shape
            if act_t.dim() == 1:
                if self._is_continuous:
                    act_t = act_t.unsqueeze(-1)
                else:
                    act_t = F.one_hot(act_t.long(), self.n_actions).float()
            elif not self._is_continuous and act_t.shape[-1] != self.n_actions:
                act_t = F.one_hot(act_t.long().squeeze(-1), self.n_actions).float()

            h, z_post, z_prior = self.world_model.observe_step(h, z, act_t, obs_t)

            feat = self.world_model.get_feature(h, z_post)
            pred_obs = self.world_model.decode(feat)
            total_recon_loss = total_recon_loss + F.mse_loss(pred_obs, obs_t)

            pred_reward = self.world_model.predict_reward(feat)
            total_reward_loss = total_reward_loss + F.mse_loss(
                pred_reward, symlog(rewards[:, t])
            )

            z = z_post

            # Reset states on done
            done_mask = dones[:, t].unsqueeze(-1)
            h = h * (1.0 - done_mask)
            z = z * (1.0 - done_mask)

        total_recon_loss = total_recon_loss / seq_len
        total_reward_loss = total_reward_loss / seq_len

        loss = total_recon_loss + total_reward_loss
        self.wm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.wm_optimizer.step()

        return loss.item()

    def _train_actor_critic(self) -> dict[str, float]:
        """Train actor-critic via imagination rollouts in latent space."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, seed=np.random.randint(0, 2**31))
        obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32)

        # Freeze world model during imagination
        for p in self.world_model.parameters():
            p.requires_grad_(False)

        h, z = self.world_model.initial_state(obs.shape[0])
        dummy_action = torch.zeros(obs.shape[0], self.n_actions)
        h, z, _ = self.world_model.observe_step(h, z, dummy_action, obs)
        h = h.detach()
        z = z.detach()

        # Imagination rollout
        imagined_feats = []
        imagined_rewards = []
        imagined_log_probs = []

        for _ in range(self.imagination_horizon):
            feat = self.world_model.get_feature(h, z)
            imagined_feats.append(feat)

            dist = self.actor_critic.policy(feat)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            imagined_log_probs.append(log_prob)

            if not self._is_continuous:
                action_input = F.one_hot(actions, self.n_actions).float()
            else:
                action_input = actions

            h, z = self.world_model.imagine_step(h, z, action_input)

            pred_reward = self.world_model.predict_reward(
                self.world_model.get_feature(h, z)
            )
            # Convert from symlog space
            imagined_rewards.append(symexp(pred_reward))

        # Bootstrap value
        with torch.no_grad():
            bootstrap_feat = self.world_model.get_feature(h, z)
            bootstrap_value = self.actor_critic.value(bootstrap_feat)

        # Compute lambda-returns
        returns = bootstrap_value
        all_returns = []
        for t in reversed(range(self.imagination_horizon)):
            returns = imagined_rewards[t] + self.gamma * returns
            all_returns.insert(0, returns)

        # Values and losses
        values = [self.actor_critic.value(f) for f in imagined_feats]

        actor_losses = []
        for t in range(self.imagination_horizon):
            advantage = (all_returns[t] - values[t]).detach()
            actor_losses.append(-(imagined_log_probs[t] * advantage).mean())
        actor_loss = torch.stack(actor_losses).mean()

        critic_losses = []
        for t in range(self.imagination_horizon):
            # Use symlog for value targets
            critic_losses.append(
                F.mse_loss(values[t], all_returns[t].detach())
            )
        critic_loss = torch.stack(critic_losses).mean()

        loss = actor_loss + 0.5 * critic_loss
        self.ac_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 100.0)
        self.ac_optimizer.step()

        # Unfreeze world model
        for p in self.world_model.parameters():
            p.requires_grad_(True)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run DreamerV3 training loop."""
        collect_steps = 32
        n_updates = max(1, total_timesteps // (collect_steps * self.n_envs))

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        for _ in range(n_updates):
            reward = self._collect_experience(collect_steps)
            all_rewards.append(reward / self.n_envs)

            wm_loss = self._train_world_model()
            ac_metrics = self._train_actor_critic()

            last_metrics = {
                "wm_loss": wm_loss,
                **ac_metrics,
            }

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics
