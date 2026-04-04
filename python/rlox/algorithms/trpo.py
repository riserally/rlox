"""Trust Region Policy Optimization (TRPO) -- KL-constrained natural gradient.

Schulman et al., 2015. Uses conjugate gradient to compute the natural
gradient direction and backtracking line search to find a step size that
satisfies the KL constraint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlox.callbacks import Callback, CallbackList
from rlox.checkpoint import Checkpoint
from rlox.collectors import RolloutCollector
from rlox.config import TRPOConfig
from rlox.gym_vec_env import GymVecEnv
from rlox.policies import DiscretePolicy
from rlox.trainer import register_algorithm
from rlox.utils import detect_env_spaces as _detect_env_spaces


_RUST_NATIVE_ENVS = {"CartPole-v1", "CartPole"}


def _conjugate_gradient(
    Ax_fn,
    b: torch.Tensor,
    cg_iters: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Solve Ax = b using the conjugate gradient algorithm.

    Parameters
    ----------
    Ax_fn : callable
        Function computing the matrix-vector product A @ x.
    b : (N,) tensor
        Right-hand side vector.
    cg_iters : int
        Maximum CG iterations (default 10).
    residual_tol : float
        Convergence tolerance on the residual norm squared.

    Returns
    -------
    x : (N,) tensor -- approximate solution.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)

    for _ in range(cg_iters):
        Ap = Ax_fn(p)
        pAp = p.dot(Ap)
        if pAp.abs() < 1e-12:
            break
        alpha = rdotr / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr

    return x


def _flat_grad(
    loss: torch.Tensor,
    params: list[nn.Parameter],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute flat gradient of loss w.r.t. params."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True,
    )
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p.view(-1)))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat)


def _get_flat_params(params: list[nn.Parameter]) -> torch.Tensor:
    """Concatenate all parameters into a flat vector."""
    return torch.cat([p.view(-1) for p in params])


def _set_flat_params(params: list[nn.Parameter], flat: torch.Tensor) -> None:
    """Set parameters from a flat vector."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset : offset + numel].view_as(p))
        offset += numel


@register_algorithm("trpo")
class TRPO:
    """Trust Region Policy Optimization -- KL-constrained natural gradient.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    seed : int
        Random seed (default 42).
    **config_kwargs
        Override any TRPOConfig fields.
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 42,
        logger: Any | None = None,
        callbacks: list[Callback] | None = None,
        **config_kwargs: Any,
    ):
        self.env_id = env_id
        self.seed = seed

        cfg_fields = {f.name for f in TRPOConfig.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in config_kwargs.items() if k in cfg_fields}
        self.config = TRPOConfig(**cfg_dict)

        self.device = "cpu"

        # Detect environment spaces
        obs_dim, action_space, is_discrete = _detect_env_spaces(env_id)
        self._obs_dim = obs_dim
        self._is_discrete = is_discrete

        # Policy
        if is_discrete:
            n_actions = int(action_space.n)
            self.policy = DiscretePolicy(obs_dim=obs_dim, n_actions=n_actions)
        else:
            from rlox.policies import ContinuousPolicy
            act_dim = int(np.prod(action_space.shape))
            self.policy = ContinuousPolicy(obs_dim=obs_dim, act_dim=act_dim)

        # Separate value function optimizer (SGD, not natural gradient)
        self.vf_optimizer = torch.optim.Adam(
            self.policy.critic.parameters() if hasattr(self.policy, "critic") else self.policy.parameters(),
            lr=self.config.vf_lr,
        )

        # Build environment
        import rlox as _rlox
        if env_id in _RUST_NATIVE_ENVS:
            raw_env = _rlox.VecEnv(n=self.config.n_envs, seed=seed, env_id=env_id)
        else:
            raw_env = GymVecEnv(env_id, n_envs=self.config.n_envs, seed=seed)

        self.collector = RolloutCollector(
            env_id=env_id,
            n_envs=self.config.n_envs,
            seed=seed,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            env=raw_env,
        )

        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self._global_step = 0

    def _get_policy_params(self) -> list[nn.Parameter]:
        """Get only the actor parameters (not value function)."""
        if hasattr(self.policy, "actor"):
            return list(self.policy.actor.parameters())
        if hasattr(self.policy, "logits_net"):
            return list(self.policy.logits_net.parameters())
        return list(self.policy.parameters())

    def _fisher_vector_product(
        self,
        obs: torch.Tensor,
        old_actions: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Fisher-vector product F @ v via Hessian-vector product of KL.

        Uses the identity: F @ v = Hessian(KL) @ v, computed via double backprop.
        KL is computed as KL(pi_old || pi) where pi_old is detached.
        At current params this is zero, but its Hessian (= Fisher) is not.
        """
        params = self._get_policy_params()

        # Compute KL divergence with create_graph=True for double backprop
        log_probs, _ = self.policy.get_logprob_and_entropy(obs, old_actions)
        # KL(pi_old || pi) = E[log_pi_old - log_pi]
        # At current params the detached and non-detached are the same,
        # so KL = 0.  But we need the Hessian w.r.t. the non-detached part.
        kl = (log_probs.detach() - log_probs).mean()

        # First derivative of KL w.r.t. params (with graph retained)
        kl_grad = _flat_grad(kl, params, retain_graph=True, create_graph=True)

        # Hessian-vector product: d/dtheta (grad_KL^T @ v)
        kl_grad_dot_v = kl_grad.dot(vector)
        hvp = _flat_grad(kl_grad_dot_v, params, retain_graph=True)

        return hvp.detach() + self.config.damping * vector

    def _trpo_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> dict[str, float]:
        """Perform a single TRPO policy update step."""
        cfg = self.config
        params = self._get_policy_params()

        # Compute policy gradient
        new_log_probs, entropy = self.policy.get_logprob_and_entropy(obs, actions)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surrogate = (ratio * advantages).mean()

        policy_grad = _flat_grad(surrogate, params, retain_graph=True)

        # If gradient is zero, skip update
        if policy_grad.norm() < 1e-10:
            return {"surrogate_loss": surrogate.item(), "kl": 0.0, "entropy": entropy.mean().item()}

        # Compute natural gradient direction via conjugate gradient
        def fvp(v: torch.Tensor) -> torch.Tensor:
            return self._fisher_vector_product(obs, actions, v)

        step_dir = _conjugate_gradient(fvp, policy_grad, cg_iters=cfg.cg_iters)

        # Compute step size: sqrt(2 * max_kl / (step_dir^T @ F @ step_dir))
        shs = step_dir.dot(fvp(step_dir))
        if shs <= 0:
            lm = 1.0
        else:
            lm = torch.sqrt(2 * cfg.max_kl / (shs + 1e-8)).item()

        full_step = lm * step_dir

        # Backtracking line search
        old_params = _get_flat_params(params)

        step_accepted = False
        for i in range(cfg.line_search_steps):
            fraction = 0.5 ** i
            new_params = old_params + fraction * full_step
            _set_flat_params(params, new_params)

            # Evaluate new surrogate and KL
            with torch.no_grad():
                new_lp, _ = self.policy.get_logprob_and_entropy(obs, actions)
                new_ratio = torch.exp(new_lp - old_log_probs)
                new_surrogate = (new_ratio * advantages).mean()
                improvement = (new_surrogate - surrogate.detach()).item()

            kl = self._compute_kl_at_params(obs, old_log_probs, actions)

            if improvement > 0 and kl <= cfg.max_kl:
                step_accepted = True
                break

        if not step_accepted:
            _set_flat_params(params, old_params)
            kl = 0.0

        final_kl = kl if step_accepted else 0.0
        return {
            "surrogate_loss": surrogate.item(),
            "kl": float(final_kl),
            "entropy": entropy.mean().item(),
        }

    def _compute_kl_at_params(
        self,
        obs: torch.Tensor,
        old_log_probs: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Compute KL divergence between old and current policy (no grad)."""
        with torch.no_grad():
            new_lp, _ = self.policy.get_logprob_and_entropy(obs, actions)
            kl = (old_log_probs - new_lp).mean().item()
        return max(kl, 0.0)  # KL should be non-negative

    def train(self, total_timesteps: int) -> dict[str, float]:
        """Run TRPO training and return final metrics."""
        cfg = self.config
        steps_per_rollout = cfg.n_envs * cfg.n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        all_rewards: list[float] = []
        last_metrics: dict[str, float] = {}

        self.callbacks.on_training_start()

        for update in range(n_updates):
            batch = self.collector.collect(self.policy, n_steps=cfg.n_steps)
            mean_ep_reward = batch.rewards.sum().item() / cfg.n_envs
            all_rewards.append(mean_ep_reward)

            # Normalize advantages
            adv = batch.advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # TRPO policy step
            policy_metrics = self._trpo_step(
                batch.obs, batch.actions, adv, batch.log_probs
            )

            # Value function update via standard SGD
            for _ in range(cfg.vf_epochs):
                values = self.policy.get_value(batch.obs)
                vf_loss = ((values - batch.returns) ** 2).mean()
                self.vf_optimizer.zero_grad(set_to_none=True)
                vf_loss.backward()
                self.vf_optimizer.step()

            last_metrics = {
                **policy_metrics,
                "vf_loss": vf_loss.item(),
                "mean_reward": mean_ep_reward,
            }

            self._global_step += steps_per_rollout

            should_continue = self.callbacks.on_step(
                reward=mean_ep_reward, step=self._global_step, algo=self
            )
            if not should_continue:
                break

        self.callbacks.on_training_end()

        last_metrics["mean_reward"] = (
            float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        )
        return last_metrics

    def save(self, path: str) -> None:
        """Save training checkpoint."""
        Checkpoint.save(
            path,
            model=self.policy,
            optimizer=self.vf_optimizer,
            step=self._global_step,
            config=self.config.to_dict(),
        )

    @classmethod
    def from_checkpoint(cls, path: str, env_id: str | None = None) -> TRPO:
        """Restore TRPO from a checkpoint."""
        data = Checkpoint.load(path)
        config = data["config"]
        eid = env_id or config.get("env_id", "CartPole-v1")
        trpo = cls(env_id=eid, **config)
        trpo.policy.load_state_dict(data["model_state_dict"])
        trpo.vf_optimizer.load_state_dict(data["optimizer_state_dict"])
        trpo._global_step = data.get("step", 0)
        return trpo
