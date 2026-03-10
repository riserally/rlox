from typing import Optional
import numpy as np
import numpy.typing as npt

# --- Phase 1: Environment Engine ---

class CartPole:
    """Rust-native CartPole-v1 environment (exact Gymnasium physics)."""
    def __init__(self, seed: Optional[int] = None) -> None: ...
    def step(self, action: int) -> dict[str, object]: ...
    def reset(self, seed: Optional[int] = None) -> npt.NDArray[np.float32]: ...
    def render(self) -> Optional[str]: ...

class VecEnv:
    """Vectorized CartPole environments with Rayon parallel stepping."""
    def __init__(self, n: int, seed: Optional[int] = None) -> None: ...
    def num_envs(self) -> int: ...
    def step_all(self, actions: list[int]) -> dict[str, object]: ...
    def reset_all(self, seed: Optional[int] = None) -> npt.NDArray[np.float32]: ...

class GymEnv:
    """Wrapper for Gymnasium environments, stepped via Python."""
    def __init__(self, env: object) -> None: ...
    def step(self, action: object) -> dict[str, object]: ...
    def reset(self, seed: Optional[int] = None) -> object: ...

# --- Phase 2: Experience Storage ---

class ExperienceTable:
    """Append-only columnar store for RL transitions."""
    def __init__(self, obs_dim: int, act_dim: int) -> None: ...
    def __len__(self) -> int: ...
    def push(
        self,
        obs: npt.NDArray[np.float32],
        action: float,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None: ...
    def observations(self) -> npt.NDArray[np.float32]: ...
    def rewards(self) -> npt.NDArray[np.float32]: ...
    def clear(self) -> None: ...

class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform random sampling."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None: ...
    def __len__(self) -> int: ...
    def push(
        self,
        obs: npt.NDArray[np.float32],
        action: float,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None: ...
    def sample(self, batch_size: int, seed: int) -> dict[str, object]: ...

class VarLenStore:
    """Packed variable-length sequence storage (offsets + flat data)."""
    def __init__(self) -> None: ...
    def push(self, seq: npt.NDArray[np.uint32]) -> None: ...
    def num_sequences(self) -> int: ...
    def total_elements(self) -> int: ...
    def get(self, index: int) -> npt.NDArray[np.uint32]: ...

# --- Phase 3: Training Core ---

def compute_gae(
    rewards: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    dones: npt.NDArray[np.float64],
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Generalized Advantage Estimation (GAE).

    Returns (advantages, returns) as f64 arrays matching input length.
    """
    ...

# --- Phase 4: LLM Post-Training ---

def compute_group_advantages(
    rewards: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """GRPO group-relative advantage: (reward - mean) / std."""
    ...

def compute_token_kl(
    log_probs_policy: npt.NDArray[np.float64],
    log_probs_ref: npt.NDArray[np.float64],
) -> float:
    """Token-level KL divergence: sum(exp(log_p) * (log_p - log_q))."""
    ...

class DPOPair:
    """Direct Preference Optimization preference pair."""
    def __init__(
        self,
        prompt_tokens: npt.NDArray[np.uint32],
        chosen_tokens: npt.NDArray[np.uint32],
        rejected_tokens: npt.NDArray[np.uint32],
    ) -> None: ...
    def chosen_len(self) -> int: ...
    def rejected_len(self) -> int: ...
