# Custom Neural Network Architectures in rlox

## Research Report: Enabling Arbitrary PyTorch Models with Rust-Accelerated Data Pipelines

---

## 1. Current State Assessment

### 1.1 Interface Contracts Per Algorithm

**PPO** requires three methods on the policy (called by `RolloutCollector` and `PPOLoss`):

```python
# RolloutCollector.collect() calls:
policy.get_action_and_logprob(obs: Tensor[B, obs_dim]) -> (actions: Tensor[B, ...], log_probs: Tensor[B])
policy.get_value(obs: Tensor[B, obs_dim]) -> Tensor[B]

# PPOLoss.__call__() calls:
policy.get_logprob_and_entropy(obs: Tensor[B, obs_dim], actions: Tensor[B, ...]) -> (log_probs: Tensor[B], entropy: Tensor[B])
policy.get_value(obs: Tensor[B, obs_dim]) -> Tensor[B]
```

**SAC** requires a `SquashedGaussianPolicy` with:
```python
actor.sample(obs: Tensor[B, obs_dim]) -> (actions: Tensor[B, act_dim], log_probs: Tensor[B])
actor.deterministic(obs: Tensor[B, obs_dim]) -> Tensor[B, act_dim]
```
And twin `QNetwork`s with:
```python
critic(obs: Tensor[B, obs_dim], action: Tensor[B, act_dim]) -> Tensor[B, 1]
```

**DQN** requires a Q-network with:
```python
q_network(obs: Tensor[B, obs_dim]) -> Tensor[B, n_actions]
```

### 1.2 Flexibility Assessment

The current design has several properties relevant to extensibility:

**What works well:**
- PPO already accepts `policy: nn.Module | None` in its constructor (line 67 of `ppo.py`). A user can pass any `nn.Module` that satisfies the three-method contract.
- The contracts are small and well-defined. There is no inheritance requirement.
- The Rust data pipeline (`compute_gae`, `ReplayBuffer`, etc.) is entirely decoupled from the policy network. It operates on numpy arrays and returns numpy arrays. The PyTorch boundary is cleanly at the collector/algorithm level.

**What breaks with non-MLP architectures:**

1. **Observation shape flattening.** `RolloutCollector.collect()` line 215: `obs_t.reshape(total, -1)`. This flattens spatial dimensions. A CNN or ViT receiving `(B, C, H, W)` images will get `(B, C*H*W)` instead. The `RolloutBatch` stores `obs` as `(N, obs_dim)` -- a 2D tensor -- and the entire minibatch sampling code assumes this.

2. **`_detect_env_spaces` flattens obs_dim.** In `ppo.py` line 31: `obs_dim = int(np.prod(tmp.observation_space.shape))`. The auto-created policy receives a scalar integer, not the original shape tuple `(C, H, W)`.

3. **SAC and DQN hardcode network construction.** Neither `SAC.__init__` nor `DQN.__init__` accept a custom network parameter. The networks are always `SquashedGaussianPolicy(obs_dim, act_dim, hidden)` or `SimpleQNetwork(obs_dim, act_dim, hidden)`. To use a custom architecture, users must monkey-patch `self.actor`, `self.critic1`, etc. after construction and re-create optimizers.

4. **No dictionary observation support.** The collector assumes `self._obs` is a single `np.ndarray`. Multi-modal observations (e.g., `{"image": ..., "proprioception": ...}`) are not handled.

5. **No recurrent state management.** For LSTMs/GRUs, the collector must carry hidden states across steps and reset them on episode boundaries. The current `collect()` loop has no mechanism for this.

6. **No observation preprocessing hook.** Frame stacking, grayscale conversion, and channel-first transposition must happen outside the collector, but the collector owns the environment stepping loop.

### 1.3 Summary: What is Required

| Architecture   | Blocked by                              | Severity |
|---------------|------------------------------------------|----------|
| MLP (custom)  | Nothing -- works today via `policy=`     | None     |
| CNN           | Obs flattening in collector/batch        | High     |
| ViT           | Obs flattening + no preprocessing hooks  | High     |
| LSTM/GRU      | No hidden state carry + obs flattening   | High     |
| GNN           | Obs flattening + no graph batching       | High     |
| Foundation    | Works if obs is flat embedding           | Low      |

---

## 2. Proposed Policy Protocols

I recommend `typing.Protocol` over ABCs. Protocols are structural (duck-typed) -- users do not need to inherit from anything, which is the PyTorch community norm. ABCs require inheritance, which creates friction when wrapping third-party models or using `torch.compile`. Documented duck-typing is fragile and produces confusing errors at call time rather than at construction time.

### 2.1 On-Policy Actor-Critic (PPO, A2C)

```python
from typing import Protocol, runtime_checkable
import torch

@runtime_checkable
class ActorCriticPolicy(Protocol):
    """Protocol for on-policy actor-critic networks.

    Any nn.Module implementing these three methods can be used with
    PPO, A2C, and RolloutCollector. No inheritance required.
    """

    def get_action_and_logprob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log-probabilities (inference).

        Parameters
        ----------
        obs : Tensor of shape ``(B, *obs_shape)``
            Batch of observations. Shape depends on the environment:
            ``(B, obs_dim)`` for vector obs, ``(B, C, H, W)`` for images.

        Returns
        -------
        actions : Tensor of shape ``(B,)`` or ``(B, act_dim)``
        log_probs : Tensor of shape ``(B,)``
        """
        ...

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value estimates.

        Returns
        -------
        values : Tensor of shape ``(B,)``
        """
        ...

    def get_logprob_and_entropy(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy for given (obs, action) pairs.

        Must be differentiable (called during training with gradient tracking).

        Returns
        -------
        log_probs : Tensor of shape ``(B,)``
        entropy : Tensor of shape ``(B,)``
        """
        ...
```

### 2.2 Off-Policy Continuous (SAC, TD3)

```python
@runtime_checkable
class StochasticActorProtocol(Protocol):
    """Squashed Gaussian policy for SAC."""

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample with log-prob correction.

        Returns
        -------
        actions : Tensor of shape ``(B, act_dim)`` in [-1, 1]
        log_probs : Tensor of shape ``(B,)``
        """
        ...

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean through squashing).

        Returns
        -------
        actions : Tensor of shape ``(B, act_dim)``
        """
        ...


@runtime_checkable
class ContinuousQFunctionProtocol(Protocol):
    """Q(s, a) network for SAC/TD3."""

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-value.

        Returns
        -------
        q_values : Tensor of shape ``(B, 1)``
        """
        ...
```

### 2.3 Off-Policy Discrete (DQN)

```python
@runtime_checkable
class DiscreteQFunctionProtocol(Protocol):
    """Q-network for DQN variants."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Returns
        -------
        q_values : Tensor of shape ``(B, n_actions)``
        """
        ...
```

### 2.4 Protocol Validation

Add a validation function that runs at algorithm construction time:

```python
def validate_actor_critic(policy: nn.Module, obs_shape: tuple, act_shape: tuple) -> None:
    """Dry-run the policy with dummy tensors to catch shape mismatches early."""
    dummy_obs = torch.randn(2, *obs_shape)

    # Test get_action_and_logprob
    actions, log_probs = policy.get_action_and_logprob(dummy_obs)
    assert log_probs.shape == (2,), f"log_probs shape {log_probs.shape}, expected (2,)"

    # Test get_value
    values = policy.get_value(dummy_obs)
    assert values.shape == (2,), f"values shape {values.shape}, expected (2,)"

    # Test get_logprob_and_entropy
    log_probs2, entropy = policy.get_logprob_and_entropy(dummy_obs, actions)
    assert log_probs2.shape == (2,), f"log_probs shape {log_probs2.shape}, expected (2,)"
    assert entropy.shape == (2,), f"entropy shape {entropy.shape}, expected (2,)"
```

This catches 90% of user errors (wrong output dimension, forgot to squeeze, etc.) immediately at construction time with a clear error message, rather than 5000 steps into training.

---

## 3. Network Builder Pattern

### 3.1 Feature Extractor + Head Architecture

The key insight from SB3 [Raffin et al., 2021] and CleanRL [Huang et al., 2022] is that most RL architectures decompose into:

```
observations --> [feature_extractor] --> latent --> [policy_head] --> actions
                                                --> [value_head]  --> V(s)
```

This decomposition naturally supports custom architectures: users swap the feature extractor while rlox provides standard policy/value heads.

```python
class FeatureExtractor(nn.Module):
    """Base class for feature extractors. Subclass and implement forward()."""

    def __init__(self, features_dim: int):
        super().__init__()
        self.features_dim = features_dim  # output dimensionality

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from observations.

        Parameters
        ----------
        obs : Tensor of shape ``(B, *obs_shape)``

        Returns
        -------
        features : Tensor of shape ``(B, features_dim)``
        """
        raise NotImplementedError
```

### 3.2 Composite Actor-Critic

```python
class CompositeActorCritic(nn.Module):
    """Actor-critic that composes a feature extractor with policy/value heads.

    This is the recommended way to use custom architectures with PPO.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Shared (or separate) feature extractor. Must have ``features_dim`` attribute.
    action_space : gym.Space
        Used to determine head type (Categorical vs Normal).
    shared_extractor : bool
        If True, actor and critic share the same feature extractor.
        If False, the extractor is deep-copied for the critic.
    ortho_init : bool
        Apply orthogonal initialization to heads (default True).

    Example
    -------
    >>> from torchvision.models import resnet18
    >>> backbone = resnet18(pretrained=True)
    >>> backbone.fc = nn.Identity()  # remove classification head
    >>> extractor = PretrainedExtractor(backbone, features_dim=512)
    >>> policy = CompositeActorCritic(extractor, env.action_space)
    >>> ppo = PPO("Breakout-v5", policy=policy)
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        action_space: gym.Space,
        shared_extractor: bool = True,
        ortho_init: bool = True,
    ):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.actor_extractor = feature_extractor

        if shared_extractor:
            self.critic_extractor = feature_extractor  # same object
        else:
            import copy
            self.critic_extractor = copy.deepcopy(feature_extractor)

        feat_dim = feature_extractor.features_dim

        # Build heads based on action space type
        if isinstance(action_space, gym.spaces.Discrete):
            self._discrete = True
            self.actor_head = nn.Linear(feat_dim, int(action_space.n))
            self._build_discrete_dist = True
        else:
            self._discrete = False
            act_dim = int(np.prod(action_space.shape))
            self.actor_head = nn.Linear(feat_dim, act_dim)
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        self.critic_head = nn.Linear(feat_dim, 1)

        if ortho_init:
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.actor_head.bias)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
            nn.init.zeros_(self.critic_head.bias)

    def get_action_and_logprob(self, obs):
        features = self.actor_extractor(obs)
        if self._discrete:
            dist = torch.distributions.Categorical(logits=self.actor_head(features))
            actions = dist.sample()
            return actions, dist.log_prob(actions)
        else:
            mean = self.actor_head(features)
            std = self.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            return actions, dist.log_prob(actions).sum(-1)

    def get_value(self, obs):
        features = self.critic_extractor(obs)
        return self.critic_head(features).squeeze(-1)

    def get_logprob_and_entropy(self, obs, actions):
        features = self.actor_extractor(obs)
        if self._discrete:
            dist = torch.distributions.Categorical(logits=self.actor_head(features))
            return dist.log_prob(actions), dist.entropy()
        else:
            mean = self.actor_head(features)
            std = self.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)
```

### 3.3 Built-in Feature Extractors

```python
class MLPExtractor(FeatureExtractor):
    """Standard MLP feature extractor."""

    def __init__(self, obs_dim: int, hidden_dims: list[int] = (64, 64),
                 activation: type[nn.Module] = nn.Tanh):
        features_dim = hidden_dims[-1] if hidden_dims else obs_dim
        super().__init__(features_dim)
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), activation()])
            in_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class NatureCNNExtractor(FeatureExtractor):
    """CNN from Mnih et al. (2015) 'Human-level control through deep RL'.

    Expects input shape (B, C, 84, 84) where C is the number of stacked frames.
    """

    def __init__(self, n_input_channels: int = 4, features_dim: int = 512):
        super().__init__(features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flat size: 64 * 7 * 7 = 3136 for 84x84 input
        self.linear = nn.Sequential(
            nn.Linear(3136, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))


class PretrainedExtractor(FeatureExtractor):
    """Wrap any pre-trained model as a feature extractor.

    Example
    -------
    >>> import timm
    >>> backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
    >>> extractor = PretrainedExtractor(backbone, features_dim=192)
    """

    def __init__(self, backbone: nn.Module, features_dim: int, freeze: bool = False):
        super().__init__(features_dim)
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, obs):
        return self.backbone(obs)
```

### 3.4 Factory Function

```python
def make_feature_extractor(
    obs_space: gym.Space,
    arch: str = "auto",
    features_dim: int = 256,
    **kwargs,
) -> FeatureExtractor:
    """Create a feature extractor from an observation space.

    Parameters
    ----------
    obs_space : gym.Space
        The environment's observation space.
    arch : str
        Architecture type: "auto", "mlp", "cnn", "nature_cnn".
        "auto" selects based on obs_space shape.
    features_dim : int
        Output feature dimensionality.

    Returns
    -------
    FeatureExtractor
    """
    if arch == "auto":
        if len(obs_space.shape) == 1:
            arch = "mlp"
        elif len(obs_space.shape) == 3:
            arch = "nature_cnn"
        else:
            raise ValueError(f"Cannot auto-detect arch for obs_shape={obs_space.shape}")

    if arch == "mlp":
        obs_dim = int(np.prod(obs_space.shape))
        return MLPExtractor(obs_dim, features_dim=features_dim, **kwargs)
    elif arch == "nature_cnn":
        n_channels = obs_space.shape[0]  # assumes channel-first
        return NatureCNNExtractor(n_channels, features_dim=features_dim)
    else:
        raise ValueError(f"Unknown arch: {arch}. Use 'mlp', 'cnn', or pass a FeatureExtractor.")
```

---

## 4. Concrete Examples

### 4a. Vision Transformer for Atari with PPO

```python
import timm
import torch
import torch.nn as nn
import gymnasium as gym
from rlox.algorithms.ppo import PPO

# -- Feature extractor: ViT-Tiny from timm --
class ViTExtractor(nn.Module):
    """ViT feature extractor for 84x84 Atari frames.

    Uses timm's ViT with patch size 14, resizing observations to 224x224.
    Output: (B, 192) feature vectors.
    """

    def __init__(self, n_input_channels: int = 4, freeze_backbone: bool = False):
        super().__init__()
        self.features_dim = 192

        # ViT expects 3 channels; project stacked frames to 3
        self.channel_proj = nn.Conv2d(n_input_channels, 3, kernel_size=1)

        # Upsample 84x84 -> 224x224 (ViT's expected resolution)
        self.resize = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)

        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, 4, 84, 84) uint8 or float
        x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs
        x = self.channel_proj(x)
        x = self.resize(x)
        return self.vit(x)  # (B, 192)


# -- Composite policy --
class ViTPPOPolicy(nn.Module):
    def __init__(self, n_actions: int, n_input_channels: int = 4):
        super().__init__()
        self.extractor = ViTExtractor(n_input_channels)
        feat_dim = self.extractor.features_dim

        self.actor_head = nn.Linear(feat_dim, n_actions)
        self.critic_head = nn.Linear(feat_dim, 1)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def get_action_and_logprob(self, obs):
        features = self.extractor(obs)
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def get_value(self, obs):
        features = self.extractor(obs)
        return self.critic_head(features).squeeze(-1)

    def get_logprob_and_entropy(self, obs, actions):
        features = self.extractor(obs)
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        return dist.log_prob(actions), dist.entropy()


# -- Usage (requires obs preprocessing -- see Section 6) --
policy = ViTPPOPolicy(n_actions=4, n_input_channels=4)
ppo = PPO("ALE/Breakout-v5", policy=policy, n_envs=8, n_steps=128)
# NOTE: This will not work out of the box today because RolloutCollector
# flattens observations. See Section 7 for the required collector changes.
```

### 4b. LSTM for Partially Observable Environments

```python
class LSTMPolicy(nn.Module):
    """Recurrent actor-critic for POMDPs.

    Maintains hidden state across steps within an episode.
    Hidden state is reset when episodes terminate.

    IMPORTANT: This requires a recurrence-aware collector that:
    1. Carries hidden states across collect() steps
    2. Resets hidden states on episode boundaries
    3. Stores hidden states in the batch for training
    See Section 7 for the RecurrentRolloutCollector proposal.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

        # Hidden state: (h, c), each (1, B, hidden_size)
        self._hidden = None

    def _forward_lstm(self, obs: torch.Tensor, hidden=None):
        # obs: (B, obs_dim) -> (B, 1, obs_dim) for single-step
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        output, hidden = self.lstm(obs, hidden)
        return output.squeeze(1), hidden  # (B, hidden_size), (h, c)

    def init_hidden(self, batch_size: int, device: str = "cpu"):
        return (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device),
        )

    def reset_hidden_for_dones(self, dones: torch.Tensor):
        """Zero out hidden states where episodes terminated."""
        if self._hidden is not None:
            mask = (1.0 - dones).unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
            self._hidden = (self._hidden[0] * mask, self._hidden[1] * mask)

    def get_action_and_logprob(self, obs):
        features, self._hidden = self._forward_lstm(obs, self._hidden)
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def get_value(self, obs):
        features, _ = self._forward_lstm(obs, self._hidden)
        return self.critic_head(features).squeeze(-1)

    def get_logprob_and_entropy(self, obs, actions):
        """For training: process full sequences, not single steps."""
        # During training, obs is (B, seq_len, obs_dim) from the
        # recurrent collector. Process the full sequence.
        features, _ = self._forward_lstm(obs)  # hidden not carried
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        return dist.log_prob(actions), dist.entropy()
```

### 4c. Graph Neural Network for Multi-Agent Coordination

```python
import torch
import torch.nn as nn

# Requires: pip install torch-geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class GNNExtractor(nn.Module):
    """Graph Neural Network feature extractor for multi-agent environments.

    Each observation is a graph where:
    - Nodes represent agents (or entities)
    - Edges represent communication/proximity
    - Node features are local observations

    The extractor produces a fixed-size graph-level embedding.
    """

    def __init__(self, node_feature_dim: int, hidden_dim: int = 64, features_dim: int = 128):
        super().__init__()
        self.features_dim = features_dim

        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, features_dim)

    def forward(self, graph_batch: Batch) -> torch.Tensor:
        """
        Parameters
        ----------
        graph_batch : torch_geometric.data.Batch
            Batched graph with node features and edge indices.

        Returns
        -------
        features : Tensor of shape (B, features_dim)
        """
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # (B, hidden_dim)
        return self.fc(x)


class GNNPolicy(nn.Module):
    """GNN-based actor-critic for graph-structured observations.

    NOTE: This requires a custom collector that converts dict/graph
    observations into torch_geometric Batch objects. The standard
    RolloutCollector cannot handle this. Users should write a custom
    training loop using rlox primitives directly:

        buffer = rlox.ExperienceTable(...)
        gae_fn = rlox.compute_gae
        loss_fn = rlox.PPOLoss(...)
    """

    def __init__(self, node_feature_dim: int, n_actions: int):
        super().__init__()
        self.extractor = GNNExtractor(node_feature_dim)
        self.actor_head = nn.Linear(self.extractor.features_dim, n_actions)
        self.critic_head = nn.Linear(self.extractor.features_dim, 1)

    def get_action_and_logprob(self, graph_batch):
        features = self.extractor(graph_batch)
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def get_value(self, graph_batch):
        features = self.extractor(graph_batch)
        return self.critic_head(features).squeeze(-1)

    def get_logprob_and_entropy(self, graph_batch, actions):
        features = self.extractor(graph_batch)
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        return dist.log_prob(actions), dist.entropy()
```

### 4d. Pre-trained Foundation Model Backbone with Fine-tuning Heads

```python
from transformers import AutoModel
import torch
import torch.nn as nn


class FoundationModelPolicy(nn.Module):
    """Use a pre-trained language model as a state encoder for text-based RL.

    Suitable for environments where observations are text strings
    (e.g., TextWorld, WebShop, ScienceWorld).

    The backbone is frozen; only the policy and value heads are trained.
    This keeps compute tractable and prevents catastrophic forgetting.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        n_actions: int = 10,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # e.g., 768

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Lightweight heads
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] embedding from the backbone."""
        with torch.set_grad_enabled(not self.backbone.training or any(
            p.requires_grad for p in self.backbone.parameters()
        )):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token: (B, hidden_size)

    def get_action_and_logprob(self, obs):
        # obs is a dict: {"input_ids": Tensor, "attention_mask": Tensor}
        features = self._encode(obs["input_ids"], obs["attention_mask"])
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def get_value(self, obs):
        features = self._encode(obs["input_ids"], obs["attention_mask"])
        return self.critic_head(features).squeeze(-1)

    def get_logprob_and_entropy(self, obs, actions):
        features = self._encode(obs["input_ids"], obs["attention_mask"])
        dist = torch.distributions.Categorical(logits=self.actor_head(features))
        return dist.log_prob(actions), dist.entropy()
```

### 4e. CNN Feature Extractor + MLP Policy Head (Mixed Architecture)

```python
class CNNMLPPolicy(nn.Module):
    """CNN encoder with MLP policy and value heads.

    The most common architecture for pixel-based RL (Atari, DMControl).
    Uses the Nature CNN [Mnih et al., 2015] as the feature extractor.

    This is the "batteries included" version that works with the proposed
    image-aware RolloutCollector (Section 7).
    """

    def __init__(self, n_input_channels: int = 4, n_actions: int = 4):
        super().__init__()
        # Nature CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        # Orthogonal init
        for m in self.cnn.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _features(self, obs):
        # obs: (B, C, H, W) float32 in [0, 1]
        return self.cnn(obs)

    def get_action_and_logprob(self, obs):
        features = self._features(obs)
        dist = torch.distributions.Categorical(logits=self.actor(features))
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def get_value(self, obs):
        return self.critic(self._features(obs)).squeeze(-1)

    def get_logprob_and_entropy(self, obs, actions):
        features = self._features(obs)
        dist = torch.distributions.Categorical(logits=self.actor(features))
        return dist.log_prob(actions), dist.entropy()
```

---

## 5. Integration with Existing Ecosystem

### 5.1 Using Models from torchvision, timm, or HuggingFace

The `PretrainedExtractor` pattern from Section 3.3 is the recommended approach. Key considerations:

```python
# -- torchvision --
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
backbone.classifier = nn.Identity()  # remove classification head
extractor = PretrainedExtractor(backbone, features_dim=1280, freeze=True)

# -- timm (recommended for vision -- more models, cleaner API) --
import timm

backbone = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
extractor = PretrainedExtractor(backbone, features_dim=768)

# -- HuggingFace (for text/multimodal) --
from transformers import AutoModel

backbone = AutoModel.from_pretrained("distilbert-base-uncased")
# Use the [CLS] token output -- wrap in a custom module
class CLSExtractor(FeatureExtractor):
    def __init__(self, model_name: str):
        model = AutoModel.from_pretrained(model_name)
        super().__init__(model.config.hidden_size)
        self.model = model

    def forward(self, obs):
        # obs: dict with input_ids, attention_mask
        return self.model(**obs).last_hidden_state[:, 0]
```

**Gotcha:** Pre-trained vision models expect specific input normalization (ImageNet stats). Users must ensure their observation preprocessing matches. rlox should document this prominently.

### 5.2 torch.compile

`torch.compile` (PyTorch 2.0+) can accelerate both the feature extractor and the policy heads. The key requirement is that the compiled module must have stable input shapes.

```python
# Compile the full policy
policy = CNNMLPPolicy(n_input_channels=4, n_actions=4)
policy = torch.compile(policy, mode="reduce-overhead")

# Or compile just the feature extractor (safer, fewer graph breaks)
policy.cnn = torch.compile(policy.cnn, mode="reduce-overhead")

# Pass to PPO as usual
ppo = PPO("ALE/Breakout-v5", policy=policy, n_envs=8)
```

**Caveats:**
- `torch.compile` requires consistent tensor shapes. The collector must produce fixed-size batches (it already does: `n_envs * n_steps`).
- First call triggers compilation (30-120s). Not an issue for training but matters for inference/eval callbacks.
- `Categorical.sample()` and `Normal.rsample()` are supported. `torch.distributions` works under compile.
- Dynamic control flow in `forward()` (if/else on tensor values) causes graph breaks. Keep feature extractors simple.

**Recommendation for rlox:** Add a `compile: bool = False` parameter to the `PPO` constructor that wraps the policy with `torch.compile` after construction.

### 5.3 Mixed Precision (AMP)

Automatic Mixed Precision reduces memory usage and can provide 1.5-3x speedup on GPU.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In the training loop (inside PPO.train):
for mb in batch.sample_minibatches(cfg.batch_size):
    with autocast(device_type="cuda", dtype=torch.float16):
        loss, metrics = loss_fn(policy, mb.obs, mb.actions, ...)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
```

**Recommendation for rlox:** Add `use_amp: bool = False` to `PPOConfig`. The implementation changes are minimal -- wrap the loss computation in `autocast` and use `GradScaler` for the optimizer step. This is a ~20-line change in `ppo.py`.

**Important:** AMP is only beneficial on CUDA GPUs with Tensor Cores (Volta+). On CPU or Apple Silicon, it provides no speedup and can reduce numerical precision. Guard with:

```python
if self.config.use_amp and self.device.startswith("cuda"):
    # use AMP
```

---

## 6. Observation Preprocessing

### 6.1 Image Observations

Atari and pixel-based environments produce `(H, W, C)` uint8 observations. RL algorithms typically need `(C, H, W)` float32 tensors with frame stacking.

**Recommended preprocessing pipeline:**

```python
class AtariPreprocessor:
    """Standard Atari preprocessing following Machado et al. (2018).

    Converts (210, 160, 3) RGB frames to (84, 84) grayscale,
    stacks 4 frames, and normalizes to [0, 1].
    """

    def __init__(self, n_stack: int = 4):
        self.n_stack = n_stack
        self._frames: deque[np.ndarray] = deque(maxlen=n_stack)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        frame = self._preprocess(obs)
        for _ in range(self.n_stack):
            self._frames.append(frame)
        return self._stack()

    def step(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(self._preprocess(obs))
        return self._stack()

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        # Grayscale + resize to 84x84
        import cv2
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def _stack(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)  # (4, 84, 84)
```

**Where preprocessing should happen:**

| Stage | Python | Rust | Recommendation |
|-------|--------|------|----------------|
| Frame stacking | Easy, flexible | Possible but complex | Python (gymnasium wrappers) |
| Grayscale + resize | cv2 is fast | Image crate exists | Python (cv2 is GPU-capable) |
| Normalization (/ 255) | Trivial | Trivial | Python (at tensor creation) |
| Channel transpose | `np.transpose` | Zero-copy possible | Python |
| Running mean/std | Slow for large obs | Fast (`RunningStats`) | Rust for the stats, Python for the division |

**Recommendation:** Use gymnasium's built-in wrappers (`AtariPreprocessing`, `FrameStack`, `TransformObservation`) at environment construction time. This means the `GymVecEnv` receives pre-processed observations. No changes to rlox's Rust core are needed.

```python
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def make_atari_env(env_id: str, seed: int = 0):
    env = gym.make(env_id)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

# Then use GymVecEnv with a factory function (see Section 7)
```

### 6.2 Dictionary Observations

Multi-modal environments (e.g., robotics with camera + proprioception) return dict observations.

```python
# Example: obs = {"image": np.array(B, 3, 64, 64), "proprio": np.array(B, 12)}
```

**Current limitation:** `RolloutCollector` assumes obs is a single ndarray. Supporting dicts requires:

1. The collector stores lists of dicts (or dicts of lists)
2. The batch stores obs as a dict of tensors
3. The minibatch sampler indexes into each dict entry

**Proposed approach:** A `DictRolloutBatch` variant:

```python
@dataclass
class DictRolloutBatch:
    obs: dict[str, torch.Tensor]  # {"image": (N, C, H, W), "proprio": (N, D)}
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def sample_minibatches(self, batch_size: int, shuffle: bool = True):
        n = next(iter(self.obs.values())).shape[0]
        indices = torch.randperm(n) if shuffle else torch.arange(n)
        n_complete = (n // batch_size) * batch_size
        for start in range(0, n_complete, batch_size):
            idx = indices[start : start + batch_size]
            yield DictRolloutBatch(
                obs={k: v[idx] for k, v in self.obs.items()},
                actions=self.actions[idx],
                rewards=self.rewards[idx],
                dones=self.dones[idx],
                log_probs=self.log_probs[idx],
                values=self.values[idx],
                advantages=self.advantages[idx],
                returns=self.returns[idx],
            )
```

### 6.3 Where to Draw the Python/Rust Boundary

The design principle should follow the Polars pattern rlox already uses: **Rust owns the numerically intensive, batch-parallel operations; Python owns the flexible, user-facing transformations.**

```
                    Rust boundary                Python boundary
                    ============                 ================
env.step_all()  --> obs (numpy) --> preprocess --> tensor --> policy.forward()
                                    (wrappers)   (torch)    (PyTorch)

rewards (numpy) --> compute_gae() --> advantages (numpy) --> tensor --> PPOLoss
```

Observation preprocessing stays in Python because:
- It varies wildly between domains (images, text, graphs, point clouds)
- gymnasium's wrapper ecosystem already handles it well
- The compute cost is small relative to NN forward passes on GPU
- Rust reimplementation would be a large surface area with little speedup

---

## 7. Recommendations for rlox API Changes

Prioritized from most impactful to least, with specific files and APIs.

### Priority 1 (High Impact, Low Effort): Fix Observation Shape Handling

**Problem:** `RolloutCollector.collect()` flattens obs to 2D, breaking CNNs/ViTs.

**File:** `python/rlox/collectors.py`

**Change:** Remove the `reshape(total, -1)` on line 215. Instead, store the original observation shape and only flatten scalar dimensions:

```python
# Current (line 215):
obs=obs_t.reshape(total, -1),

# Proposed:
obs=obs_t.reshape(total, *obs_t.shape[2:]),  # preserve spatial dims
```

This single change unblocks CNNs, ViTs, and any architecture that needs spatial observation structure. It is backward compatible because `reshape(total, obs_dim)` and `reshape(total, *[obs_dim])` produce the same result for 1D observations.

**Also change** `RolloutBatch` docstring in `python/rlox/batch.py` to document that `obs` shape is `(N, *obs_shape)` not just `(N, obs_dim)`.

### Priority 2 (High Impact, Low Effort): Accept Custom Networks in SAC and DQN

**File:** `python/rlox/algorithms/sac.py`

**Change:** Add optional `actor`, `critic1`, `critic2` parameters to `SAC.__init__`:

```python
def __init__(
    self,
    env_id: str,
    actor: nn.Module | None = None,      # NEW
    critic: nn.Module | None = None,      # NEW (used for both critics)
    ...
):
    ...
    self.actor = actor or SquashedGaussianPolicy(obs_dim, act_dim, hidden)
    self.critic1 = critic or QNetwork(obs_dim, act_dim, hidden)
    self.critic2 = copy.deepcopy(self.critic1) if critic else QNetwork(obs_dim, act_dim, hidden)
    ...
```

**File:** `python/rlox/algorithms/dqn.py`

**Change:** Add optional `q_network` parameter to `DQN.__init__`:

```python
def __init__(
    self,
    env_id: str,
    q_network: nn.Module | None = None,  # NEW
    ...
):
    ...
    if q_network is not None:
        self.q_network = q_network
    else:
        net_cls = DuelingQNetwork if dueling else SimpleQNetwork
        self.q_network = net_cls(obs_dim, act_dim, hidden)
    self.target_network = copy.deepcopy(self.q_network)
    ...
```

This matches the pattern PPO already uses (`policy: nn.Module | None = None`).

### Priority 3 (High Impact, Medium Effort): Add Protocol Definitions

**New file:** `python/rlox/protocols.py`

Add the `@runtime_checkable` Protocol classes from Section 2. Then validate at algorithm construction:

```python
# In PPO.__init__, after policy is set:
if not isinstance(self.policy, ActorCriticPolicy):
    warnings.warn(
        f"Policy {type(self.policy).__name__} does not implement ActorCriticPolicy protocol. "
        "Required methods: get_action_and_logprob, get_value, get_logprob_and_entropy"
    )
```

Use `warnings.warn` rather than raising -- this preserves backward compatibility for duck-typed policies that work but were not designed with the protocol in mind.

### Priority 4 (Medium Impact, Medium Effort): Add CompositeActorCritic

**New file:** `python/rlox/composite.py`

Add the `FeatureExtractor`, `CompositeActorCritic`, and built-in extractors (`MLPExtractor`, `NatureCNNExtractor`, `PretrainedExtractor`) from Section 3.

**File:** `python/rlox/__init__.py`

Export the new classes.

**File:** `python/rlox/trainers.py`

Allow `PPOTrainer` to accept an extractor:

```python
class PPOTrainer:
    def __init__(
        self,
        env: str,
        model: str = "mlp",           # existing
        feature_extractor=None,        # NEW
        ...
    ):
        if feature_extractor is not None:
            env_tmp = gym.make(env)
            policy = CompositeActorCritic(feature_extractor, env_tmp.action_space)
            env_tmp.close()
            self.algo = PPO(env_id=env, policy=policy, ...)
        else:
            # existing auto-detection logic
```

### Priority 5 (Medium Impact, Medium Effort): Add AMP Support

**File:** `python/rlox/config.py`

Add `use_amp: bool = False` to `PPOConfig`.

**File:** `python/rlox/algorithms/ppo.py`

Wrap the training inner loop:

```python
# Add to __init__:
self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and self.device.startswith("cuda"))

# In train(), replace the minibatch loop body:
with torch.autocast(device_type=self.device.split(":")[0], enabled=self.config.use_amp):
    loss, metrics = self.loss_fn(...)

self.scaler.scale(loss).backward()
self.scaler.unscale_(self.optimizer)
nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
self.scaler.step(self.optimizer)
self.scaler.update()
self.optimizer.zero_grad()
```

### Priority 6 (Medium Impact, High Effort): GymVecEnv with Factory Functions

**File:** `python/rlox/gym_vec_env.py`

Allow `GymVecEnv` to accept a factory function instead of just an `env_id` string. This enables wrapped environments (Atari preprocessing, frame stacking, etc.):

```python
class GymVecEnv:
    def __init__(
        self,
        env_id_or_factory: str | Callable[[], gym.Env],
        n_envs: int = 1,
        seed: int = 0,
    ):
        if callable(env_id_or_factory):
            self.envs = [env_id_or_factory() for _ in range(n_envs)]
        else:
            self.envs = [gym.make(env_id_or_factory) for _ in range(n_envs)]
```

**File:** `python/rlox/collectors.py`

Propagate the factory function through `RolloutCollector`:

```python
class RolloutCollector:
    def __init__(
        self,
        env_id: str | Callable[[], gym.Env],  # accept factory
        ...
    ):
```

This is the key enabler for image-based RL, because Atari preprocessing wrappers must be applied at environment construction time.

### Priority 7 (Low Impact, High Effort): Recurrent Policy Support

This requires a new `RecurrentRolloutCollector` that:
1. Carries hidden states across `collect()` calls
2. Resets hidden states on episode boundaries
3. Collects sequences rather than flat transitions
4. Stores hidden states in the batch for BPTT

This is a substantial feature. I recommend deferring it and documenting that LSTM/GRU users should write custom training loops using rlox primitives (`compute_gae`, `ExperienceTable`, `PPOLoss`).

### Priority 8 (Low Impact, Low Effort): torch.compile Flag

**File:** `python/rlox/algorithms/ppo.py`

```python
class PPO:
    def __init__(self, ..., compile: bool = False):
        ...
        if compile:
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
```

---

## Summary: Recommended Implementation Order

| Phase | Task | Files Changed | Effort |
|-------|------|---------------|--------|
| 1 | Fix obs shape flattening | `collectors.py`, `batch.py` | 1 hour |
| 2 | Accept custom networks in SAC/DQN | `sac.py`, `dqn.py` | 1 hour |
| 3 | Add protocol definitions | New `protocols.py`, `ppo.py`, `sac.py`, `dqn.py` | 2 hours |
| 4 | Add CompositeActorCritic + extractors | New `composite.py`, `__init__.py`, `trainers.py` | 4 hours |
| 5 | AMP support | `config.py`, `ppo.py` | 2 hours |
| 6 | GymVecEnv factory functions | `gym_vec_env.py`, `collectors.py` | 3 hours |
| 7 | torch.compile flag | `ppo.py` | 30 min |
| 8 | Recurrent collector (deferred) | New `recurrent_collector.py` | 8+ hours |

Phases 1-3 are the highest leverage. They unblock the majority of custom architecture use cases with minimal changes to the existing codebase. Phase 4 provides the "batteries included" experience. Phases 5-7 are quality-of-life improvements. Phase 8 is a separate project.

---

## References

[1] J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
[2] M. Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," ICLR 2021.
[3] S. Huang et al., "CleanRL: High-quality Single-file Implementations of Deep RL Algorithms," JMLR, vol. 23, pp. 1-18, 2022.
[4] A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," JMLR, vol. 22, pp. 1-8, 2021.
[5] V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, pp. 529-533, 2015.
[6] M. Machado et al., "Revisiting the Arcade Learning Environment," JAIR, vol. 61, pp. 523-562, 2018.
[7] S. Keshav, "How to Read a Paper," ACM SIGCOMM Computer Communication Review, vol. 37, no. 3, 2007.
