# Full-Rust Custom Neural Network Architectures in rlox

## Research Report: Enabling Custom NN Architectures and Pure-Rust Training Pipelines

---

## 1. Current Rust NN Architecture Analysis

### 1.1 Trait Hierarchy

The NN abstraction lives in `rlox-nn` and defines five core traits, all verified object-safe:

```
ActorCritic          -- PPO/A2C: act(), value(), evaluate(), ppo_step()
QFunction            -- DQN: q_values(), q_value_at(), td_step()
StochasticPolicy     -- SAC actor: sample_actions(), deterministic_action()
ContinuousQFunction  -- SAC/TD3 critic: q_value(), twin_q_values(), critic_step()
DeterministicPolicy  -- TD3 actor: act(), target_act()
EntropyTuner         -- SAC alpha: alpha(), update()
```

**Key design decision**: Training steps that require autograd to flow across actor-critic boundaries (SAC's `sac_actor_step`, TD3's `td3_actor_step`) are **not** on the traits. They are inherent methods on concrete types. This is because trait methods go through `TensorData` (a `Vec<f32>` container), which severs the computation graph.

**Consequence for custom architectures**: Users can implement `ActorCritic` for their own struct and get `ppo_step()` working (since PPO's actor and critic gradients are independent). But SAC/TD3 actor steps require concrete type knowledge of both the policy and critic, which is currently hardcoded.

### 1.2 TensorData Bridge

`TensorData` is the lingua franca between `rlox-core` (which stores rollout data as flat `Vec<f32>`) and NN backends:

```rust
pub struct TensorData {
    pub data: Vec<f32>,   // flat row-major
    pub shape: Vec<usize>,
}
```

Each backend provides convert utilities:
- **Candle**: `to_tensor_2d(data, device) -> candle_core::Tensor`, `from_tensor_1d(tensor) -> TensorData`
- **Burn**: `to_tensor_2d::<B>(data, device) -> Tensor<B, 2>`, `from_tensor_1d(tensor) -> TensorData`

The conversion involves a `Vec<f32>` clone on every call. For inference (act/value), this is acceptable since the forward pass dominates. For training, both backends keep tensors in native form within `ppo_step` and only convert inputs/outputs at the boundary.

### 1.3 Burn Implementation Structure

Burn uses a split pattern to work around `#[derive(Module)]` requirements:

```
MLPParams<B: Backend>          -- #[derive(Module)], holds Vec<Linear<B>>
MLP<B: Backend>                -- wraps MLPParams + ActivationKind (not Module)

DiscreteActorCriticParams<B>   -- #[derive(Module)], holds actor: MLPParams + critic: MLPParams
DiscreteActorCriticModel<B>    -- wraps Params + activation config

BurnActorCritic<B: AutodiffBackend>  -- holds Model + Optimizer + device + rng
  impl ActorCritic for BurnActorCritic<B>
```

The Burn `Module` derive macro only works on types that are purely learnable parameters (tensors + other Modules). Non-parameter config (activation functions, hidden sizes) must be stored outside the Module. The `valid()` method strips autograd for inference by converting `B` to `B::InnerBackend`.

### 1.4 Candle Implementation Structure

Candle is simpler since it doesn't require derive macros:

```
MLP                    -- holds Vec<Linear> + Activation config
CandleActorCritic      -- holds actor: MLP + critic: MLP + VarMap + AdamW optimizer
  impl ActorCritic for CandleActorCritic
```

Candle uses `VarMap` as the parameter registry. All trainable variables are registered in the VarMap, and the optimizer operates on `varmap.all_vars()`. The `VarBuilder` pattern namespaces weights (e.g., `vb.pp("actor").pp("layer_0")`).

### 1.5 Training Methods

| Algorithm | Method | Location | Autograd Requirement |
|-----------|--------|----------|---------------------|
| PPO | `ppo_step()` | On `ActorCritic` trait | Actor/critic independent -- works through trait |
| DQN | `td_step()` | On `QFunction` trait | Single network -- works through trait |
| SAC actor | `sac_actor_step()` | Inherent on concrete type | Needs autograd through critic Q-network |
| SAC critic | `critic_step()` | On `ContinuousQFunction` trait | Single network -- works through trait |
| TD3 actor | `td3_actor_step()` | Inherent on concrete type | Needs autograd through critic Q-network |
| TD3 critic | `critic_step()` | On `ContinuousQFunction` trait | Single network -- works through trait |

### 1.6 Current Limitations for Custom Architectures

1. **No feature extractor abstraction**: The MLP is the only supported backbone. CNNs, transformers, or recurrent networks require reimplementing the entire `BurnActorCritic`/`CandleActorCritic` from scratch.

2. **No composition pattern**: There is no way to say "use my CNN as the feature extractor, but reuse the existing categorical/Gaussian distribution heads."

3. **SAC/TD3 actor steps are hardcoded**: `sac_actor_step` takes a concrete `CandleTwinQ` / `BurnTwinQ`. A custom policy cannot use the existing SAC training loop without also knowing the exact critic type.

4. **No Rust-side training loop**: The `AsyncCollector` collects rollouts but expects `action_fn` and `value_fn` as closures over `&[f32]`. There is no `PPOTrainer` struct that ties env + policy + GAE + minibatch updates together.

5. **No scheduler/callback infrastructure**: Learning rate scheduling, early stopping, logging -- all live in Python.

6. **MLPConfig is the only network config**: No `CnnConfig`, `TransformerConfig`, or generic `NetworkConfig`.

---

## 2. Making Rust Traits Extensible for Custom Architectures

### 2.1 The FeatureExtractor Pattern

The key insight from SB3/CleanRL is that most RL networks share a common pattern:

```
observations -> [FeatureExtractor] -> features -> [Head] -> output
```

Where `FeatureExtractor` can be an MLP, CNN, transformer, or any custom module, and `Head` is algorithm-specific (categorical logits, Gaussian mean/std, Q-values).

#### Proposed Trait: `FeatureExtractor`

```rust
/// A feature extractor that maps observations to a fixed-size feature vector.
///
/// This is the primary extension point for custom architectures. Users implement
/// this trait for their CNN, transformer, RNN, etc., and compose it with
/// standard distribution heads.
pub trait FeatureExtractor: Send {
    /// Output dimension of the feature vector.
    fn feature_dim(&self) -> usize;

    /// Map observations to features. No gradient tracking.
    /// Input: [batch_size, ...obs_shape], Output: [batch_size, feature_dim].
    fn extract(&self, obs: &TensorData) -> Result<TensorData, NNError>;
}
```

This trait is intentionally simple and object-safe. It operates on `TensorData`, which means it erases the computation graph -- suitable for inference during rollout collection, but **not** for training.

For training, we need backend-specific versions:

```rust
// In rlox-candle:
pub trait CandleFeatureExtractor: Send {
    fn feature_dim(&self) -> usize;

    /// Extract features, returning a Candle tensor with autograd intact.
    fn extract_tensor(&self, obs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>;

    /// Get all trainable variables for optimizer registration.
    fn var_map(&self) -> &candle_nn::VarMap;
}

// In rlox-burn:
pub trait BurnFeatureExtractor<B: Backend>: Send {
    fn feature_dim(&self) -> usize;

    /// Extract features, returning a Burn tensor with autograd intact.
    fn extract_tensor(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;
}
```

### 2.2 Can Users Already Implement ActorCritic?

**Yes**, but with significant boilerplate. A user would need to:

1. Build their custom network (CNN, transformer, etc.) using raw Candle/Burn ops.
2. Implement all 7 methods of `ActorCritic` (or 6 for `QFunction`, etc.).
3. Handle the full PPO loss computation inside `ppo_step()` themselves.
4. Manage their own optimizer.

This is viable but painful. The `ppo_step` implementation alone is ~80 lines of loss computation that would be duplicated for every custom architecture.

### 2.3 Proposed: Composable Actor-Critic Builder

Instead of reimplementing everything, users should be able to plug in a feature extractor:

```rust
// Candle version -- user only writes the extractor:
pub struct CandleComposableActorCritic {
    extractor: Box<dyn CandleFeatureExtractor>,
    actor_head: MLP,      // reuse existing MLP for logits
    critic_head: MLP,     // reuse existing MLP for values
    varmap: VarMap,       // merged from extractor + heads
    optimizer: AdamW,
    // ...
}

impl CandleComposableActorCritic {
    pub fn new(
        extractor: Box<dyn CandleFeatureExtractor>,
        n_actions: usize,
        head_hidden: Vec<usize>,
        lr: f64,
        device: Device,
        seed: u64,
    ) -> Result<Self, NNError> {
        let feat_dim = extractor.feature_dim();
        let varmap = VarMap::new();
        // Merge extractor's variables into the unified VarMap
        // ...build actor_head and critic_head using feat_dim as input...
    }
}

impl ActorCritic for CandleComposableActorCritic { /* reuse PPO loss logic */ }
```

### 2.4 Should There Be a `CustomBackend` Trait?

**No.** Adding a `CustomBackend` trait that wraps arbitrary forward passes would create another layer of indirection without solving the core problem (autograd through the training step). Instead, the extension points should be:

1. **`FeatureExtractor` trait** (inference-only, object-safe) -- for use with `AsyncCollector` and Python bridge.
2. **Backend-specific extractor traits** (training, preserves autograd) -- for `ppo_step`, `sac_actor_step`, etc.
3. **Direct `ActorCritic` implementation** (power users) -- for full control.

---

## 3. Full-Rust Training Loop Design

### 3.1 Dream API

```rust
use rlox_core::env::builtins::CartPole;
use rlox_core::env::parallel::VecEnv;
use rlox_core::training::PPOTrainer;
use rlox_candle::actor_critic::CandleActorCritic;

fn main() {
    let envs = VecEnv::new_homogeneous::<CartPole>(16, 42);

    let policy = CandleActorCritic::new(4, 2, 64, 2.5e-4, Device::Cpu, 42).unwrap();

    let config = PPOConfig {
        total_timesteps: 1_000_000,
        n_steps: 128,
        n_epochs: 4,
        n_minibatches: 4,
        gamma: 0.99,
        gae_lambda: 0.95,
        ppo: PPOStepConfig::default(),
        anneal_lr: true,
        normalize_advantages: true,
    };

    let metrics = PPOTrainer::train(envs, policy, config, |update, metrics| {
        if update % 10 == 0 {
            println!("update {}: reward={:.1}", update, metrics.mean_reward);
        }
    });
}
```

### 3.2 What Infrastructure Is Missing

#### a) PPOTrainer (new, in rlox-core or new rlox-training crate)

```rust
/// Full PPO training loop configuration.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    pub total_timesteps: usize,
    pub n_steps: usize,
    pub n_epochs: usize,
    pub n_minibatches: usize,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub ppo: PPOStepConfig,
    pub anneal_lr: bool,
    pub normalize_advantages: bool,
    pub seed: u64,
}

/// Metrics returned after each update.
#[derive(Debug, Clone)]
pub struct UpdateMetrics {
    pub update: usize,
    pub total_steps: usize,
    pub mean_reward: f64,
    pub mean_episode_length: f64,
    pub train_metrics: TrainMetrics,  // from ppo_step
    pub fps: f64,
}

/// PPO trainer that owns the full loop.
pub struct PPOTrainer;

impl PPOTrainer {
    pub fn train<E, P, F>(
        mut envs: E,
        mut policy: P,
        config: PPOConfig,
        callback: F,
    ) -> Vec<UpdateMetrics>
    where
        E: BatchSteppable,
        P: ActorCritic,
        F: FnMut(usize, &UpdateMetrics),
    {
        // Implementation follows CleanRL's PPO structure:
        // 1. Initialize obs from envs.reset_batch()
        // 2. For each update:
        //    a. Collect n_steps of rollout data
        //    b. Compute GAE
        //    c. For each epoch:
        //       - Shuffle and split into minibatches
        //       - Call policy.ppo_step() on each minibatch
        //    d. Optionally anneal LR
        //    e. Track episode rewards via done signals
        //    f. Call callback with metrics
    }
}
```

#### b) Synchronous Rollout Collector (complement to AsyncCollector)

The existing `AsyncCollector` takes function pointers for action/value, designed for the Python bridge. A Rust-native collector should work directly with `ActorCritic`:

```rust
/// Synchronous rollout collector that works with any ActorCritic.
pub struct RolloutCollector;

impl RolloutCollector {
    /// Collect n_steps of experience from envs using the policy.
    /// Returns a RolloutBatch with GAE computed.
    pub fn collect<E: BatchSteppable, P: ActorCritic>(
        envs: &mut E,
        policy: &P,
        current_obs: &mut Vec<f32>,
        n_steps: usize,
        gamma: f64,
        gae_lambda: f64,
    ) -> Result<RolloutBatchOwned, RloxError> {
        // Similar to AsyncCollector but:
        // - Calls policy.act() and policy.value() directly
        // - Returns log_probs and old_values for PPO
        // - No channel, no thread -- just a function call
    }
}
```

The returned batch needs to include `log_probs` and `old_values` (which the current `RolloutBatch` lacks, since the Python side computes those separately):

```rust
/// Extended rollout batch with PPO-specific fields.
pub struct PPORolloutBatch {
    pub observations: Vec<f32>,   // [n_steps * n_envs * obs_dim]
    pub actions: Vec<f32>,        // [n_steps * n_envs * act_dim]
    pub log_probs: Vec<f32>,      // [n_steps * n_envs] -- from policy.act()
    pub rewards: Vec<f64>,        // [n_steps * n_envs]
    pub dones: Vec<f64>,          // [n_steps * n_envs]
    pub values: Vec<f64>,         // [n_steps * n_envs] -- from policy.value()
    pub advantages: Vec<f64>,     // [n_steps * n_envs] -- from GAE
    pub returns: Vec<f64>,        // [n_steps * n_envs] -- advantages + values
    pub obs_dim: usize,
    pub act_dim: usize,
    pub n_steps: usize,
    pub n_envs: usize,
    // Episode tracking
    pub completed_episodes: Vec<f64>,  // rewards of completed episodes
}
```

#### c) Minibatch Iterator

```rust
/// Iterator over random minibatches from a PPORolloutBatch.
pub struct MinibatchIterator<'a> {
    batch: &'a PPORolloutBatch,
    indices: Vec<usize>,
    minibatch_size: usize,
    pos: usize,
}

impl<'a> Iterator for MinibatchIterator<'a> {
    type Item = PPOMinibatch;

    fn next(&mut self) -> Option<PPOMinibatch> {
        // Returns TensorData slices for obs, actions, old_log_probs,
        // advantages, returns, old_values
    }
}

pub struct PPOMinibatch {
    pub obs: TensorData,           // [mb_size, obs_dim]
    pub actions: TensorData,       // [mb_size] or [mb_size, act_dim]
    pub old_log_probs: TensorData, // [mb_size]
    pub advantages: TensorData,    // [mb_size]
    pub returns: TensorData,       // [mb_size]
    pub old_values: TensorData,    // [mb_size]
}
```

#### d) Learning Rate Scheduler

```rust
pub enum LRSchedule {
    Constant,
    Linear { start: f32, end: f32, total_steps: usize },
    Cosine { start: f32, min: f32, total_steps: usize },
}

impl LRSchedule {
    pub fn get_lr(&self, step: usize) -> f32 { /* ... */ }
}
```

### 3.3 Where Should the Training Loop Live?

**Recommendation: `rlox-core`**, not a new crate. Reasons:

1. The training loop depends on `BatchSteppable`, `RolloutBatch`, `compute_gae` -- all in `rlox-core`.
2. It depends on `ActorCritic`, `PPOStepConfig`, `TensorData` -- all in `rlox-nn` (which `rlox-core` already depends on or can depend on).
3. A new crate would add dependency management overhead with minimal benefit.

The module structure would be:

```
rlox-core/src/training/
    mod.rs
    gae.rs              (existing)
    ppo_trainer.rs      (new)
    rollout.rs          (new -- sync collector + minibatch iterator)
    lr_schedule.rs      (new)
    callbacks.rs        (new -- trait for logging/early stopping)
```

### 3.4 Composing with Existing Infrastructure

The `AsyncCollector` would remain for the Python path (policy in Python, env stepping in Rust). The new `PPOTrainer` handles the fully-Rust path:

```
Python path:
  Python policy -> action_fn/value_fn closures -> AsyncCollector (threaded) -> RolloutBatch -> Python training

Rust path:
  Rust ActorCritic -> PPOTrainer -> SyncCollector -> PPORolloutBatch -> MinibatchIterator -> ppo_step()
```

Both share `BatchSteppable`, `compute_gae`, and the environment infrastructure.

---

## 4. Candle Custom Architecture Examples

### 4.1 Custom CNN Feature Extractor

```rust
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder, VarMap, linear, Linear};
use rlox_nn::{NNError, TensorData};

/// Nature-CNN feature extractor (Mnih et al. 2015) for Atari-like inputs.
///
/// Input: [batch, 84*84*4] (flattened grayscale frames)
/// Output: [batch, 512]
pub struct NatureCNN {
    conv1: Conv2d,  // 32 filters, 8x8, stride 4
    conv2: Conv2d,  // 64 filters, 4x4, stride 2
    conv3: Conv2d,  // 64 filters, 3x3, stride 1
    fc: Linear,     // 3136 -> 512
    varmap: VarMap,
}

impl NatureCNN {
    pub fn new(device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

        let conv1 = conv2d(
            4, 32, 8,
            Conv2dConfig { stride: 4, ..Default::default() },
            vb.pp("conv1"),
        )?;
        let conv2 = conv2d(
            32, 64, 4,
            Conv2dConfig { stride: 2, ..Default::default() },
            vb.pp("conv2"),
        )?;
        let conv3 = conv2d(
            64, 64, 3,
            Conv2dConfig { stride: 1, ..Default::default() },
            vb.pp("conv3"),
        )?;
        let fc = linear(3136, 512, vb.pp("fc"))?;

        Ok(Self { conv1, conv2, conv3, fc, varmap })
    }

    pub fn feature_dim(&self) -> usize { 512 }

    /// Forward pass: [batch, 4, 84, 84] -> [batch, 512]
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(obs)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        let x = self.conv3.forward(&x)?.relu()?;
        let x = x.flatten_from(1)?;  // [batch, 3136]
        self.fc.forward(&x)?.relu()
    }

    pub fn var_map(&self) -> &VarMap { &self.varmap }
}

/// Complete CNN actor-critic for Atari PPO.
pub struct CandleCnnActorCritic {
    extractor: NatureCNN,
    actor_head: Linear,   // 512 -> n_actions
    critic_head: Linear,  // 512 -> 1
    varmap: VarMap,
    optimizer: candle_nn::AdamW,
    device: Device,
    n_actions: usize,
    lr: f64,
    rng: std::cell::RefCell<rand_chacha::ChaCha8Rng>,
}

impl CandleCnnActorCritic {
    pub fn new(
        n_actions: usize,
        lr: f64,
        device: Device,
        seed: u64,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Build extractor with shared VarMap
        let conv1 = conv2d(4, 32, 8,
            Conv2dConfig { stride: 4, ..Default::default() },
            vb.pp("cnn.conv1"))?;
        let conv2 = conv2d(32, 64, 4,
            Conv2dConfig { stride: 2, ..Default::default() },
            vb.pp("cnn.conv2"))?;
        let conv3 = conv2d(64, 64, 3,
            Conv2dConfig { stride: 1, ..Default::default() },
            vb.pp("cnn.conv3"))?;
        let fc = linear(3136, 512, vb.pp("cnn.fc"))?;
        let extractor = NatureCNN { conv1, conv2, conv3, fc,
            varmap: varmap.clone() /* shares underlying vars */ };

        let actor_head = linear(512, n_actions, vb.pp("actor"))?;
        let critic_head = linear(512, 1, vb.pp("critic"))?;

        let params = varmap.all_vars();
        let optimizer = candle_nn::AdamW::new(
            params,
            candle_nn::ParamsAdamW { lr, ..Default::default() },
        )?;

        Ok(Self {
            extractor, actor_head, critic_head,
            varmap, optimizer, device, n_actions, lr,
            rng: std::cell::RefCell::new(
                rand_chacha::ChaCha8Rng::seed_from_u64(seed)
            ),
        })
    }

    fn forward_features(&self, obs_flat: &Tensor) -> Result<Tensor> {
        // Reshape from [batch, 84*84*4] to [batch, 4, 84, 84]
        let batch_size = obs_flat.dim(0)?;
        let obs_4d = obs_flat.reshape((batch_size, 4, 84, 84))?;
        self.extractor.forward(&obs_4d)
    }
}

// impl ActorCritic for CandleCnnActorCritic follows the same pattern
// as CandleActorCritic, but calls self.forward_features() instead of
// self.actor.forward() / self.critic.forward().
```

### 4.2 Transformer/Attention-Based Policy

```rust
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{linear, Linear, VarBuilder, VarMap, Module};

/// Single-head self-attention for sequence-based observations.
///
/// Useful for multi-agent or variable-length entity observations
/// (e.g., neighboring agents, inventory items).
pub struct SelfAttentionBlock {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    d_k: f64,
}

impl SelfAttentionBlock {
    pub fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let d_k = d_model / n_heads;
        Ok(Self {
            q_proj: linear(d_model, d_model, vb.pp("q"))?,
            k_proj: linear(d_model, d_model, vb.pp("k"))?,
            v_proj: linear(d_model, d_model, vb.pp("v"))?,
            out_proj: linear(d_model, d_model, vb.pp("out"))?,
            d_k: d_k as f64,
        })
    }

    /// Forward: [batch, seq_len, d_model] -> [batch, seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Scaled dot-product attention
        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
        let attn_weights = (attn_weights / self.d_k.sqrt())?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, 2)?;
        let attn_output = attn_weights.matmul(&v)?;

        self.out_proj.forward(&attn_output)
    }
}

/// Transformer-based policy for entity-structured observations.
///
/// Input: [batch, n_entities * entity_dim] (flattened entity features)
/// Output: features [batch, d_model] (mean-pooled over entities)
pub struct TransformerExtractor {
    entity_embed: Linear,       // entity_dim -> d_model
    attn_layers: Vec<SelfAttentionBlock>,
    ffn_layers: Vec<(Linear, Linear)>,  // d_model -> 4*d_model -> d_model
    n_entities: usize,
    entity_dim: usize,
    d_model: usize,
}

impl TransformerExtractor {
    pub fn new(
        n_entities: usize,
        entity_dim: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let entity_embed = linear(entity_dim, d_model, vb.pp("embed"))?;

        let mut attn_layers = Vec::new();
        let mut ffn_layers = Vec::new();
        for i in 0..n_layers {
            attn_layers.push(SelfAttentionBlock::new(
                d_model, n_heads, vb.pp(format!("attn_{i}"))
            )?);
            ffn_layers.push((
                linear(d_model, 4 * d_model, vb.pp(format!("ffn_{i}_up")))?,
                linear(4 * d_model, d_model, vb.pp(format!("ffn_{i}_down")))?,
            ));
        }

        Ok(Self {
            entity_embed, attn_layers, ffn_layers,
            n_entities, entity_dim, d_model,
        })
    }

    pub fn feature_dim(&self) -> usize { self.d_model }

    /// Forward: [batch, n_entities * entity_dim] -> [batch, d_model]
    pub fn forward(&self, obs_flat: &Tensor) -> Result<Tensor> {
        let batch = obs_flat.dim(0)?;
        let x = obs_flat.reshape((batch, self.n_entities, self.entity_dim))?;
        let mut h = self.entity_embed.forward(&x)?;  // [batch, n_entities, d_model]

        for (attn, (ffn_up, ffn_down)) in
            self.attn_layers.iter().zip(self.ffn_layers.iter())
        {
            // Self-attention + residual
            let attn_out = attn.forward(&h)?;
            h = (&h + &attn_out)?;

            // FFN + residual
            let ffn_out = ffn_up.forward(&h)?.relu()?;
            let ffn_out = ffn_down.forward(&ffn_out)?;
            h = (&h + &ffn_out)?;
        }

        // Mean pool over entities: [batch, d_model]
        h.mean(1)
    }
}
```

### 4.3 Recurrent (GRU) Policy for POMDP

```rust
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{linear, Linear, VarBuilder, VarMap};

/// GRU cell for recurrent policies.
///
/// Handles partial observability by maintaining hidden state
/// across timesteps within an episode.
pub struct GRUCell {
    w_z: Linear,  // input -> update gate
    u_z: Linear,  // hidden -> update gate
    w_r: Linear,  // input -> reset gate
    u_r: Linear,  // hidden -> reset gate
    w_h: Linear,  // input -> candidate
    u_h: Linear,  // hidden -> candidate
    hidden_dim: usize,
}

impl GRUCell {
    pub fn new(input_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w_z: linear(input_dim, hidden_dim, vb.pp("w_z"))?,
            u_z: linear(hidden_dim, hidden_dim, vb.pp("u_z"))?,
            w_r: linear(input_dim, hidden_dim, vb.pp("w_r"))?,
            u_r: linear(hidden_dim, hidden_dim, vb.pp("u_r"))?,
            w_h: linear(input_dim, hidden_dim, vb.pp("w_h"))?,
            u_h: linear(hidden_dim, hidden_dim, vb.pp("u_h"))?,
            hidden_dim,
        })
    }

    /// Single step: (x: [batch, input_dim], h: [batch, hidden_dim]) -> h': [batch, hidden_dim]
    pub fn step(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let z = candle_nn::ops::sigmoid(
            &(&self.w_z.forward(x)? + &self.u_z.forward(h)?)?
        )?;
        let r = candle_nn::ops::sigmoid(
            &(&self.w_r.forward(x)? + &self.u_r.forward(h)?)?
        )?;
        let h_candidate = (&self.w_h.forward(x)?
            + &self.u_h.forward(&(&r * h)?)?)?
            .tanh()?;

        // h' = (1 - z) * h + z * h_candidate
        let one_minus_z = (1.0 - &z)?;
        (&(&one_minus_z * h)? + &(&z * &h_candidate)?).map_err(Into::into)
    }
}

/// Recurrent actor-critic for POMDP environments.
///
/// Maintains a hidden state vector per environment that gets
/// reset when episodes terminate.
pub struct RecurrentActorCritic {
    gru: GRUCell,
    actor_head: Linear,
    critic_head: Linear,
    varmap: VarMap,
    optimizer: candle_nn::AdamW,
    device: Device,
    hidden_dim: usize,
    n_actions: usize,
    // Per-env hidden states: [n_envs, hidden_dim]
    hidden_states: Option<Tensor>,
}

impl RecurrentActorCritic {
    pub fn new(
        obs_dim: usize,
        n_actions: usize,
        hidden_dim: usize,
        lr: f64,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let gru = GRUCell::new(obs_dim, hidden_dim, vb.pp("gru"))?;
        let actor_head = linear(hidden_dim, n_actions, vb.pp("actor"))?;
        let critic_head = linear(hidden_dim, 1, vb.pp("critic"))?;

        let params = varmap.all_vars();
        let optimizer = candle_nn::AdamW::new(
            params,
            candle_nn::ParamsAdamW { lr, ..Default::default() },
        )?;

        Ok(Self {
            gru, actor_head, critic_head,
            varmap, optimizer, device, hidden_dim, n_actions,
            hidden_states: None,
        })
    }

    /// Initialize hidden states for n_envs.
    pub fn init_hidden(&mut self, n_envs: usize) -> Result<()> {
        self.hidden_states = Some(
            Tensor::zeros((n_envs, self.hidden_dim), DType::F32, &self.device)?
        );
        Ok(())
    }

    /// Reset hidden states for environments that terminated.
    /// dones: [n_envs], 1.0 for terminated environments.
    pub fn reset_hidden(&mut self, dones: &[f64]) -> Result<()> {
        if let Some(ref mut h) = self.hidden_states {
            let mask: Vec<f32> = dones.iter()
                .map(|&d| if d > 0.5 { 0.0 } else { 1.0 })
                .collect();
            let mask = Tensor::from_vec(mask, h.dim(0)?, &self.device)?
                .unsqueeze(1)?;
            *h = (h.clone() * mask)?;
        }
        Ok(())
    }

    /// Forward pass: obs [batch, obs_dim] -> (logits [batch, n_actions], value [batch])
    /// Updates internal hidden state.
    pub fn forward_step(&mut self, obs: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.hidden_states.as_ref()
            .expect("call init_hidden() first");
        let new_h = self.gru.step(obs, h)?;
        let logits = self.actor_head.forward(&new_h)?;
        let value = self.critic_head.forward(&new_h)?.squeeze(1)?;
        self.hidden_states = Some(new_h);
        Ok((logits, value))
    }
}

// Note: Implementing ActorCritic for a recurrent policy requires changes to
// the rollout collector to pass hidden states and episode boundaries.
// This is a significant extension beyond the current trait design.
// See Section 8 for recommendations on recurrent support.
```

### 4.4 Composing Custom Extractors with Existing MLP Heads

The key pattern is to share the `VarMap`:

```rust
use candle_nn::{VarBuilder, VarMap};
use rlox_candle::mlp::MLP;
use rlox_nn::MLPConfig;

/// Example: custom extractor + standard MLP heads.
pub fn build_custom_actor_critic(
    extractor_builder: impl FnOnce(VarBuilder) -> candle_core::Result<(Box<dyn CandleFeatureExtractor>, usize)>,
    n_actions: usize,
    head_hidden: Vec<usize>,
    lr: f64,
    device: &candle_core::Device,
) -> Result<CandleComposableActorCritic, rlox_nn::NNError> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    // User's custom feature extractor
    let (extractor, feat_dim) = extractor_builder(vb.pp("extractor"))
        .map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?;

    // Standard MLP heads from rlox-candle
    let actor_config = MLPConfig::new(feat_dim, n_actions)
        .with_hidden(head_hidden.clone());
    let actor_head = MLP::new(&actor_config, vb.pp("actor"))
        .map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?;

    let critic_config = MLPConfig::new(feat_dim, 1)
        .with_hidden(head_hidden);
    let critic_head = MLP::new(&critic_config, vb.pp("critic"))
        .map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?;

    // Single optimizer over all parameters
    let optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW { lr, ..Default::default() },
    ).map_err(|e| rlox_nn::NNError::Backend(e.to_string()))?;

    // ... assemble into CandleComposableActorCritic
    todo!()
}
```

---

## 5. Burn Custom Architecture Examples

### 5.1 Custom CNN with Burn's Module System

Burn's `#[derive(Module)]` requires that all fields are either:
- Other `Module`-implementing types (tensors, Linear, Conv2d, etc.)
- Types that implement `burn::module::Module` via the derive macro

Non-parameter config (activation type, dimensions) must be stored outside the Module struct.

```rust
use burn::prelude::*;
use burn::nn::{conv, Linear, LinearConfig};
use burn::tensor::activation;

/// Nature CNN parameters -- Module-derivable.
#[derive(Module, Debug)]
pub struct NatureCNNParams<B: Backend> {
    conv1: conv::Conv2d<B>,
    conv2: conv::Conv2d<B>,
    conv3: conv::Conv2d<B>,
    fc: Linear<B>,
}

/// Nature CNN wrapper with config.
#[derive(Debug)]
pub struct NatureCNN<B: Backend> {
    pub params: NatureCNNParams<B>,
}

impl<B: Backend> NatureCNN<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = conv::Conv2dConfig::new([4, 32], [8, 8])
            .with_stride([4, 4])
            .init(device);
        let conv2 = conv::Conv2dConfig::new([32, 64], [4, 4])
            .with_stride([2, 2])
            .init(device);
        let conv3 = conv::Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .init(device);
        let fc = LinearConfig::new(3136, 512).init(device);

        Self {
            params: NatureCNNParams { conv1, conv2, conv3, fc },
        }
    }

    pub fn feature_dim(&self) -> usize { 512 }

    /// Forward: [batch, 4, 84, 84] -> [batch, 512]
    pub fn forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = activation::relu(self.params.conv1.forward(obs));
        let x = activation::relu(self.params.conv2.forward(x));
        let x = activation::relu(self.params.conv3.forward(x));
        let [batch, c, h, w] = x.dims();
        let x = x.reshape([batch, c * h * w]);
        activation::relu(self.params.fc.forward(x))
    }

    pub fn valid(&self) -> NatureCNN<B::InnerBackend>
    where
        B: burn::tensor::backend::AutodiffBackend,
    {
        NatureCNN {
            params: self.params.valid(),
        }
    }
}

impl<B: Backend> Clone for NatureCNN<B> {
    fn clone(&self) -> Self {
        Self { params: self.params.clone() }
    }
}
```

### 5.2 How Burn's Backend Generics Affect Custom Architectures

Every Burn model is generic over `B: Backend`. This means:

1. **At the type level**, your custom model carries the backend: `NatureCNN<NdArray>`, `NatureCNN<Wgpu>`, `NatureCNN<Autodiff<NdArray>>`.

2. **For training**, you need `B: AutodiffBackend`. The `valid()` method converts to `B::InnerBackend` for inference (no gradient tracking).

3. **For the rlox trait boundary**, you need to pick a concrete backend when implementing `ActorCritic`:

```rust
impl<B: AutodiffBackend> rlox_nn::ActorCritic for BurnCnnActorCritic<B>
where
    B::Device: Clone,
{
    fn act(&self, obs: &TensorData) -> Result<ActionOutput, NNError> {
        // Convert TensorData -> Tensor<B::InnerBackend, 4>
        // Forward through self.cnn.valid().forward()
        // Convert back to TensorData
    }

    fn ppo_step(&mut self, /* ... */) -> Result<TrainMetrics, NNError> {
        // Convert TensorData -> Tensor<B, 4>  (with autograd!)
        // Forward through self.cnn.forward()
        // PPO loss computation (same as existing BurnActorCritic)
        // Backward + optimizer step
    }
}
```

4. **Backend switching** is free at the type level:

```rust
// CPU training with NdArray backend:
let policy = BurnCnnActorCritic::<Autodiff<NdArray>>::new(
    18, 64, 3e-4, Default::default(), 42
);

// GPU training with Wgpu backend:
let policy = BurnCnnActorCritic::<Autodiff<Wgpu>>::new(
    18, 64, 3e-4, WgpuDevice::default(), 42
);
```

### 5.3 Burn's `#[derive(Module)]` for Custom Models

The complete pattern for a custom Burn actor-critic:

```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::AutodiffBackend;

/// All learnable parameters -- must be Module-derivable.
#[derive(Module, Debug)]
pub struct CnnActorCriticParams<B: Backend> {
    cnn: NatureCNNParams<B>,
    actor_head: Linear<B>,
    critic_head: Linear<B>,
}

/// Full model with non-parameter config.
#[derive(Debug)]
pub struct CnnActorCriticModel<B: Backend> {
    pub params: CnnActorCriticParams<B>,
    // Store NatureCNN config separately if needed for forward()
}

impl<B: Backend> CnnActorCriticModel<B> {
    pub fn new(n_actions: usize, device: &B::Device) -> Self {
        let cnn = NatureCNN::new(device);
        let actor_head = LinearConfig::new(512, n_actions).init(device);
        let critic_head = LinearConfig::new(512, 1).init(device);

        Self {
            params: CnnActorCriticParams {
                cnn: cnn.params,
                actor_head,
                critic_head,
            },
        }
    }

    fn cnn_forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        // Inline NatureCNN forward since we separated params from logic
        let x = burn::tensor::activation::relu(self.params.cnn.conv1.forward(obs));
        let x = burn::tensor::activation::relu(self.params.cnn.conv2.forward(x));
        let x = burn::tensor::activation::relu(self.params.cnn.conv3.forward(x));
        let [batch, c, h, w] = x.dims();
        let x = x.reshape([batch, c * h * w]);
        burn::tensor::activation::relu(self.params.cnn.fc.forward(x))
    }

    pub fn actor_forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        let features = self.cnn_forward(obs);
        self.params.actor_head.forward(features)
    }

    pub fn critic_forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        let features = self.cnn_forward(obs);
        self.params.critic_head.forward(features)
    }

    pub fn valid(&self) -> CnnActorCriticModel<B::InnerBackend>
    where
        B: AutodiffBackend,
    {
        CnnActorCriticModel {
            params: self.params.valid(),
        }
    }
}

impl<B: Backend> Clone for CnnActorCriticModel<B> {
    fn clone(&self) -> Self {
        Self { params: self.params.clone() }
    }
}
```

**Key pain point**: Burn's Module derive requires all params in the Module struct, but forward logic (activation functions, reshaping) can't be derived. This forces the split between `Params` (Module) and `Model` (wraps Params + logic). For complex architectures, this is verbose but workable.

**Potential improvement**: A macro that generates the Params struct from the Model definition, similar to how PyTorch's `nn.Module` works:

```rust
// Hypothetical future ergonomic macro:
#[rlox_module]
pub struct NatureCNN<B: Backend> {
    #[param] conv1: Conv2d<B>,
    #[param] conv2: Conv2d<B>,
    #[config] activation: ActivationKind,  // not Module-derived
}
```

---

## 6. PyO3 Bridge for Rust-Native Networks

### 6.1 Current PyActorCritic Design

The existing `PyActorCritic` in `rlox-python/src/nn.rs` uses an enum dispatch pattern:

```rust
enum Backend {
    Burn(BurnActorCritic<Autodiff<NdArray>>),
    Candle(CandleActorCritic),
}

#[pyclass(name = "ActorCritic", unsendable)]
pub struct PyActorCritic {
    inner: Backend,
    obs_dim: usize,
}
```

Python receives flat `numpy` arrays, converts to `TensorData`, dispatches to the correct backend, and returns `numpy` arrays. This works well for the default MLP architecture.

### 6.2 Extending for Custom Architectures

There are two approaches:

#### Approach A: Pre-registered Architecture Enum (Simple)

Add more variants to the `Backend` enum for each built-in architecture:

```rust
enum Backend {
    BurnMlp(BurnActorCritic<Autodiff<NdArray>>),
    CandleMlp(CandleActorCritic),
    CandleCnn(CandleCnnActorCritic),      // new
    CandleTransformer(CandleTransformerAC), // new
}
```

Python API:

```python
# MLP (existing)
policy = rlox.ActorCritic("candle", obs_dim=4, n_actions=2)

# CNN (new)
policy = rlox.CnnActorCritic("candle", n_actions=18, frame_stack=4)

# Transformer (new)
policy = rlox.TransformerActorCritic("candle", n_entities=10, entity_dim=8, n_actions=5)
```

**Pros**: Simple, no dynamic dispatch overhead, full type safety.
**Cons**: Every new architecture requires changes to `rlox-python`.

#### Approach B: Dynamic Registration via Python Trait Objects (Advanced)

Allow Python users to register custom Rust architectures via a plugin system:

```rust
/// A type-erased wrapper around any ActorCritic implementation.
#[pyclass(name = "RustActorCritic", unsendable)]
pub struct PyRustActorCritic {
    inner: Box<dyn ActorCritic>,
    obs_dim: usize,
}
```

This requires the custom architecture to be compiled into a shared library that rlox can load. This is complex and not recommended for the initial implementation.

#### Recommended: Approach A with a Builder Pattern

```rust
#[pyclass(name = "ActorCriticBuilder")]
pub struct PyActorCriticBuilder {
    backend: String,
    obs_dim: usize,
    n_actions: usize,
    // Optional architecture config
    architecture: Architecture,
}

enum Architecture {
    Mlp { hidden: Vec<usize> },
    Cnn { frame_stack: usize, frame_size: (usize, usize) },
    Custom { config_json: String },
}
```

### 6.3 Serialization: Saving and Loading Rust Models from Python

Both backends already implement `save`/`load` on the `ActorCritic` trait:

- **Candle**: `VarMap::save(path)` / `VarMap::load(path)` -- uses safetensors format.
- **Burn**: `params.save_file(path, &recorder)` / `params.load_file(path, &recorder, &device)` -- uses Burn's binary format.

The Python bridge should expose these:

```python
policy = rlox.ActorCritic("candle", obs_dim=4, n_actions=2)
# ... train ...
policy.save("model.safetensors")

# Later:
policy = rlox.ActorCritic("candle", obs_dim=4, n_actions=2)
policy.load("model.safetensors")
```

**Missing**: The `save`/`load` methods are not yet exposed in `PyActorCritic`. This is a straightforward addition:

```rust
#[pymethods]
impl PyActorCritic {
    fn save(&self, path: &str) -> PyResult<()> {
        let path = Path::new(path);
        match &self.inner {
            Backend::Burn(ac) => ac.save(path),
            Backend::Candle(ac) => ac.save(path),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn load(&mut self, path: &str) -> PyResult<()> {
        let path = Path::new(path);
        match &mut self.inner {
            Backend::Burn(ac) => ac.load(path),
            Backend::Candle(ac) => ac.load(path),
        }
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
```

### 6.4 Can Users Register Custom Rust Networks from Python?

Not directly at runtime. Rust code must be compiled ahead of time. However, we can support a **plugin crate** pattern:

1. User creates a Rust crate that depends on `rlox-nn` and `rlox-candle`.
2. User implements `ActorCritic` for their custom architecture.
3. User creates a PyO3 `#[pymodule]` that wraps their custom type.
4. User imports both `rlox` and their custom crate in Python.

```python
import rlox
import my_custom_policy  # separate PyO3 crate

policy = my_custom_policy.MyCnnPolicy(n_actions=18)
# policy implements the same act/value/ppo_step interface

# Can use with rlox's training utilities:
collector = rlox.AsyncCollector(envs, policy.act, policy.value, ...)
```

This requires documenting the trait contracts clearly and providing example template crates.

---

## 7. Hybrid Patterns

### 7.1 Rust Feature Extractor + Python Policy Head

This is useful when the feature extraction is compute-heavy (CNN on images) but the policy logic is experimental (custom distribution, meta-learning).

```python
import torch
import rlox

# Rust-accelerated CNN feature extraction
extractor = rlox.NatureCNN("candle")  # returns numpy features

class HybridPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = rlox.NatureCNN("candle")
        self.actor = torch.nn.Linear(512, 18)
        self.critic = torch.nn.Linear(512, 1)

    def forward(self, obs):
        # obs: [batch, 84*84*4] numpy/torch
        features = self.extractor.forward(obs.numpy())  # Rust CNN
        features = torch.from_numpy(features)  # back to PyTorch
        return self.actor(features), self.critic(features)
```

**Limitation**: The Rust-to-Python boundary copies data twice (Rust tensor -> numpy -> PyTorch tensor). For CNNs processing 84x84x4 frames, the feature vector is only 512 floats -- the copy is negligible. The CNN forward pass savings dominate.

**Implementation note**: This requires exposing the `NatureCNN::forward()` as a standalone Python function that accepts and returns numpy arrays, separate from the full `ActorCritic` interface.

### 7.2 Python Feature Extractor + Rust Training Step

This is the current design with `AsyncCollector`: Python owns the network, Rust handles env stepping and buffer management.

```python
import torch
from rlox import VecEnv, AsyncCollector, compute_gae

# Python policy (any architecture)
policy = MyCustomTransformerPolicy()

# Rust environments + collection
envs = VecEnv("CartPole-v1", n_envs=16)
collector = AsyncCollector(
    envs, n_steps=128,
    action_fn=lambda obs: policy.act(torch.from_numpy(obs)),
    value_fn=lambda obs: policy.value(torch.from_numpy(obs)),
)

# Rust-accelerated GAE
batch = collector.recv()
advantages, returns = compute_gae(batch.rewards, batch.values, batch.dones, ...)

# Python training step (full PyTorch autograd)
policy.ppo_step(batch.obs, batch.actions, advantages, returns)
```

### 7.3 When Does Pure-Rust Training Beat PyTorch?

Based on the existing benchmarks (Apple M4):

| Component | Rust/PyTorch Ratio | When Rust Wins |
|-----------|--------------------|----------------|
| Env stepping (CartPole) | 37ns vs ~1us | Always for CPU envs |
| GAE computation | 142x vs numpy | Always |
| Buffer push | 61x vs TorchRL | Always |
| Buffer sample | 6-13x | Always |
| MLP forward (small) | ~1-2x | Marginal for small networks |
| CNN forward (Candle CPU) | ~0.5-1x vs PyTorch CPU | Rarely -- PyTorch MKL/BLAS tuning wins |
| CNN forward (GPU) | N/A | Only if using Burn with Wgpu/CUDA backend |
| E2E PPO rollout | 3x vs SB3 | When env stepping dominates |

**Summary**: Pure-Rust training wins decisively when:
1. **CPU-bound environments** (robotics sim, custom envs, Classic Control) -- env stepping dominates.
2. **Small networks** (MLP with 64-256 hidden) -- the Rust overhead savings outweigh missing BLAS optimizations.
3. **High env-to-compute ratio** (many parallel envs, short episodes) -- maximizes Rust's env stepping advantage.
4. **Deployment** -- single binary, no Python/PyTorch dependency, deterministic.

Pure-Rust training is **not** the right choice when:
1. **Large CNNs/Transformers on GPU** -- PyTorch + CUDA is heavily optimized.
2. **Rapid prototyping** -- Python iteration speed matters more than runtime.
3. **Exotic architectures** -- if you need PyTorch's ecosystem (custom CUDA kernels, Hugging Face models).

---

## 8. Concrete Recommendations

### Priority 1: Foundation (Effort: ~2 weeks)

#### 1a. `FeatureExtractor` trait in rlox-nn

Add to `rlox-nn/src/traits.rs`:

```rust
/// Feature extractor that maps observations to a fixed-size feature vector.
/// This is the primary extension point for custom architectures.
pub trait FeatureExtractor: Send {
    fn feature_dim(&self) -> usize;
    fn extract(&self, obs: &TensorData) -> Result<TensorData, NNError>;
}
```

**Effort**: 1 day. No breaking changes. Tests: shape validation, object safety.

#### 1b. PPO loss computation as reusable function

Extract the PPO loss math from `BurnActorCritic::ppo_step()` and `CandleActorCritic::ppo_step()` into shared helper functions:

```rust
// In rlox-candle:
pub fn candle_ppo_loss(
    logits: &Tensor,          // [batch, n_actions]
    values: &Tensor,          // [batch]
    actions: &Tensor,         // [batch] (int indices)
    old_log_probs: &Tensor,   // [batch]
    advantages: &Tensor,      // [batch]
    returns: &Tensor,         // [batch]
    old_values: &Tensor,      // [batch]
    config: &PPOStepConfig,
) -> Result<(Tensor, TrainMetrics), NNError>;
```

This way, a custom architecture only needs to produce `logits` and `values` -- the loss computation is shared.

**Effort**: 2-3 days. Refactoring existing code, no new logic. Tests: verify identical outputs before/after refactor.

#### 1c. Composable actor-critic constructors

Create `CandleComposableActorCritic` and `BurnComposableActorCritic` that take a feature extractor + heads:

```rust
// In rlox-candle:
impl CandleComposableActorCritic {
    pub fn with_extractor(
        extractor_fn: impl FnOnce(VarBuilder) -> candle_core::Result<Box<dyn CandleModule>>,
        feat_dim: usize,
        n_actions: usize,
        lr: f64,
        device: Device,
        seed: u64,
    ) -> Result<Self, NNError>;
}
```

**Effort**: 3-4 days. New code, but follows existing patterns.

### Priority 2: Rust Training Loop (Effort: ~2 weeks)

#### 2a. SyncCollector in rlox-core

Synchronous rollout collection using `ActorCritic` trait objects directly.

```rust
// rlox-core/src/training/rollout.rs
pub fn collect_rollout(
    envs: &mut dyn BatchSteppable,
    policy: &dyn ActorCritic,
    current_obs: &mut Vec<f32>,
    n_steps: usize,
    obs_dim: usize,
    act_dim: usize,
) -> Result<PPORolloutBatch, RloxError>;
```

**Effort**: 3-4 days. Similar to `AsyncCollector` but simpler (no threads, no closures). Tests: compare rollout shapes, verify GAE values match.

#### 2b. MinibatchIterator

Random shuffling and chunking of rollout data into minibatches.

**Effort**: 2 days. Pure data manipulation. Tests: coverage of all samples, correct shapes.

#### 2c. PPOTrainer

Ties everything together: collect -> GAE -> minibatch -> ppo_step -> log.

**Effort**: 3-4 days. Integration code. Tests: loss decreases on CartPole, matches Python PPO convergence curve.

#### 2d. LR scheduler + basic callbacks

```rust
pub trait TrainingCallback {
    fn on_update(&mut self, metrics: &UpdateMetrics);
    fn should_stop(&self) -> bool { false }
}
```

**Effort**: 2 days. Simple traits + implementations (ConsoleLogger, EarlyStop).

### Priority 3: Built-in Architectures (Effort: ~2 weeks)

#### 3a. NatureCNN for Candle and Burn

As designed in Sections 4.1 and 5.1. Both implementing `ActorCritic` with the refactored PPO loss.

**Effort**: 3-4 days per backend.

#### 3b. PyO3 exposure

Add `CnnActorCritic` to Python bridge, plus `save`/`load` on existing `PyActorCritic`.

**Effort**: 2-3 days.

### Priority 4: Advanced Architectures (Effort: ~3 weeks)

#### 4a. Transformer extractor

As designed in Section 4.2. Useful for entity-based environments.

**Effort**: 1 week including tests and Python bridge.

#### 4b. Recurrent (GRU) policy

Requires changes to the rollout collector to pass hidden states and episode masks. This is a more invasive change.

**Effort**: 1.5 weeks. Includes collector changes, new `RecurrentActorCritic` trait variant, tests.

#### 4c. SAC/TD3 composable actor steps

Redesign `sac_actor_step` to work with composable extractors. This may require a `CandlePolicyWithExtractor` trait that provides both `forward()` (tensor-level, with autograd) and `extract()` (TensorData, inference-only).

**Effort**: 1 week. Design-heavy.

### Priority 5: Documentation and Examples (Effort: ~1 week)

- Example crate: `examples/custom_cnn_ppo/` -- full CNN PPO on Atari in pure Rust.
- Example crate: `examples/custom_extractor/` -- how to write a FeatureExtractor.
- Tutorial: "Writing a Custom Rust Policy" in docs.
- Tutorial: "Full-Rust Training Loop" in docs.
- API docs: all new traits and types.

---

## Architecture Diagram

```
                          rlox-nn (traits)
                    FeatureExtractor    ActorCritic    QFunction ...
                         |                  |              |
            +------------+--------+---------+----+---------+
            |                     |              |
        rlox-candle           rlox-burn      rlox-core
   CandleFeatureExtractor  BurnFeatureExtractor  SyncCollector
   CandleComposableAC      BurnComposableAC      PPOTrainer
   NatureCNN               NatureCNN             MinibatchIterator
   TransformerExtractor    TransformerExtractor   LRScheduler
        |                     |                      |
        +----------+----------+----------+-----------+
                   |                     |
              rlox-python           examples/
         PyActorCritic            custom_cnn_ppo/
         PyCnnActorCritic         full_rust_ppo/
         save/load
```

## Summary of Changes by Crate

| Crate | Changes | New Files |
|-------|---------|-----------|
| `rlox-nn` | Add `FeatureExtractor` trait | -- |
| `rlox-candle` | Refactor PPO loss to helper fn, add `CandleComposableAC`, `NatureCNN` | `composable.rs`, `cnn.rs`, `ppo_loss.rs` |
| `rlox-burn` | Same as candle | `composable.rs`, `cnn.rs`, `ppo_loss.rs` |
| `rlox-core` | Add `SyncCollector`, `PPOTrainer`, `MinibatchIterator`, `LRScheduler` | `training/rollout.rs`, `training/ppo_trainer.rs`, `training/lr_schedule.rs`, `training/callbacks.rs` |
| `rlox-python` | Add `save`/`load`, `CnnActorCritic`, `ActorCriticBuilder` | -- (extend `nn.rs`) |
| new | Example crates | `examples/custom_cnn_ppo/`, `examples/custom_extractor/` |

## Total Estimated Effort

| Priority | Description | Effort |
|----------|-------------|--------|
| P1 | Foundation (traits, refactored loss, composable AC) | ~2 weeks |
| P2 | Rust training loop (collector, minibatch, trainer) | ~2 weeks |
| P3 | Built-in architectures (CNN) | ~2 weeks |
| P4 | Advanced architectures (transformer, RNN, SAC/TD3) | ~3 weeks |
| P5 | Documentation and examples | ~1 week |
| **Total** | | **~10 weeks** |

P1 and P2 are the highest-value items -- they unlock the full-Rust training loop and the extensibility pattern. P3-P5 build on that foundation incrementally.
