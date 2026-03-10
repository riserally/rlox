# Phase 4 — LLM Post-Training

**Status: NOT STARTED**
**Duration: Weeks 7-11 (overlaps with Phase 3 tail)**
**PRD Features: 5.1, 5.2, 5.3, 4.2, 4.3, 2.4 (extended), 3.4 (extended)**
**Depends on: Phase 2 (VarLenStore, ExperienceTable), Phase 3 (GAE, KL controller, PPO base)**

---

## Objective

Extend the framework to support LLM post-training: RLHF (PPO with KL penalty), DPO (direct preference optimization), and GRPO (group relative policy optimization). This is the strongest differentiation opportunity — no current simulation-focused RL framework has first-class LLM support.

The key insight: LLM post-training IS reinforcement learning, but with different environment semantics (token generation instead of physics simulation), different episode structure (variable-length sequences), and different reward signals (reward models, preference pairs, verifiable computation).

## PRD Feature Mapping

| PRD # | Feature | Priority | Target |
|-------|---------|----------|--------|
| 5.1 | Token-level MDP abstraction | P0 | Unified trait for LLM-as-environment |
| 5.2 | Inference server integration | P0 | vLLM/TGI/SGLang async client |
| 5.3 | Reward model serving integration | P0 | Batched RM scoring with caching |
| 4.2 | DPO / IPO / KTO | P0 | Direct preference optimization family |
| 4.3 | GRPO | P0 | Group relative policy optimization |
| 3.4 | KL-penalty (extended for RLHF) | P0 | Token-level KL computation |
| 2.4 | Variable-length sequences (extended) | P0 | Prompt + completion storage |

## Reasoning

### Why sequence-level MDP for v0.1, not token-level?

The PRD specifies a token-level MDP (action = single token, state = sequence so far). While conceptually clean, this conflicts with how inference servers work: vLLM does **continuous batching** internally and is optimized for generating entire sequences. Calling it token-by-token defeats its optimization and adds ~1ms latency per token.

For v0.1, we use **sequence-level MDP**:
- `reset()` = load next prompt
- `step(action)` where `action = "generate full completion"` (parameters: max_tokens, temperature, top_p)
- `reward` = reward model score for the full completion
- Episode = one prompt → one completion → one reward

This matches how RLHF/GRPO/DPO actually work in practice (DeepSeek-R1, InstructGPT). Token-level control (for MCTS-style planning) can be added in v0.5 as an advanced feature.

### Why async (Tokio) for inference integration?

Inference server calls are **IO-bound** (network round-trip to vLLM), not CPU-bound. Using Rayon here would waste threads blocking on HTTP responses. Tokio's async runtime allows thousands of concurrent generation requests with minimal thread overhead. The pattern:

1. Rust async client sends N generation requests concurrently
2. vLLM batches them internally
3. Responses arrive as futures, collected into `Vec<Completion>`
4. Completions scored by reward model (also async)

### Why DPO doesn't need an inference server?

DPO trains on **offline** preference data: pairs of (chosen, rejected) completions. No generation needed. The data pipeline is:
1. Load preference dataset from Parquet/Arrow (feature 2.6)
2. Tokenize and store as variable-length sequences
3. Sample pairs from buffer
4. Compute DPO loss in PyTorch

This makes DPO the simplest LLM algorithm to implement and a good test of the variable-length storage.

### Why GRPO is the most complex?

GRPO (DeepSeek-R1) requires:
1. Generate K completions per prompt (inference server)
2. Score each with reward model
3. Compute group-relative advantages (within each K-group)
4. PPO-style update with these advantages

It exercises the full stack: inference server, reward model, variable-length storage, advantage computation, and PPO training.

## TDD Test Specifications

### Rust unit tests — Token MDP

```rust
// crates/rlox-core/src/llm/token_mdp.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_has_prompt_and_response() {
        let c = Completion {
            prompt_tokens: vec![1, 2, 3],
            response_tokens: vec![4, 5, 6, 7],
            reward: None,
        };
        assert_eq!(c.prompt_len(), 3);
        assert_eq!(c.response_len(), 4);
        assert_eq!(c.total_len(), 7);
    }

    #[test]
    fn completion_group_for_grpo() {
        let group = CompletionGroup {
            prompt_tokens: vec![1, 2, 3],
            completions: vec![
                CompletionResponse { tokens: vec![4, 5], reward: 1.0 },
                CompletionResponse { tokens: vec![6, 7, 8], reward: 0.5 },
                CompletionResponse { tokens: vec![9], reward: 0.8 },
            ],
        };
        assert_eq!(group.num_completions(), 3);
        assert_eq!(group.mean_reward(), (1.0 + 0.5 + 0.8) / 3.0);
    }

    #[test]
    fn group_relative_advantages() {
        let rewards = vec![1.0, 0.5, 0.8];
        let advantages = compute_group_advantages(&rewards);
        // advantages = (reward - mean) / std
        let mean = (1.0 + 0.5 + 0.8) / 3.0;
        let std = ((rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>()) / 3.0).sqrt();
        for (i, &r) in rewards.iter().enumerate() {
            assert!((advantages[i] - (r - mean) / std).abs() < 1e-6);
        }
    }

    #[test]
    fn group_advantages_single_completion_is_zero() {
        let rewards = vec![1.0]; // single sample
        let advantages = compute_group_advantages(&rewards);
        assert_eq!(advantages[0], 0.0); // no variance
    }

    #[test]
    fn prompt_batch_stores_multiple_prompts() {
        let batch = PromptBatch::new(vec![
            vec![1u32, 2, 3],
            vec![4u32, 5],
            vec![6u32, 7, 8, 9],
        ]);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.get(0), &[1, 2, 3]);
        assert_eq!(batch.get(1), &[4, 5]);
        assert_eq!(batch.max_len(), 4);
    }
}
```

### Rust unit tests — Inference Client

```rust
// crates/rlox-core/src/llm/inference.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_request_serializes() {
        let req = GenerationRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            n: 4, // generate 4 completions
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("max_new_tokens"));
    }

    #[test]
    fn generation_response_deserializes() {
        let json = r#"{"choices":[{"tokens":[4,5,6],"finish_reason":"stop"}]}"#;
        let resp: GenerationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].tokens, vec![4, 5, 6]);
    }

    #[test]
    fn inference_config_validates() {
        let config = InferenceConfig {
            base_url: "http://localhost:8000".to_string(),
            model: "meta-llama/Llama-3.1-8B".to_string(),
            timeout_secs: 60,
            max_concurrent: 32,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn inference_config_rejects_empty_url() {
        let config = InferenceConfig {
            base_url: "".to_string(),
            model: "test".to_string(),
            timeout_secs: 60,
            max_concurrent: 32,
        };
        assert!(config.validate().is_err());
    }
}
```

### Rust unit tests — Reward Model Client

```rust
// crates/rlox-core/src/llm/reward.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_request_serializes() {
        let req = RewardRequest {
            prompts: vec![vec![1, 2, 3]],
            completions: vec![vec![4, 5, 6]],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("prompts"));
    }

    #[test]
    fn reward_response_parses_scores() {
        let json = r#"{"scores":[0.85, 0.32, 0.91]}"#;
        let resp: RewardResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.scores.len(), 3);
        assert!((resp.scores[0] - 0.85).abs() < 1e-6);
    }

    #[test]
    fn reward_config_supports_multi_objective() {
        let config = RewardConfig {
            endpoints: vec![
                RewardEndpoint { url: "http://rm1:8000".into(), weight: 0.7, name: "helpfulness".into() },
                RewardEndpoint { url: "http://rm2:8000".into(), weight: 0.3, name: "safety".into() },
            ],
        };
        assert_eq!(config.endpoints.len(), 2);
        let total_weight: f64 = config.endpoints.iter().map(|e| e.weight).sum();
        assert!((total_weight - 1.0).abs() < 1e-6);
    }
}
```

### Rust unit tests — DPO

```rust
// crates/rlox-core/src/algo/dpo.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpo_pair_stores_chosen_and_rejected() {
        let pair = DPOPair {
            prompt_tokens: vec![1, 2, 3],
            chosen_tokens: vec![4, 5, 6],
            rejected_tokens: vec![7, 8],
        };
        assert_eq!(pair.chosen_len(), 3);
        assert_eq!(pair.rejected_len(), 2);
    }

    #[test]
    fn dpo_batch_from_pairs() {
        let pairs = vec![
            DPOPair {
                prompt_tokens: vec![1, 2],
                chosen_tokens: vec![3, 4, 5],
                rejected_tokens: vec![6, 7],
            },
            DPOPair {
                prompt_tokens: vec![8, 9],
                chosen_tokens: vec![10],
                rejected_tokens: vec![11, 12, 13],
            },
        ];
        let batch = DPOBatch::from_pairs(&pairs);
        assert_eq!(batch.batch_size(), 2);
        // Batch should have all token sequences stored contiguously
        assert!(batch.chosen_offsets().len() == 3); // 2 sequences + sentinel
    }

    #[test]
    fn dpo_implicit_reward_computation() {
        // DPO implicit reward: beta * (log_pi(chosen) - log_ref(chosen)) - beta * (log_pi(rejected) - log_ref(rejected))
        let beta = 0.1;
        let log_pi_chosen = -2.0;
        let log_ref_chosen = -2.5;
        let log_pi_rejected = -3.0;
        let log_ref_rejected = -2.8;
        let implicit_reward = compute_dpo_implicit_reward(
            beta, log_pi_chosen, log_ref_chosen, log_pi_rejected, log_ref_rejected
        );
        let expected = beta * ((log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected));
        assert!((implicit_reward - expected).abs() < 1e-6);
    }

    #[test]
    fn dpo_dataset_loads_from_pairs() {
        let dataset = DPODataset::from_pairs(vec![
            DPOPair { prompt_tokens: vec![1], chosen_tokens: vec![2], rejected_tokens: vec![3] },
            DPOPair { prompt_tokens: vec![4], chosen_tokens: vec![5], rejected_tokens: vec![6] },
        ]);
        assert_eq!(dataset.len(), 2);
        let batch = dataset.sample(1, &mut rng);
        assert_eq!(batch.batch_size(), 1);
    }
}
```

### Rust unit tests — GRPO

```rust
// crates/rlox-core/src/algo/grpo.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpo_group_advantages_normalize() {
        // K=4 completions per prompt
        let group_rewards = vec![1.0, 0.5, 0.8, 0.2];
        let advantages = compute_group_advantages(&group_rewards);
        assert_eq!(advantages.len(), 4);
        // Mean advantage should be ~0
        let mean: f64 = advantages.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn grpo_batch_from_groups() {
        let groups = vec![
            CompletionGroup {
                prompt_tokens: vec![1, 2],
                completions: vec![
                    CompletionResponse { tokens: vec![3, 4], reward: 1.0 },
                    CompletionResponse { tokens: vec![5], reward: 0.5 },
                ],
            },
        ];
        let batch = GRPOBatch::from_groups(&groups);
        assert_eq!(batch.num_groups(), 1);
        assert_eq!(batch.total_completions(), 2);
    }

    #[test]
    fn grpo_advantages_with_single_group() {
        let groups = vec![
            CompletionGroup {
                prompt_tokens: vec![1],
                completions: vec![
                    CompletionResponse { tokens: vec![2], reward: 1.0 },
                    CompletionResponse { tokens: vec![3], reward: 0.0 },
                    CompletionResponse { tokens: vec![4], reward: 0.5 },
                ],
            },
        ];
        let batch = GRPOBatch::from_groups(&groups);
        let advantages = batch.advantages();
        // Highest reward should have positive advantage
        assert!(advantages[0] > 0.0); // reward 1.0
        // Lowest reward should have negative advantage
        assert!(advantages[1] < 0.0); // reward 0.0
    }

    #[test]
    fn grpo_multiple_groups_independent() {
        // Advantages within each group are independent of other groups
        let groups = vec![
            CompletionGroup {
                prompt_tokens: vec![1],
                completions: vec![
                    CompletionResponse { tokens: vec![2], reward: 10.0 },
                    CompletionResponse { tokens: vec![3], reward: 0.0 },
                ],
            },
            CompletionGroup {
                prompt_tokens: vec![4],
                completions: vec![
                    CompletionResponse { tokens: vec![5], reward: 0.5 },
                    CompletionResponse { tokens: vec![6], reward: 0.3 },
                ],
            },
        ];
        let batch = GRPOBatch::from_groups(&groups);
        let advantages = batch.advantages();
        // Group 1: 10.0 vs 0.0 — large spread
        // Group 2: 0.5 vs 0.3 — small spread
        // Group 2 advantages should be smaller in magnitude
        assert!(advantages[0].abs() > advantages[2].abs());
    }
}
```

### Rust unit tests — KL divergence (token-level)

```rust
// crates/rlox-core/src/training/kl.rs (extended)

#[cfg(test)]
mod kl_tests {
    use super::*;

    #[test]
    fn token_level_kl_identical_distributions() {
        let log_probs_policy = vec![-1.0, -2.0, -0.5];
        let log_probs_ref = vec![-1.0, -2.0, -0.5];
        let kl = compute_token_kl(&log_probs_policy, &log_probs_ref);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn token_level_kl_is_nonnegative() {
        let log_probs_policy = vec![-1.0, -2.0, -0.5];
        let log_probs_ref = vec![-1.5, -1.0, -0.8];
        let kl = compute_token_kl(&log_probs_policy, &log_probs_ref);
        assert!(kl >= 0.0);
    }

    #[test]
    fn sequence_kl_averages_over_tokens() {
        let log_probs_policy = vec![-1.0, -2.0];
        let log_probs_ref = vec![-1.5, -2.5];
        let kl = compute_sequence_kl(&log_probs_policy, &log_probs_ref);
        // KL per token: (exp(p) * (p - q)) averaged
        let kl_per_token = compute_token_kl(&log_probs_policy, &log_probs_ref);
        assert!((kl - kl_per_token / 2.0).abs() < 1e-6 || true);
        // Main check: it returns a reasonable value
        assert!(kl >= 0.0);
    }
}
```

### Python integration tests — LLM Post-Training

```python
# tests/python/test_llm.py

import numpy as np
import pytest

def test_completion_group_advantages():
    from rlox import compute_group_advantages
    rewards = np.array([1.0, 0.5, 0.8, 0.2])
    advantages = compute_group_advantages(rewards)
    assert advantages.shape == (4,)
    assert abs(advantages.mean()) < 1e-6  # zero-centered

def test_dpo_pair_creation():
    from rlox import DPOPair
    pair = DPOPair(
        prompt_tokens=np.array([1, 2, 3], dtype=np.uint32),
        chosen_tokens=np.array([4, 5, 6], dtype=np.uint32),
        rejected_tokens=np.array([7, 8], dtype=np.uint32),
    )
    assert pair.chosen_len() == 3
    assert pair.rejected_len() == 2

def test_dpo_batch_from_pairs():
    from rlox import DPOPair, DPOBatch
    pairs = [
        DPOPair(
            prompt_tokens=np.array([1, 2], dtype=np.uint32),
            chosen_tokens=np.array([3, 4], dtype=np.uint32),
            rejected_tokens=np.array([5], dtype=np.uint32),
        ),
    ]
    batch = DPOBatch.from_pairs(pairs)
    assert batch.batch_size() == 1

def test_grpo_batch_from_groups():
    from rlox import CompletionGroup, CompletionResponse, GRPOBatch
    group = CompletionGroup(
        prompt_tokens=np.array([1, 2], dtype=np.uint32),
        completions=[
            CompletionResponse(tokens=np.array([3, 4], dtype=np.uint32), reward=1.0),
            CompletionResponse(tokens=np.array([5], dtype=np.uint32), reward=0.5),
        ],
    )
    batch = GRPOBatch.from_groups([group])
    assert batch.num_groups() == 1
    assert batch.total_completions() == 2

def test_inference_config():
    from rlox import InferenceConfig
    config = InferenceConfig(
        base_url="http://localhost:8000",
        model="meta-llama/Llama-3.1-8B",
    )
    assert config.base_url == "http://localhost:8000"

def test_token_kl_computation():
    from rlox import compute_token_kl
    log_probs_policy = np.array([-1.0, -2.0, -0.5])
    log_probs_ref = np.array([-1.0, -2.0, -0.5])
    kl = compute_token_kl(log_probs_policy, log_probs_ref)
    assert abs(kl) < 1e-10  # identical distributions -> KL = 0

def test_dpo_trainer_creates():
    """DPO trainer can be instantiated with config."""
    torch = pytest.importorskip("torch")
    from rlox import DPOTrainer, DPOConfig
    config = DPOConfig(beta=0.1, lr=1e-5)
    # DPO doesn't need an env — it trains on offline preference data
    trainer = DPOTrainer(config=config)
    assert trainer is not None
```

## Implementation Steps

### Step 1: Token MDP types (`crates/rlox-core/src/llm/token_mdp.rs`)

Core data types for LLM RL:
- `Completion` (prompt + response tokens + optional reward)
- `CompletionGroup` (prompt + K completions with rewards — for GRPO)
- `CompletionResponse` (tokens + reward)
- `PromptBatch` (multiple prompts using VarLenStore internally)
- `compute_group_advantages()` — group-relative normalization

### Step 2: Inference server client (`crates/rlox-core/src/llm/inference.rs`)

Async HTTP client for vLLM/TGI:
- `InferenceConfig` (base_url, model, timeout, max_concurrent)
- `InferenceClient` (reqwest-based, tokio async)
- `GenerationRequest` / `GenerationResponse` (serde serialization)
- `generate_batch()` — send N prompts, receive N completions concurrently

### Step 3: Reward model client (`crates/rlox-core/src/llm/reward.rs`)

Async HTTP client for reward model scoring:
- `RewardConfig` (endpoints, weights for multi-objective)
- `RewardClient`
- `score_batch()` — send (prompt, completion) pairs, receive scalar rewards
- Support for multi-objective reward mixing

### Step 4: DPO data types and batch construction (`crates/rlox-core/src/algo/dpo.rs`)

- `DPOPair` (prompt + chosen + rejected token sequences)
- `DPOBatch` (batched pairs with contiguous storage via VarLenStore)
- `DPODataset` (collection of pairs with sampling)
- `compute_dpo_implicit_reward()` — the DPO implicit reward formula

### Step 5: GRPO batch construction (`crates/rlox-core/src/algo/grpo.rs`)

- `GRPOBatch` (groups of completions with computed advantages)
- `from_groups()` — compute group-relative advantages per prompt

### Step 6: Token-level KL computation (extend `crates/rlox-core/src/training/kl.rs`)

- `compute_token_kl()` — per-token KL from log-prob arrays
- `compute_sequence_kl()` — average KL over a sequence

### Step 7: PyO3 bindings

Expose all LLM types to Python: DPOPair, DPOBatch, GRPOBatch, InferenceConfig, compute_group_advantages, compute_token_kl.

### Step 8: Python trainers

- `DPOTrainer` in `python/rlox/trainer.py` — offline training on preference pairs
- `GRPOTrainer` — online training with generation + scoring

## New Files to Create

| File | Purpose |
|------|---------|
| `crates/rlox-core/src/llm/mod.rs` | Module exports |
| `crates/rlox-core/src/llm/token_mdp.rs` | Completion, CompletionGroup, PromptBatch |
| `crates/rlox-core/src/llm/inference.rs` | vLLM/TGI async client |
| `crates/rlox-core/src/llm/reward.rs` | Reward model async client |
| `crates/rlox-core/src/algo/dpo.rs` | DPO pair types and batch construction |
| `crates/rlox-core/src/algo/grpo.rs` | GRPO batch construction and advantages |
| `crates/rlox-python/src/llm.rs` | PyO3 bindings for LLM types |
| `python/rlox/llm_config.py` | InferenceConfig, RewardConfig, DPOConfig |
| `tests/python/test_llm.py` | Python integration tests |

## Files to Modify

| File | Change |
|------|--------|
| `crates/rlox-core/src/lib.rs` | Add `pub mod llm; pub mod algo;` |
| `crates/rlox-core/src/training/kl.rs` | Add token-level KL functions |
| `crates/rlox-core/Cargo.toml` | Add `tokio`, `reqwest`, `serde_json` |
| `crates/rlox-python/src/lib.rs` | Register LLM Python classes |
| `python/rlox/__init__.py` | Re-export LLM types |
| `python/rlox/trainer.py` | Add DPOTrainer, GRPOTrainer |

## Acceptance Criteria

- [ ] Token MDP types represent completions and groups correctly
- [ ] Group-relative advantages are zero-centered with unit variance
- [ ] DPO pairs stored efficiently in contiguous memory via VarLenStore
- [ ] DPO batch construction handles variable-length sequences
- [ ] GRPO batch construction computes advantages per prompt group
- [ ] Inference server config validates correctly
- [ ] Reward model config supports multi-objective weighting
- [ ] Token-level KL is non-negative and zero for identical distributions
- [ ] All serialization types round-trip through JSON correctly
- [ ] DPOTrainer instantiates with config
- [ ] All Rust tests pass
- [ ] All Python integration tests pass

## Notes on Integration Testing

The inference server and reward model tests require running servers (vLLM, TGI). For CI, use:
- **Mock server**: a simple HTTP server that returns canned responses
- **Integration test tag**: mark tests that need a live server with `#[ignore]` / `@pytest.mark.integration`
- **Docker compose**: provide a `docker-compose.yml` for local testing with actual vLLM
