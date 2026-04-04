# GRPO -- Group Relative Policy Optimization

## Intuition

GRPO is a policy optimization algorithm designed for LLM post-training. Instead of training a separate value function (as in PPO), GRPO estimates advantages by generating a group of completions for each prompt and normalizing rewards within the group. Completions that score above the group average receive positive advantage; those below receive negative. This group-relative normalization eliminates the need for a critic network, simplifying the training pipeline and reducing memory footprint. A KL penalty against a frozen reference model prevents the policy from drifting too far.

## Key Equations

For each prompt $x$, generate $G$ completions $\{y_1, \ldots, y_G\}$ with rewards $\{r_1, \ldots, r_G\}$.

**Group-relative advantages:**

$$
A_i = \frac{r_i - \mu_G}{\sigma_G}, \quad \mu_G = \frac{1}{G} \sum_j r_j, \quad \sigma_G = \text{std}(\{r_j\})
$$

**GRPO loss:**

$$
L(\theta) = -\mathbb{E}_{x, \{y_i\}} \left[ \sum_{i=1}^{G} A_i \sum_{t} \log \pi_\theta(y_{i,t} | x, y_{i,<t}) \right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

**Per-token KL penalty:**

$$
\text{KL} = \mathbb{E}_t \left[ \log \pi_\theta(y_t | \cdot) - \log \pi_{\text{ref}}(y_t | \cdot) \right]
$$

## Pseudocode

```
algorithm GRPO:
    initialize policy model pi_theta
    freeze reference model pi_ref = copy(pi_theta)
    set group_size G, KL coefficient beta

    for each batch of prompts do
        for each prompt x:
            generate G completions {y_1, ..., y_G} ~ pi_theta
            compute rewards {r_1, ..., r_G} via reward_fn

        # Group-relative advantages (batched via Rust)
        A = compute_batch_group_advantages(rewards, G)

        # Per-token log probs
        logprobs_policy = get_per_token_logprobs(pi_theta, completions)
        logprobs_ref = get_per_token_logprobs(pi_ref, completions)

        # Loss
        seq_logprobs = sum(logprobs_policy, dim=tokens)
        loss = -mean(A * seq_logprobs)
        kl = mean(logprobs_policy - logprobs_ref)
        loss = loss + beta * kl

        update theta with gradient clipping
```

## Quick Start

```python
import torch.nn as nn
from rlox.algorithms.grpo import GRPO

model = MyLanguageModel()        # forward(input_ids) -> logits
ref_model = copy.deepcopy(model)
ref_model.eval()

def reward_fn(completions, prompts):
    # Return list of float rewards
    return [score(c) for c in completions]

trainer = GRPO(
    model=model,
    ref_model=ref_model,
    reward_fn=reward_fn,
    group_size=4,
    kl_coef=0.1,
    learning_rate=1e-4,
    max_new_tokens=8,
)
metrics = trainer.train(prompts=prompt_tensor, n_epochs=3)
print(f"Mean reward: {metrics['mean_reward']:.3f}, KL: {metrics['kl']:.4f}")
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | `4` | Number of completions generated per prompt |
| `kl_coef` | `0.1` | KL penalty coefficient against reference model |
| `learning_rate` | `1e-4` | Adam learning rate |
| `max_new_tokens` | `8` | Maximum tokens to generate per completion |
| `max_grad_norm` | `1.0` | Gradient clipping norm |

## When to Use

- **Use GRPO when:** you are doing LLM post-training (RLHF) and want to avoid the complexity of training a separate value network.
- **Prefer GRPO over PPO for LLMs when:** memory is constrained (no critic network needed), or the reward signal is well-suited to group-relative comparison.
- **Do not use GRPO when:** you need per-token advantage estimates (PPO with a critic may be more precise), or you are doing standard RL (not LLM training).

## References

- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.
