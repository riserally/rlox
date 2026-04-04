# DPO -- Direct Preference Optimization

## Intuition

DPO bypasses the need for an explicit reward model by directly optimizing a language model on human preference data. Given pairs of (chosen, rejected) completions for the same prompt, DPO increases the relative log-probability of the chosen completion while decreasing that of the rejected one -- all relative to a frozen reference model. The key insight is that the optimal policy under a KL-constrained reward maximization objective has a closed-form relationship with the reward, allowing DPO to skip the reward modeling step entirely.

## Key Equations

The DPO loss for a single preference pair $(y_w, y_l)$ given prompt $x$:

$$
L_{\text{DPO}}(\theta) = -\log \sigma\!\left(\beta \left[ \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right] \right)
$$

where $\sigma$ is the sigmoid function and $\beta$ is the temperature controlling how much the policy can deviate from the reference.

The implicit reward under DPO is:

$$
r(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)} + \text{const}
$$

Sequence log-probabilities are computed as:

$$
\log \pi(y | x) = \sum_{t=1}^{T} \log \pi(y_t | x, y_{<t})
$$

## Pseudocode

```
algorithm DPO:
    initialize policy model pi_theta
    freeze reference model pi_ref = copy(pi_theta)
    set temperature beta

    for each batch of (prompt, chosen, rejected) do
        # Concatenate prompt with each completion
        chosen_ids = concat(prompt, chosen)
        rejected_ids = concat(prompt, rejected)

        # Sequence log-probs under policy and reference
        log_pi_chosen = sum_token_logprobs(pi_theta, chosen_ids)
        log_pi_rejected = sum_token_logprobs(pi_theta, rejected_ids)
        log_ref_chosen = sum_token_logprobs(pi_ref, chosen_ids)
        log_ref_rejected = sum_token_logprobs(pi_ref, rejected_ids)

        # DPO loss
        log_ratio_w = log_pi_chosen - log_ref_chosen
        log_ratio_l = log_pi_rejected - log_ref_rejected
        loss = -mean(log_sigmoid(beta * (log_ratio_w - log_ratio_l)))

        update theta with gradient clipping
```

## Quick Start

```python
import torch
import copy
from rlox.algorithms.dpo import DPO

model = MyLanguageModel()
ref_model = copy.deepcopy(model)
ref_model.eval()

trainer = DPO(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    learning_rate=1e-4,
)

# Each training step takes (prompt, chosen, rejected) token tensors
metrics = trainer.train_step(
    prompt=prompt_ids,     # (B, P)
    chosen=chosen_ids,     # (B, C)
    rejected=rejected_ids, # (B, R)
)
print(f"Loss: {metrics['loss']:.4f}")
print(f"Chosen reward: {metrics['chosen_reward']:.3f}")
print(f"Rejected reward: {metrics['rejected_reward']:.3f}")
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | `0.1` | Temperature for the DPO loss (lower = more aggressive optimization) |
| `learning_rate` | `1e-4` | Adam learning rate |
| `max_grad_norm` | `1.0` | Gradient clipping norm |

## When to Use

- **Use DPO when:** you have pairwise human preference data (chosen vs. rejected completions) and want to align an LLM without training a separate reward model.
- **Prefer DPO over RLHF/PPO when:** you want a simpler pipeline (no reward model, no value function, no RL loop), or your preference data is readily available.
- **Do not use DPO when:** you need online exploration or reward-guided generation (prefer [GRPO](grpo.md)), or your preference data is noisy and would benefit from a learned reward model.

## References

- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*. *arXiv:2305.18290*.
