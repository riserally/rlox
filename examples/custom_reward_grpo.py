"""GRPO with a custom reward function for LLM post-training.

This example uses a tiny toy model to demonstrate the API.
Replace with your own LLM and reward function for real use.
"""

import torch
import torch.nn as nn
from rlox.algorithms import GRPO


# --- Toy LLM (replace with your model) ---
class TinyLM(nn.Module):
    def __init__(self, vocab_size=100, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        return self.head(self.embed(x))

    def generate(self, prompts, max_new_tokens=8):
        """Simple greedy generation."""
        tokens = prompts.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(tokens)[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


# --- Custom reward function ---
def math_reward(completions, prompts):
    """Reward completions that contain the 'answer' token (id=42)."""
    rewards = []
    for completion in completions:
        if 42 in completion.tolist():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# --- Train ---
model = TinyLM()
ref_model = TinyLM()  # frozen reference
ref_model.load_state_dict(model.state_dict())
for p in ref_model.parameters():
    p.requires_grad = False

grpo = GRPO(
    model=model,
    ref_model=ref_model,
    reward_fn=math_reward,
    group_size=4,
    kl_coef=0.1,
    max_new_tokens=8,
)

prompts = torch.randint(0, 100, (4, 5))  # 4 prompts, length 5
metrics = grpo.train(prompts, n_epochs=3)
print(f"Loss: {metrics['loss']:.4f}, Mean reward: {metrics['mean_reward']:.2f}")
