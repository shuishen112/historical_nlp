"""Policy-optimization objectives (Section 3, "Policy Optimization Frameworks").

  * PPO clipped surrogate:
        L = -min( r*A, clip(r, 1-eps, 1+eps)*A ),   r = exp(logp_new - logp_old)
  * GRPO group-relative advantage:
        A_i = (R_i - mean(R)) / (std(R) + eps)
"""

from __future__ import annotations

import torch


def ppo_clipped_loss(
    logprob_new: torch.Tensor,
    logprob_old: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
) -> torch.Tensor:
    """Negative clipped surrogate (a loss to minimize).

    All tensor args broadcast together. ``advantage`` may be a python float.
    """
    if not torch.is_tensor(advantage):
        advantage = torch.as_tensor(
            float(advantage), dtype=logprob_new.dtype, device=logprob_new.device
        )
    ratio = torch.exp(logprob_new - logprob_old)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.minimum(unclipped, clipped)
    if kl_coef > 0:
        approx_kl = (logprob_new - logprob_old) ** 2
        loss = loss + kl_coef * approx_kl
    return loss.mean()


def grpo_advantages(rewards: torch.Tensor, std_eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantages ``(R - mean) / (std + eps)`` (population std)."""
    if not torch.is_tensor(rewards):
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
    rewards = rewards.float()
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + std_eps)
