import torch

from historian_llm.rl import grpo_advantages, ppo_clipped_loss


def test_ppo_loss_equals_neg_advantage_at_ratio_one():
    lp = torch.zeros(1)
    loss = ppo_clipped_loss(lp, lp.clone(), advantage=0.7, clip_eps=0.2)
    assert abs(loss.item() - (-0.7)) < 1e-6


def test_ppo_clipping_caps_positive_advantage():
    # Large positive ratio with positive advantage => clipped to (1+eps)*A.
    logp_new = torch.tensor([2.0])
    logp_old = torch.tensor([0.0])
    adv = 1.0
    clip_eps = 0.2
    loss = ppo_clipped_loss(logp_new, logp_old, advantage=adv, clip_eps=clip_eps)
    expected = -(1.0 + clip_eps) * adv
    assert abs(loss.item() - expected) < 1e-6


def test_grpo_advantages_standardized():
    rewards = torch.tensor([0.0, 0.5, 1.0, 0.25, 0.75])
    adv = grpo_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std(unbiased=False).item() - 1.0) < 1e-3


def test_grpo_orders_extremes():
    rewards = torch.tensor([0.1, 0.9, 0.5, 0.3])
    adv = grpo_advantages(rewards)
    assert torch.argmax(adv) == torch.argmax(rewards)
    assert torch.argmin(adv) == torch.argmin(rewards)


def test_grpo_degenerate_group_no_nan():
    rewards = torch.tensor([0.5, 0.5, 0.5])
    adv = grpo_advantages(rewards)
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-3)
