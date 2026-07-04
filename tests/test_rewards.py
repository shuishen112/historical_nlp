import torch

from historian_llm.rewards import (
    HistorianRewardModel,
    auto_reward,
    char_error_rate,
    expert_reward,
    hybrid_reward,
    pairwise_ranking_loss,
)


def test_cer_identical_is_zero():
    assert char_error_rate("gouvernement", "gouvernement") == 0.0
    assert char_error_rate("  Batavia ", "batavia") == 0.0


def test_cer_single_substitution():
    # "gouverpement" vs "gouvernement": 1 substitution over 12 chars.
    gold = "gouvernement"
    pred = "gouverpement"
    cer = char_error_rate(pred, gold)
    assert abs(cer - 1.0 / len(gold)) < 1e-9
    r = auto_reward(pred, gold)
    assert 0.0 < r < 1.0


def test_auto_reward_bounds():
    assert auto_reward("batavia", "batavia") == 1.0
    assert auto_reward("", "batavia") == 0.0
    # Wildly wrong long prediction stays clamped at 0.
    assert auto_reward("x" * 100, "ab") == 0.0


def test_pairwise_loss_monotonic():
    small = pairwise_ranking_loss(torch.tensor([0.1]), torch.tensor([0.0]))
    large = pairwise_ranking_loss(torch.tensor([5.0]), torch.tensor([0.0]))
    assert large < small


def test_hrm_learns_toy_preference():
    torch.manual_seed(0)
    feat_dim = 4
    model = HistorianRewardModel(feature_dim=feat_dim, hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    # Winners cluster around +1, losers around -1 => linearly separable.
    n = 64
    win = torch.ones(n, feat_dim) + 0.05 * torch.randn(n, feat_dim)
    los = -torch.ones(n, feat_dim) + 0.05 * torch.randn(n, feat_dim)

    first_loss = None
    for _ in range(200):
        opt.zero_grad()
        s_w = model(win)
        s_l = model(los)
        loss = pairwise_ranking_loss(s_w, s_l)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
    assert loss.item() < first_loss
    # Frozen online use ranks winner above loser.
    r_w = expert_reward(model, torch.ones(1, feat_dim))
    r_l = expert_reward(model, -torch.ones(1, feat_dim))
    assert r_w.item() > r_l.item()


def test_hybrid_reward_formula_and_kl_penalty():
    val = hybrid_reward(r_auto=0.8, r_expert=0.5, kl=2.0, alpha=1.0, beta=2.0, lam=0.1)
    assert abs(val - (1.0 * 0.8 + 2.0 * 0.5 - 0.1 * 2.0)) < 1e-9
    with_kl = hybrid_reward(0.8, 0.5, kl=3.0, lam=0.5)
    without_kl = hybrid_reward(0.8, 0.5, kl=0.0, lam=0.5)
    assert with_kl < without_kl
