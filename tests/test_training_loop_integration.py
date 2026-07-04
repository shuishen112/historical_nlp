"""End-to-end training-loop smoke test.

Exercises the *real* RL functions from ``rl_lora_dutch_vqa.py`` (reward computation,
REINFORCE loss, PPO clipped surrogate, GRPO advantages) wired to a tiny dummy policy,
proving the CER-based R_auto + HRM R_expert reward flows through to gradients and an
optimizer step. This runs fully in `mttl_upgrade` without the 2B VL model, so it is
independent of the transformers/qwen3_vl version needed for the real backbone.
"""

import importlib

import torch
import torch.nn as nn

rl = importlib.import_module("rl_lora_dutch_vqa")
from historian_llm.hrm import hrm_score, train_hrm  # noqa: E402


class DummyPolicy(nn.Module):
    """Minimal causal-LM stand-in exposing `.logits` and `.loss`."""

    def __init__(self, vocab_size=32, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        logits = self.head(h)

        class Out:
            pass

        out = Out()
        out.logits = logits
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            out.loss = nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100
            )
        return out


def _make_inputs(prompt_len=3, gen_len=4, vocab_size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, vocab_size, (1, prompt_len), generator=g)
    gen = torch.randint(0, vocab_size, (1, gen_len), generator=g)
    sequences = torch.cat([prompt, gen], dim=1)
    return {"input_ids": prompt}, sequences, prompt_len


def test_reinforce_step_produces_gradients_with_cer_hrm_reward():
    torch.manual_seed(0)
    # Train a real (tiny) HRM so R_expert is meaningful.
    prefs = [
        {"gold": "batavia", "chosen": "batavia", "rejected": "btvxia"},
        {"gold": "gouvernement", "chosen": "gouvernement", "rejected": "gouverpement"},
    ]
    hrm_model, _ = train_hrm(prefs, epochs=80, lr=0.05, seed=0)
    cfg = rl.RewardConfig(
        mode="cer",
        alpha=1.0,
        beta=0.5,
        hrm_scorer=lambda p, g: hrm_score(hrm_model, p, g),
    )

    # Reward combines CER R_auto and HRM R_expert.
    reward = rl.compute_reward("batavia", "batavia", cfg)
    baseline = rl.compute_reward("btvxia", "batavia", cfg)
    advantage = reward - baseline
    assert reward > baseline  # correct answer preferred

    model = DummyPolicy()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model_inputs, sequences, prompt_len = _make_inputs()

    base_loss = rl.rl_loss_from_sample(model, model_inputs, sequences, prompt_len)
    loss = base_loss * advantage
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()


def test_ppo_step_runs_and_updates():
    torch.manual_seed(1)
    model = DummyPolicy()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model_inputs, sequences, prompt_len = _make_inputs(seed=2)

    with torch.no_grad():
        old_lp = rl.compute_sequence_logprob(model, model_inputs, sequences, prompt_len)
    loss, new_lp = rl.ppo_loss_from_sample(
        model, model_inputs, sequences, prompt_len,
        old_logprob=old_lp, advantage=0.8, clip_epsilon=0.2, kl_coef=0.1,
    )
    before = model.head.weight.detach().clone()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.allclose(before, model.head.weight)


def test_grpo_group_advantages_from_real_rewards():
    cfg = rl.RewardConfig(mode="cer", alpha=1.0, beta=0.0, hrm_scorer=None)
    gold = "gouvernement"
    candidates = ["gouvernement", "gouverpement", "xxxx", "gouvernment"]
    rewards = torch.tensor([rl.compute_reward(c, gold, cfg) for c in candidates])
    from historian_llm.rl import grpo_advantages

    adv = grpo_advantages(rewards)
    # Best candidate (exact match) gets the largest advantage.
    assert torch.argmax(adv).item() == 0
    assert torch.isfinite(adv).all()
