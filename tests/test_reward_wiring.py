"""Tests that the training scripts use the CER-based R_auto and HRM R_expert."""

import importlib

from historian_llm.rewards import auto_reward

rl = importlib.import_module("rl_lora_dutch_vqa")
verl_reward = importlib.import_module("verl_reward_dutch_vqa")


def test_rl_compute_reward_cer_mode_matches_auto_reward():
    cfg = rl.RewardConfig(mode="cer", alpha=1.0, beta=1.0, hrm_scorer=None)
    pred, gold = "gouverpement", "gouvernement"
    assert abs(rl.compute_reward(pred, gold, cfg) - auto_reward(pred, gold)) < 1e-9


def test_rl_compute_reward_perfect_and_alpha():
    cfg = rl.RewardConfig(mode="cer", alpha=2.0, beta=1.0, hrm_scorer=None)
    assert abs(rl.compute_reward("batavia", "batavia", cfg) - 2.0) < 1e-9


def test_rl_compute_reward_adds_hrm_expert_term():
    cfg = rl.RewardConfig(mode="cer", alpha=1.0, beta=0.5, hrm_scorer=lambda p, g: 4.0)
    # R = alpha*R_auto + beta*R_expert = 1.0 + 0.5*4.0 = 3.0 for a perfect match.
    assert abs(rl.compute_reward("batavia", "batavia", cfg) - 3.0) < 1e-9


def test_rl_blend_mode_uses_cer_component():
    cfg = rl.RewardConfig(mode="blend", f1_weight=0.5, alpha=1.0, beta=1.0, hrm_scorer=None)
    # Perfect match => CER-reward=1, F1=1 => blend=1.
    assert abs(rl.compute_reward("batavia", "batavia", cfg) - 1.0) < 1e-9


def test_verl_default_is_cer():
    pred, gold = "gouverpement", "gouvernement"
    score = verl_reward.compute_score("dutch_vqa", pred, gold, {})
    assert abs(score - auto_reward(pred, gold)) < 1e-9


def test_verl_perfect_match():
    assert abs(verl_reward.compute_score("dutch_vqa", "batavia", "batavia", {}) - 1.0) < 1e-9
