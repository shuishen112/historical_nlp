"""Reward stack for Stage-2 RLHF (Section 3 of the HistorianLLM methodology).

Components:
  * ``char_error_rate`` / ``auto_reward``  ->  R_auto = 1 - CER(y, y*)
  * ``HistorianRewardModel`` + ``pairwise_ranking_loss``  ->  R_expert = R_phi(x, y)
  * ``hybrid_reward``  ->  R = alpha*R_auto + beta*R_expert - lambda*KL
"""

from __future__ import annotations

import re
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 3a. Automated VQA grounding reward: R_auto = 1 - CER
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance (insert/delete/substitute cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_error_rate(pred: str, gold: str) -> float:
    """Normalized Character Error Rate = edit_distance(pred, gold) / len(gold)."""
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    if not gold_n:
        return 0.0 if not pred_n else 1.0
    return levenshtein(pred_n, gold_n) / len(gold_n)


def auto_reward(pred: str, gold: str) -> float:
    """R_auto = clamp(1 - CER, 0, 1)."""
    return max(0.0, min(1.0, 1.0 - char_error_rate(pred, gold)))


def exact_match(pred: str, gold: str) -> float:
    """1.0 iff normalized strings are identical."""
    return float(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 over whitespace tokens of the normalized strings."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict = {}
    gold_counts: dict = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = sum(min(c, gold_counts.get(t, 0)) for t, c in pred_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# 3b. Historian Expert Reward: frozen HRM R_phi trained via pairwise ranking
# --------------------------------------------------------------------------- #
class HistorianRewardModel(nn.Module):
    """Small MLP that scores a (x, y) feature vector; proxy for expert judgment.

    In the full system the feature vector is a multimodal encoding of the image,
    question and candidate answer. Here it is left generic so it can be unit-tested
    with tiny synthetic features and later wired to real encodings.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: ``(B, feature_dim)`` -> scores ``(B,)``."""
        return self.net(features).squeeze(-1)


def pairwise_ranking_loss(
    score_w: torch.Tensor, score_l: torch.Tensor
) -> torch.Tensor:
    """Bradley-Terry loss: ``-log sigmoid(score_w - score_l)`` (mean-reduced)."""
    return -F.logsigmoid(score_w - score_l).mean()


def expert_reward(model: HistorianRewardModel, features: torch.Tensor) -> torch.Tensor:
    """Frozen online use of the HRM: returns detached R_phi(x, y)."""
    model.eval()
    with torch.no_grad():
        return model(features)


# --------------------------------------------------------------------------- #
# 3c. Hybrid reward with KL regularization
# --------------------------------------------------------------------------- #
def hybrid_reward(
    r_auto: float,
    r_expert: float,
    kl: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    lam: float = 0.0,
) -> float:
    """R(x, y) = alpha*R_auto + beta*R_expert - lambda*KL(pi_theta || pi_ref)."""
    return alpha * r_auto + beta * r_expert - lam * kl
