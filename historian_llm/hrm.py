"""Historian Reward Model (R_expert) — training + frozen online scoring.

The paper's HRM ``R_phi(x, y)`` is a frozen scorer trained on expert preference
pairs ``D_expert = {(x, y_w, y_l)}`` via the Bradley-Terry pairwise ranking loss.
The full system feeds a multimodal (image, question, answer) encoding into the HRM.

For a dependency-light, trainable, and testable realization we score a candidate
answer against the gold reference using a fixed vector of orthographic / factuality
proxy features (:func:`answer_features`). This keeps the HRM runnable inside the RL
loop on CPU and is straightforward to swap for real multimodal encodings later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch

from .rewards import (
    HistorianRewardModel,
    auto_reward,
    char_error_rate,
    exact_match,
    normalize_text,
    pairwise_ranking_loss,
    token_f1,
)

FEATURE_DIM = 6


def _char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _prefix_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    match = 0
    for i in range(n):
        if a[i] == b[i]:
            match += 1
        else:
            break
    return match / max(len(a), len(b))


def answer_features(pred: str, gold: str) -> List[float]:
    """Fixed-length (``FEATURE_DIM``) proxy features for expert judgment.

    Captures signals a historian implicitly uses: orthographic closeness to the
    documented reference, lexical overlap, exact correctness, length plausibility.
    """
    p = normalize_text(pred)
    g = normalize_text(gold)
    len_ratio = min(len(p), len(g)) / max(1, max(len(p), len(g)))
    return [
        auto_reward(pred, gold),        # 1 - CER (orthographic closeness)
        token_f1(pred, gold),           # lexical overlap
        exact_match(pred, gold),        # exact correctness
        char_error_rate(pred, gold),    # raw error rate
        _char_jaccard(p, g),            # character-set overlap
        _prefix_overlap(p, g),          # leading-substring agreement
    ]


def features_tensor(pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
    """Stack ``answer_features`` for a batch of (pred, gold) pairs -> (B, FEATURE_DIM)."""
    return torch.tensor([answer_features(p, g) for p, g in pairs], dtype=torch.float32)


def new_hrm(hidden_dim: int = 32) -> HistorianRewardModel:
    return HistorianRewardModel(feature_dim=FEATURE_DIM, hidden_dim=hidden_dim)


def train_hrm(
    preferences: Sequence[dict],
    epochs: int = 50,
    lr: float = 0.05,
    hidden_dim: int = 32,
    seed: int = 0,
) -> Tuple[HistorianRewardModel, List[float]]:
    """Train an HRM on preference records.

    Each record is ``{"gold": str, "chosen": str, "rejected": str}``; the HRM learns
    ``R_phi(chosen) > R_phi(rejected)`` via the pairwise ranking loss.
    Returns the trained model and the per-epoch loss history.
    """
    torch.manual_seed(seed)
    win = features_tensor([(r["chosen"], r["gold"]) for r in preferences])
    los = features_tensor([(r["rejected"], r["gold"]) for r in preferences])

    model = new_hrm(hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = pairwise_ranking_loss(model(win), model(los))
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return model, history


def save_hrm(model: HistorianRewardModel, path: Union[str, Path], hidden_dim: int = 32) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "feature_dim": FEATURE_DIM, "hidden_dim": hidden_dim},
        path,
    )


def load_hrm(path: Union[str, Path]) -> HistorianRewardModel:
    ckpt = torch.load(path, map_location="cpu")
    model = HistorianRewardModel(
        feature_dim=ckpt.get("feature_dim", FEATURE_DIM),
        hidden_dim=ckpt.get("hidden_dim", 32),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def hrm_score(model: HistorianRewardModel, pred: str, gold: str) -> float:
    """Frozen online R_expert for a single (pred, gold)."""
    model.eval()
    with torch.no_grad():
        feats = features_tensor([(pred, gold)])
        return float(model(feats).item())


def load_preferences(path: Union[str, Path]) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of preference records in {path}")
    return data
