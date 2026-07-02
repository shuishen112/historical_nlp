"""HistorianLLM: a testable reference implementation of the methodology in
``paper.latex`` / ``analysis.md``.

Public API groups the four building blocks described in ``analysis.md``:
visual alignment, LoRA adaptation, the RLHF reward stack, and policy optimization.
"""

from .alignment import VisualAligner, visual_cross_attention
from .hrm import (
    FEATURE_DIM,
    answer_features,
    features_tensor,
    hrm_score,
    load_hrm,
    load_preferences,
    new_hrm,
    save_hrm,
    train_hrm,
)
from .lora import LoRALinear, lora_delta
from .rewards import (
    HistorianRewardModel,
    auto_reward,
    char_error_rate,
    exact_match,
    expert_reward,
    hybrid_reward,
    levenshtein,
    normalize_text,
    pairwise_ranking_loss,
    token_f1,
)
from .rl import grpo_advantages, ppo_clipped_loss

__all__ = [
    "VisualAligner",
    "visual_cross_attention",
    "LoRALinear",
    "lora_delta",
    "HistorianRewardModel",
    "auto_reward",
    "char_error_rate",
    "exact_match",
    "expert_reward",
    "hybrid_reward",
    "levenshtein",
    "normalize_text",
    "pairwise_ranking_loss",
    "token_f1",
    "grpo_advantages",
    "ppo_clipped_loss",
    "FEATURE_DIM",
    "answer_features",
    "features_tensor",
    "hrm_score",
    "load_hrm",
    "load_preferences",
    "new_hrm",
    "save_hrm",
    "train_hrm",
]
