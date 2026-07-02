"""Visual cross-attention alignment (Eq. 1 of the HistorianLLM methodology).

    F_aligned = Softmax( (Wq Xtxt)(Wk Ximg)^T / sqrt(d) ) (Wv Ximg)

Maps ViT visual tokens into the text decoder's input space so that text queries can
pull the visual evidence they need from a native-resolution document image.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def visual_cross_attention(
    x_txt: torch.Tensor,
    x_img: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    return_attn: bool = False,
):
    """Scaled dot-product cross-attention from text queries onto visual tokens.

    Args:
        x_txt: text query embeddings, shape ``(B, N_txt, d_txt)``.
        x_img: visual tokens, shape ``(B, N_img, d_img)``.
        w_q:   query projection, shape ``(d_txt, d_model)``.
        w_k:   key projection,   shape ``(d_img, d_model)``.
        w_v:   value projection, shape ``(d_img, d_model)``.
        return_attn: if True, also return the attention weight matrix.

    Returns:
        ``F_aligned`` of shape ``(B, N_txt, d_model)`` (and attention weights
        ``(B, N_txt, N_img)`` if ``return_attn``).
    """
    q = x_txt @ w_q  # (B, N_txt, d_model)
    k = x_img @ w_k  # (B, N_img, d_model)
    v = x_img @ w_v  # (B, N_img, d_model)

    d_model = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_model)  # (B, N_txt, N_img)
    attn = F.softmax(scores, dim=-1)
    aligned = attn @ v  # (B, N_txt, d_model)

    if return_attn:
        return aligned, attn
    return aligned


class VisualAligner(nn.Module):
    """Learnable wrapper around :func:`visual_cross_attention`."""

    def __init__(self, d_txt: int, d_img: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Parameter(torch.empty(d_txt, d_model))
        self.w_k = nn.Parameter(torch.empty(d_img, d_model))
        self.w_v = nn.Parameter(torch.empty(d_img, d_model))
        for w in (self.w_q, self.w_k, self.w_v):
            nn.init.xavier_uniform_(w)

    def forward(
        self, x_txt: torch.Tensor, x_img: torch.Tensor, return_attn: bool = False
    ):
        return visual_cross_attention(
            x_txt, x_img, self.w_q, self.w_k, self.w_v, return_attn=return_attn
        )
