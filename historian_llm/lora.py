"""Low-Rank Adaptation (Eq. 2 of the HistorianLLM methodology).

    h = W0 x + dW x = W0 x + (alpha / r) B A x

A frozen base weight ``W0`` is adapted by a trainable low-rank product ``B A`` with
rank ``r << min(d, k)``. ``B`` starts at zero so training begins exactly at the
backbone (``dW x = 0``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def lora_delta(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float
) -> torch.Tensor:
    """Pure adapter path ``(alpha/r) B A x``.

    Args:
        x: input, shape ``(..., in_features)``.
        a: down-projection, shape ``(r, in_features)``.
        b: up-projection,   shape ``(out_features, r)``.
        scaling: ``alpha / r``.
    """
    return scaling * (x @ a.t() @ b.t())


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight and a trainable LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        base = nn.Linear(in_features, out_features, bias=bias)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        self.base = base

        # A ~ small random, B = 0  =>  dW = B A = 0 at init.
        self.lora_a = nn.Parameter(torch.empty(r, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = lora_delta(self.dropout(x), self.lora_a, self.lora_b, self.scaling)
        return base_out + delta

    def delta_weight(self) -> torch.Tensor:
        """Materialized ``dW = (alpha/r) B A`` of shape ``(out_features, in_features)``."""
        return self.scaling * (self.lora_b @ self.lora_a)
