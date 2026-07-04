import torch

from historian_llm.lora import LoRALinear


def test_init_reproduces_base():
    layer = LoRALinear(8, 4, r=2, alpha=8.0)
    x = torch.randn(3, 8)
    out = layer(x)
    base_out = layer.base(x)
    assert torch.allclose(out, base_out, atol=1e-6)


def test_adapter_activates_when_b_nonzero():
    layer = LoRALinear(8, 4, r=2, alpha=8.0)
    x = torch.randn(3, 8)
    with torch.no_grad():
        layer.lora_b.add_(torch.randn_like(layer.lora_b))
    out = layer(x)
    base_out = layer.base(x)
    assert not torch.allclose(out, base_out, atol=1e-4)


def test_requires_grad_flags():
    layer = LoRALinear(8, 4, r=2)
    assert layer.base.weight.requires_grad is False
    assert layer.lora_a.requires_grad is True
    assert layer.lora_b.requires_grad is True


def test_scaling_value():
    layer = LoRALinear(8, 4, r=4, alpha=16.0)
    assert layer.scaling == 16.0 / 4


def test_delta_weight_rank_bounded_by_r():
    layer = LoRALinear(16, 12, r=3, alpha=6.0)
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn_like(layer.lora_b))
    dw = layer.delta_weight()
    assert dw.shape == (12, 16)
    rank = torch.linalg.matrix_rank(dw)
    assert int(rank) <= 3
