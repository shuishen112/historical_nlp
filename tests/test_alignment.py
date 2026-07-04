import torch

from historian_llm.alignment import VisualAligner, visual_cross_attention


def _weights(d_txt, d_img, d_model, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_q = torch.randn(d_txt, d_model, generator=g)
    w_k = torch.randn(d_img, d_model, generator=g)
    w_v = torch.randn(d_img, d_model, generator=g)
    return w_q, w_k, w_v


def test_output_shape():
    B, N_txt, N_img, d_txt, d_img, d_model = 2, 4, 5, 6, 7, 8
    x_txt = torch.randn(B, N_txt, d_txt)
    x_img = torch.randn(B, N_img, d_img)
    w_q, w_k, w_v = _weights(d_txt, d_img, d_model)
    out = visual_cross_attention(x_txt, x_img, w_q, w_k, w_v)
    assert out.shape == (B, N_txt, d_model)


def test_attention_is_distribution():
    x_txt = torch.randn(2, 3, 6)
    x_img = torch.randn(2, 5, 7)
    w_q, w_k, w_v = _weights(6, 7, 8)
    _, attn = visual_cross_attention(x_txt, x_img, w_q, w_k, w_v, return_attn=True)
    assert torch.all(attn >= 0)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_identical_image_tokens_average_to_value():
    # If all image tokens are identical, output = W_v applied to that token.
    B, N_txt, N_img, d_img, d_model = 1, 3, 4, 7, 8
    single = torch.randn(1, 1, d_img)
    x_img = single.repeat(B, N_img, 1)
    x_txt = torch.randn(B, N_txt, 6)
    w_q, w_k, w_v = _weights(6, d_img, d_model)
    out = visual_cross_attention(x_txt, x_img, w_q, w_k, w_v)
    expected = (single @ w_v).repeat(1, N_txt, 1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_module_is_differentiable():
    aligner = VisualAligner(d_txt=6, d_img=7, d_model=8)
    x_txt = torch.randn(2, 3, 6)
    x_img = torch.randn(2, 5, 7)
    out = aligner(x_txt, x_img)
    out.sum().backward()
    assert aligner.w_q.grad is not None
    assert torch.isfinite(aligner.w_q.grad).all()
