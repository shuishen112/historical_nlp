# Requirements: Implementing HistorianLLM (`analysis.md`)

Step-by-step plan to build a small, unit-tested reference package that realizes every
equation in `analysis.md`, plus the **minimal tests** that prove each piece works.

- **Environment:** conda env `mttl_upgrade` (torch 2.8.0, pytest 9.1.1).
- **Constraint:** everything must run on **CPU**, with **no model downloads** and in
  **seconds** (tiny random tensors / toy strings only). Heavy training stays in the
  existing scripts.
- **Package layout:**

```
historian_llm/
  __init__.py
  alignment.py      # Eq.1  visual cross-attention
  lora.py           # Eq.2  low-rank adapter
  rewards.py        # CER, R_auto, HRM (R_expert), hybrid R + KL
  rl.py             # PPO clipped surrogate, GRPO group advantage
tests/
  test_alignment.py
  test_lora.py
  test_rewards.py
  test_rl.py
```

Run tests with:

```bash
conda run -n mttl_upgrade python -m pytest tests/ -q
```

---

## Step 1 — Visual alignment (Eq. 1) → `historian_llm/alignment.py`

**Implement** `visual_cross_attention(X_txt, X_img, W_q, W_k, W_v)` computing
`Softmax((Wq·Xtxt)(Wk·Ximg)^T / sqrt(d)) (Wv·Ximg)`, and a thin
`nn.Module` `VisualAligner(d_txt, d_img, d_model)` wrapping learnable `W_q,W_k,W_v`.

**Requirements**
- Batched input `(B, N_txt, d_txt)` and `(B, N_img, d_img)`; output
  `(B, N_txt, d_model)`.
- Attention weights are a valid distribution (rows sum to 1, non-negative).
- Scaling by `1/sqrt(d_model)`.

**Minimal tests** (`test_alignment.py`)
1. Output shape is `(B, N_txt, d_model)`.
2. Internal attention matrix rows sum to 1 and are ≥ 0.
3. If all image tokens are identical, output equals `W_v` applied to that token
   (attention can't do anything but average identical values).
4. `VisualAligner` forward runs and is differentiable (grad flows to `W_q`).

---

## Step 2 — LoRA adapter (Eq. 2) → `historian_llm/lora.py`

**Implement** `LoRALinear(in_features, out_features, r, alpha, dropout)` with a frozen
base weight `W0` and trainable `A, B`, forward `h = W0 x + (alpha/r) B A x`.

**Requirements**
- `B` initialized to zeros → at init, output == frozen base output.
- Base weight `requires_grad == False`; `A`, `B` trainable.
- `scaling == alpha / r`.
- Helper `lora_delta(x, A, B, scaling)` for the pure `ΔW x` term.

**Minimal tests** (`test_lora.py`)
1. At init (`B=0`) the layer reproduces the frozen linear base output.
2. After perturbing `B`, output changes (adapter is active).
3. Base parameter has `requires_grad=False`; `A`/`B` have `requires_grad=True`.
4. Delta path has the correct rank (rank of `ΔW = B A` ≤ `r`).

---

## Step 3 — Rewards → `historian_llm/rewards.py`

Implement the whole reward stack from §3 of `analysis.md`.

### 3a. Character Error Rate + `R_auto`
- `char_error_rate(pred, gold)` = Levenshtein edit distance (chars) / `len(gold)`
  (normalized text; empty-gold handled).
- `auto_reward(pred, gold) = 1 - CER`, clamped to `[0, 1]`.

### 3b. Historian Reward Model (`R_expert`)
- `HistorianRewardModel(nn.Module)`: small scorer `R_φ(features) -> scalar`
  (feature vector per (x,y); e.g. an MLP). Kept tiny for tests.
- `pairwise_ranking_loss(score_w, score_l) = -log σ(score_w - score_l)`
  (Bradley–Terry).
- `expert_reward(model, features)` returns `R_φ` as a detached scalar (frozen use).

### 3c. Hybrid reward + KL
- `hybrid_reward(r_auto, r_expert, kl, alpha, beta, lam)
   = alpha*r_auto + beta*r_expert - lam*kl`.

**Minimal tests** (`test_rewards.py`)
1. `char_error_rate("gouvernement","gouvernement") == 0`; identical strings → 0.
2. One-char substitution (`gouverpement`) → CER ≈ 1/len; `auto_reward` in `(0,1)`.
3. `auto_reward` for a perfect match == 1.0; for empty-vs-nonempty == 0.0.
4. `pairwise_ranking_loss` decreases when `score_w - score_l` grows.
5. **HRM learns a toy preference:** train `HistorianRewardModel` a few steps on a
   toy `D_expert` where `y_w` features are separable from `y_l`; assert final
   `R_φ(y_w) > R_φ(y_l)` and loss went down.
6. `hybrid_reward` matches `alpha*a + beta*b - lam*kl`; KL penalty lowers reward.

---

## Step 4 — Policy optimization → `historian_llm/rl.py`

### 4a. PPO clipped surrogate
- `ppo_clipped_loss(logprob_new, logprob_old, advantage, clip_eps, kl_coef=0)`
  returns `-min(r·A, clip(r,1-ε,1+ε)·A) (+ kl_coef·kl)`, with
  `r = exp(logprob_new - logprob_old)`.

### 4b. GRPO group advantage
- `grpo_advantages(rewards, std_eps)` = `(r - mean) / (std + eps)` over a group.

**Minimal tests** (`test_rl.py`)
1. When `logprob_new == logprob_old` (`r=1`), PPO loss == `-advantage`.
2. Positive advantage with large positive ratio → clipping caps the loss at
   `-(1+ε)·A` (loss not smaller than the clipped bound).
3. `grpo_advantages` outputs mean ≈ 0 and (population) std ≈ 1 for a spread group.
4. In a group, the max-reward sample gets the max (positive) advantage and the
   min-reward sample gets the min (negative) advantage.
5. A degenerate group (all equal rewards) yields all-≈0 advantages (no NaNs).

---

## Step 5 — Integration & acceptance

1. `historian_llm/__init__.py` re-exports the public API.
2. All tests pass: `conda run -n mttl_upgrade python -m pytest tests/ -q`.
3. Sanity import: `conda run -n mttl_upgrade python -c "import historian_llm"`.

**Definition of done:** every equation in `analysis.md` has a corresponding
implemented function/module and at least one passing minimal test, all green in
`mttl_upgrade`.
