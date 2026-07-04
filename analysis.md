# Analysis: HistorianLLM Architecture

This document restates the methodology from `paper.latex`
(`\section{Methodology: HistorianLLM Architecture}`) and explains the idea behind
each component so it can be turned into working code. See `requirements.md` for the
step-by-step implementation plan and tests.

---

## 1. The core idea

Generic frontier vision-language models (VLMs) collapse on **archaic colonial
scripts**: degraded 18th–19th-century newspaper scans, obsolete orthography, and
dense multi-column layouts. **HistorianLLM** is a VLM specialized for these
documents. Two problems dominate:

1. **Domain mismatch** — the backbone has never seen colonial Dutch spelling or
   faded archival typography.
2. **Hallucination / exposure bias** — when a model misreads one faded character,
   the error cascades and it invents *plausible-looking but fabricated* archaic
   words (e.g. reading `gouvernement` and drifting into `gouverpement`).

The design attacks both: a resolution-preserving visual front-end plus a
two-stage adaptation pipeline (LoRA fine-tuning, then RL grounded in a Character
Error Rate signal and an expert reward model).

---

## 2. Base architecture and visual alignment

- **Encoder/decoder.** A ViT visual encoder is paired with an autoregressive
  transformer text decoder.
- **Naive Dynamic Resolution.** Document images keep their native resolution
  instead of being downsampled to a fixed grid, so dense text is not distorted.
- **Cross-attention projection.** Visual tokens are mapped into the text input
  space with a single scaled dot-product attention block:

  \[
  F_{\text{aligned}}
  = \operatorname{Softmax}\!\left(
      \frac{(W_q X_{\text{txt}})(W_k X_{\text{img}})^{\top}}{\sqrt{d}}
    \right)(W_v X_{\text{img}})
  \]

  - `X_txt` = text query embeddings, `X_img` = visual tokens.
  - `W_q, W_k, W_v` = learnable projections.
  - The text queries *attend over* the visual tokens; the output has one aligned
    vector per text query, living in the decoder's input space.

**Why it matters.** Preserving resolution + letting text queries pull the visual
evidence they need is what keeps the model anchored to the raw pixels, which the
RL stage later exploits to punish ungrounded generations.

---

## 3. Domain adaptation: a two-stage training paradigm

### Stage 1 — Parameter-Efficient Fine-Tuning via LoRA

Freeze the backbone; inject low-rank adapters into the attention projections
(`q/k/v/o`). For a frozen weight `W0 ∈ R^{d×k}`:

\[
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x
\]

- `A ∈ R^{r×k}`, `B ∈ R^{d×r}`, rank `r ≪ min(d, k)`.
- `α` is a fixed scaling constant; the effective scale is `α/r`.
- `B` is initialized to zero so training starts exactly at the backbone
  (`ΔW x = 0`).
- Trained with **cross-entropy** over curated VQA pairs.

**Why it matters.** Cheap adaptation to archaic vocabulary without destroying the
backbone's general vision-language ability.

### Stage 2 — Reinforcement Learning from Historian Feedback (RLHF)

SFT with teacher forcing suffers from **exposure bias**: it only ever sees
ground-truth prefixes, so at inference an early misread cascades. Stage 2 moves
from token-level imitation to **sequence-level optimization**.

**VQA as a Vision-Language MDP.** Context `x = (I, Q)` (image snippet + query);
the policy `π_θ` autoregressively emits `y = (y_1,…,y_T)`, each token
`y_t ~ π_θ(y_t | I, Q, y_{<t})`. Optimizing over whole sequences penalizes
cumulative drift and forces visual grounding.

#### Expert-informed reward function

\[
R(x, y) = \alpha\,R_{\text{auto}}(x, y) + \beta\,R_{\text{expert}}(x, y)
          - \lambda\,\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})
\]

- **`R_auto` — Automated VQA grounding reward:**
  \[
  R_{\text{auto}}(x, y) = 1 - \operatorname{CER}(y, y^{*})
  \]
  where CER is the normalized Character Error Rate (Levenshtein edit distance over
  characters, divided by reference length). Catches crude character misalignment;
  blind to historical nuance.

- **`R_expert` — Historian Expert Reward:** a frozen **Historian Reward Model
  (HRM)** `R_φ`. Historians annotate preference pairs
  `D_expert = {(x, y_w, y_l)}` where `y_w` beats `y_l` on:
  - *Orthographic authenticity* — documented colonial Dutch spelling vs.
    invented pseudo-archaic tokens.
  - *Contextual factuality* — correct colonial entities/currencies (Batavia,
    rijksdaalders, …).

  The HRM is trained with the standard **pairwise (Bradley–Terry) ranking loss**
  \[
  \mathcal{L} = -\log \sigma\!\big(R_\phi(x, y_w) - R_\phi(x, y_l)\big)
  \]
  and then frozen to provide `R_expert(x, y) = R_φ(x, y)` online.

- **KL term** keeps `π_θ` close to the reference policy `π_ref` (anti-reward-hacking).

#### Policy optimization frameworks

Both maximize `E_{y~π_θ}[R(x, y)]`.

1. **PPO (actor–critic).** Clipped surrogate:
   \[
   L_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t\!\left[
     \min\!\big(r_t(\theta)\hat{A}_t,\;
       \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)
   \right],\quad
   r_t(\theta) = \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\text{old}}(y_t|x,y_{<t})}
   \]
   `Â_t` = Generalized Advantage Estimate from a value network `V_ψ`.

2. **GRPO (critic-free).** Sample a group of `G` outputs per prompt; normalize each
   reward against the group:
   \[
   A_i = \frac{R(x, y_i) - \frac{1}{G}\sum_{j=1}^{G} R(x, y_j)}{\sigma_R}
   \]
   Drops the memory-hungry value network. A rollout that drifts scores below the
   group mean → negative advantage → its faulty trajectory's log-likelihood is
   suppressed, training the policy to stay pixel-grounded.

**Why it matters.** GRPO gives the exposure-bias fix without a separate VLM-sized
critic — key under GPU memory constraints.

---

## 4. Relation to the existing codebase

| Methodology piece            | Where it lives today                              | Gap to close |
|------------------------------|---------------------------------------------------|--------------|
| Visual cross-attention (Eq.1)| implicit in Qwen3-VL backbone                     | no standalone, testable reference |
| LoRA Stage 1 (Eq.2)          | `peft_dutch_vqa_train_eval.py`                     | uses PEFT lib; no explicit math reference |
| `R_auto = 1 − CER`           | `rl_lora_dutch_vqa.py`, `verl_reward_dutch_vqa.py`| **uses EM/token-F1, not CER** |
| `R_expert` (HRM)             | —                                                 | **not implemented** |
| Hybrid `R` + KL              | partial (KL only inside PPO)                       | no unified reward combiner |
| PPO clipped surrogate        | `rl_lora_dutch_vqa.py` (`ppo_loss_from_sample`)   | reusable/tested version |
| GRPO group advantage         | `rl_lora_dutch_vqa.py`, `trl_grpo_dutch_vqa.py`   | reusable/tested version |

**Goal of the implementation:** a small, dependency-light, unit-tested reference
package `historian_llm/` that realizes every equation above (especially the CER
reward, the HRM, and the unified hybrid reward), verified on CPU without model
downloads, and usable by the existing training scripts.
