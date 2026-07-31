import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm


def parse_semver(text: str) -> Tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", text)[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def validate_torch_transformers_pair() -> None:
    tf_v = parse_semver(transformers.__version__)
    torch_v = parse_semver(torch.__version__)
    if tf_v >= (5, 0, 0) and torch_v < (2, 4, 0):
        raise RuntimeError(
            f"Incompatible environment: transformers=={transformers.__version__} "
            f"with torch=={torch.__version__}. Upgrade torch>=2.4 or use older transformers."
        )


def load_transformers_symbols():
    auto_processor_cls = getattr(transformers, "AutoProcessor", None)
    auto_model_v2s = getattr(transformers, "AutoModelForVision2Seq", None)
    auto_model_it2t = getattr(transformers, "AutoModelForImageTextToText", None)
    auto_model_cls = auto_model_v2s if auto_model_v2s is not None else auto_model_it2t
    if auto_processor_cls is None or auto_model_cls is None:
        raise RuntimeError(
            "No compatible AutoProcessor/AutoModelForVision2Seq/AutoModelForImageTextToText found."
        )
    return auto_processor_cls, auto_model_cls


def load_vl_model(auto_model_cls, model_id: str, dtype: torch.dtype):
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    if parse_semver(transformers.__version__) >= (5, 0, 0):
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    return auto_model_cls.from_pretrained(model_id, **kwargs)


# Reward stack (Section 3 of the HistorianLLM methodology): R_auto = 1 - CER,
# optional HRM R_expert, combined as R = alpha*R_auto + beta*R_expert. The KL term
# is handled inside the PPO/GRPO objective via --ppo-kl-coef.
from historian_llm.rewards import (  # noqa: E402
    auto_reward,
    char_error_rate,
    exact_match,
    normalize_text,
    token_f1,
)


@dataclass
class RewardConfig:
    """Configuration for the sequence-level VQA reward."""

    mode: str = "cer"          # "cer" (1-CER), "blend" (CER+F1), or "legacy" (EM+F1)
    f1_weight: float = 0.5     # blend weight in "blend"/"legacy" modes
    alpha: float = 1.0         # weight on R_auto
    beta: float = 1.0          # weight on R_expert (HRM); ignored if no HRM
    hrm_scorer: Optional[Any] = None  # callable(pred, gold) -> float, or None


def _base_reward(pred: str, gold: str, cfg: RewardConfig) -> float:
    if cfg.mode == "legacy":
        return (1.0 - cfg.f1_weight) * exact_match(pred, gold) + cfg.f1_weight * token_f1(
            pred, gold
        )
    if cfg.mode == "blend":
        return (1.0 - cfg.f1_weight) * auto_reward(pred, gold) + cfg.f1_weight * token_f1(
            pred, gold
        )
    return auto_reward(pred, gold)  # "cer"


def compute_reward(pred: str, gold: str, cfg: RewardConfig) -> float:
    """R(x, y) = alpha*R_auto + beta*R_expert (KL added in the policy objective)."""
    score = cfg.alpha * _base_reward(pred, gold, cfg)
    if cfg.hrm_scorer is not None:
        score += cfg.beta * float(cfg.hrm_scorer(pred, gold))
    return float(score)


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def find_images_dir(dataset_path: Path, user_images_dir: Optional[Path]) -> Path:
    if user_images_dir is not None:
        return user_images_dir
    candidates = [
        dataset_path.parent / "extracted_images of Dutch",
        dataset_path.parent / "extracted_images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not infer image directory. Pass --images-dir explicitly.")


def split_train_test(
    raw_data: List[Dict],
    test_ratio: float,
    seed: int,
    train_max_samples: int,
    test_max_samples: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    data = raw_data[:]
    rng.shuffle(data)
    test_size = max(1, int(len(data) * test_ratio))
    test_data = data[:test_size]
    train_data = data[test_size:]
    if train_max_samples > 0:
        train_data = train_data[:train_max_samples]
    if test_max_samples > 0:
        test_data = test_data[:test_max_samples]
    return train_data, test_data


def make_samples(raw: List[Dict], images_dir: Path) -> List[Dict]:
    samples: List[Dict] = []
    for ex in raw:
        image_name = ex.get("image")
        question = ex.get("question")
        answer = ex.get("answer")
        if not image_name or not question or answer is None:
            continue
        image_path = images_dir / str(image_name)
        if not image_path.exists():
            continue
        samples.append(
            {
                "image_path": image_path,
                "question": str(question),
                "answer": str(answer),
            }
        )
    return samples


def build_prompt(question: str) -> str:
    return (
        "Answer the question from the provided image. "
        "Return a short, exact answer.\n"
        f"Question: {question}"
    )


def prepare_inputs(
    processor: Any,
    image_path: Path,
    question: str,
    device: torch.device,
    max_prompt_length: int,
    vision_min_pixels: int,
    vision_max_pixels: int,
) -> Dict[str, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_prompt(question)},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # NOTE: For multimodal chat templates, truncation can desync image-token placeholders
    # from tokenized input_ids in recent transformers versions.
    # We intentionally avoid truncation here to preserve image/text alignment.
    _ = max_prompt_length  # kept for CLI compatibility
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": vision_min_pixels,
                "longest_edge": vision_max_pixels,
            }
        },
    )
    return {k: v.to(device) for k, v in inputs.items()}


def sample_for_rl(
    model,
    processor: Any,
    model_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, torch.Tensor, int]:
    with torch.no_grad():
        sequences = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_len = model_inputs["input_ids"].shape[1]
    gen_tokens = sequences[:, prompt_len:]
    text = processor.batch_decode(
        gen_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text, sequences, prompt_len


def build_forward_kwargs(
    model_inputs: Dict[str, torch.Tensor],
    sequences: torch.Tensor,
    prompt_len: int,
    add_labels: bool,
) -> Dict[str, torch.Tensor]:
    labels = sequences.clone()
    labels[:, :prompt_len] = -100
    attn_mask = torch.ones_like(sequences, dtype=torch.long, device=sequences.device)

    forward_kwargs = {
        "input_ids": sequences,
        "attention_mask": attn_mask,
    }
    if add_labels:
        forward_kwargs["labels"] = labels
    for key in ("pixel_values", "image_grid_thw"):
        if key in model_inputs:
            forward_kwargs[key] = model_inputs[key]
    if "mm_token_type_ids" in model_inputs:
        mm_token_type_ids = model_inputs["mm_token_type_ids"]
        total_len = sequences.shape[1]
        if mm_token_type_ids.shape[1] < total_len:
            # Generated continuation is text; extend with text token-type ids (0).
            pad = torch.zeros(
                (mm_token_type_ids.shape[0], total_len - mm_token_type_ids.shape[1]),
                dtype=mm_token_type_ids.dtype,
                device=mm_token_type_ids.device,
            )
            mm_token_type_ids = torch.cat([mm_token_type_ids, pad], dim=1)
        else:
            mm_token_type_ids = mm_token_type_ids[:, :total_len]
        forward_kwargs["mm_token_type_ids"] = mm_token_type_ids
    return forward_kwargs


def generate_text(
    model,
    processor: Any,
    model_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
) -> str:
    with torch.no_grad():
        sequences = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_len = model_inputs["input_ids"].shape[1]
    gen_tokens = sequences[:, prompt_len:]
    text = processor.batch_decode(
        gen_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del sequences
    return text


def rl_loss_from_sample(
    model,
    model_inputs: Dict[str, torch.Tensor],
    sampled_sequences: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    forward_kwargs = build_forward_kwargs(
        model_inputs=model_inputs,
        sequences=sampled_sequences,
        prompt_len=prompt_len,
        add_labels=True,
    )
    outputs = model(**forward_kwargs)
    return outputs.loss


def compute_sequence_logprob(
    model,
    model_inputs: Dict[str, torch.Tensor],
    sampled_sequences: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    forward_kwargs = build_forward_kwargs(
        model_inputs=model_inputs,
        sequences=sampled_sequences,
        prompt_len=prompt_len,
        add_labels=False,
    )
    outputs = model(**forward_kwargs)
    logits = outputs.logits[:, :-1, :]
    targets = sampled_sequences[:, 1:]

    token_logprobs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    gen_mask = torch.zeros_like(token_logprobs, dtype=token_logprobs.dtype)
    start = max(0, prompt_len - 1)
    if start < gen_mask.shape[1]:
        gen_mask[:, start:] = 1.0
    denom = torch.clamp(gen_mask.sum(), min=1.0)
    seq_logprob = (token_logprobs * gen_mask).sum() / denom
    return seq_logprob


def ppo_loss_from_sample(
    model,
    model_inputs: Dict[str, torch.Tensor],
    sampled_sequences: torch.Tensor,
    prompt_len: int,
    old_logprob: torch.Tensor,
    advantage: float,
    clip_epsilon: float,
    kl_coef: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    new_logprob = compute_sequence_logprob(
        model=model,
        model_inputs=model_inputs,
        sampled_sequences=sampled_sequences,
        prompt_len=prompt_len,
    )
    ratio = torch.exp(new_logprob - old_logprob)
    adv = torch.tensor(float(advantage), dtype=ratio.dtype, device=ratio.device)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    loss = -torch.minimum(unclipped, clipped)
    if kl_coef > 0:
        approx_kl = (new_logprob - old_logprob) ** 2
        loss = loss + kl_coef * approx_kl
    return loss, new_logprob.detach()


def evaluate(
    model,
    processor: Any,
    samples: List[Dict],
    device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
    vision_min_pixels: int,
    vision_max_pixels: int,
    limit: int = 0,
) -> Dict:
    if limit > 0:
        samples = samples[:limit]
    rows: List[Dict] = []
    em_sum = 0.0
    f1_sum = 0.0
    for sample in tqdm(samples, desc="Eval", leave=False):
        model_inputs = prepare_inputs(
            processor=processor,
            image_path=sample["image_path"],
            question=sample["question"],
            device=device,
            max_prompt_length=max_prompt_length,
            vision_min_pixels=vision_min_pixels,
            vision_max_pixels=vision_max_pixels,
        )
        pred = generate_text(
            model=model,
            processor=processor,
            model_inputs=model_inputs,
            max_new_tokens=max_new_tokens,
        )
        em = exact_match(pred, sample["answer"])
        f1 = token_f1(pred, sample["answer"])
        em_sum += em
        f1_sum += f1
        rows.append(
            {
                "image": sample["image_path"].name,
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "prediction": pred,
                "exact_match": round(em, 4),
                "token_f1": round(f1, 4),
            }
        )
    n = max(1, len(samples))
    return {
        "evaluated_examples": len(samples),
        "avg_exact_match": em_sum / n,
        "avg_token_f1": f1_sum / n,
        "predictions": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("RL (REINFORCE/PPO/GRPO) LoRA fine-tuning for Dutch VQA.")
    parser.add_argument("--dataset", type=Path, default=Path("Dutch/DutchVQA_gemini.json"))
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl_lora_dutch_vqa"))

    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--train-max-samples", type=int, default=300)
    parser.add_argument("--test-max-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--vision-min-pixels", type=int, default=32 * 32 * 28)
    parser.add_argument("--vision-max-pixels", type=int, default=32 * 32 * 96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument(
        "--reward-mode",
        choices=["cer", "blend", "legacy"],
        default="cer",
        help="R_auto source: 'cer' (1-CER, paper default), 'blend' (CER+F1), or "
        "'legacy' (EM+F1).",
    )
    parser.add_argument("--reward-alpha", type=float, default=1.0, help="Weight on R_auto.")
    parser.add_argument(
        "--reward-beta", type=float, default=1.0, help="Weight on R_expert (HRM)."
    )
    parser.add_argument(
        "--hrm-checkpoint",
        type=Path,
        default=None,
        help="Path to a trained Historian Reward Model (.pt). Enables R_expert.",
    )
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--empty-cache-steps", type=int, default=10)
    parser.add_argument("--algo", choices=["reinforce", "ppo", "grpo"], default="reinforce")
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-kl-coef", type=float, default=0.0)
    parser.add_argument("--grpo-group-size", type=int, default=2)
    parser.add_argument("--grpo-std-eps", type=float, default=1e-4)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--baseline",
        choices=["greedy", "running_mean"],
        default="greedy",
        help="Advantage baseline: greedy rollout reward or exponential running mean.",
    )
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="historical-nlp-vqa-rl")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-log-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_torch_transformers_pair()

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("`peft` is required. Install with: pip install peft") from exc

    AutoProcessor, AutoModelForVL = load_transformers_symbols()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = find_images_dir(args.dataset, args.images_dir)
    raw_data = load_json(args.dataset)
    train_raw, test_raw = split_train_test(
        raw_data=raw_data,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_max_samples=args.train_max_samples,
        test_max_samples=args.test_max_samples,
    )

    train_samples = make_samples(train_raw, images_dir)
    test_samples = make_samples(test_raw, images_dir)
    if len(train_samples) == 0:
        raise RuntimeError("No valid train samples after filtering missing images.")
    if len(test_samples) == 0:
        raise RuntimeError("No valid test samples after filtering missing images.")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging requested but `wandb` is not installed. "
                "Install with: pip install wandb"
            ) from exc

        run_name = args.wandb_name.strip() or f"rl-lora-{int(time.time())}"
        init_kwargs: Dict[str, Any] = {
            "project": args.wandb_project,
            "name": run_name,
            "config": vars(args),
        }
        if args.wandb_entity.strip():
            init_kwargs["entity"] = args.wandb_entity.strip()
        wandb_run = wandb.init(**init_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = load_vl_model(AutoModelForVL, args.model_id, dtype)
    model.to(device)
    model.train()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    optimizer = AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Using images from: {images_dir}")
    print(f"Raw examples: {len(raw_data)} | train: {len(train_samples)} | test: {len(test_samples)}")

    hrm_scorer = None
    if args.hrm_checkpoint is not None:
        from historian_llm.hrm import hrm_score, load_hrm

        hrm_model = load_hrm(args.hrm_checkpoint)
        hrm_scorer = lambda pred, gold: hrm_score(hrm_model, pred, gold)  # noqa: E731
        print(f"Loaded Historian Reward Model (R_expert) from: {args.hrm_checkpoint}")
    reward_cfg = RewardConfig(
        mode=args.reward_mode,
        f1_weight=args.f1_weight,
        alpha=args.reward_alpha,
        beta=args.reward_beta,
        hrm_scorer=hrm_scorer,
    )
    print(
        f"Reward: mode={reward_cfg.mode} alpha={reward_cfg.alpha} "
        f"beta={reward_cfg.beta} hrm={'on' if hrm_scorer else 'off'} "
        f"ppo_kl_coef={args.ppo_kl_coef}"
    )

    running_mean_reward = 0.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        random.shuffle(train_samples)
        pbar = tqdm(enumerate(train_samples, start=1), total=len(train_samples), desc=f"Epoch {epoch+1}")
        for step_idx, sample in pbar:
            model_inputs = prepare_inputs(
                processor=processor,
                image_path=sample["image_path"],
                question=sample["question"],
                device=device,
                max_prompt_length=args.max_prompt_length,
                vision_min_pixels=args.vision_min_pixels,
                vision_max_pixels=args.vision_max_pixels,
            )

            log_reward = 0.0
            log_baseline = 0.0
            log_advantage = 0.0
            log_primary_loss = 0.0
            log_ratio = 1.0

            if args.algo in ("reinforce", "ppo"):
                sampled_text, sampled_sequences, prompt_len = sample_for_rl(
                    model=model,
                    processor=processor,
                    model_inputs=model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                sampled_reward = compute_reward(sampled_text, sample["answer"], reward_cfg)

                if args.baseline == "greedy":
                    greedy_text = generate_text(
                        model=model,
                        processor=processor,
                        model_inputs=model_inputs,
                        max_new_tokens=args.max_new_tokens,
                    )
                    baseline_reward = compute_reward(greedy_text, sample["answer"], reward_cfg)
                else:
                    running_mean_reward = (
                        args.baseline_momentum * running_mean_reward
                        + (1.0 - args.baseline_momentum) * sampled_reward
                    )
                    baseline_reward = running_mean_reward

                advantage = sampled_reward - baseline_reward

                if args.algo == "reinforce":
                    base_loss = rl_loss_from_sample(
                        model=model,
                        model_inputs=model_inputs,
                        sampled_sequences=sampled_sequences,
                        prompt_len=prompt_len,
                    )
                    loss = base_loss * advantage
                else:
                    with torch.no_grad():
                        old_logprob = compute_sequence_logprob(
                            model=model,
                            model_inputs=model_inputs,
                            sampled_sequences=sampled_sequences,
                            prompt_len=prompt_len,
                        )
                    loss, new_logprob = ppo_loss_from_sample(
                        model=model,
                        model_inputs=model_inputs,
                        sampled_sequences=sampled_sequences,
                        prompt_len=prompt_len,
                        old_logprob=old_logprob,
                        advantage=advantage,
                        clip_epsilon=args.ppo_clip_eps,
                        kl_coef=args.ppo_kl_coef,
                    )
                    log_ratio = float(torch.exp(new_logprob - old_logprob).item())
                    base_loss = loss

                (loss / max(1, args.grad_accum)).backward()
                del sampled_sequences

                log_reward = float(sampled_reward)
                log_baseline = float(baseline_reward)
                log_advantage = float(advantage)
                log_primary_loss = float(base_loss.item())
            else:
                group_size = max(2, args.grpo_group_size)
                group_sequences: List[torch.Tensor] = []
                group_prompt_lens: List[int] = []
                group_rewards: List[float] = []
                for _ in range(group_size):
                    sampled_text, sampled_sequences, prompt_len = sample_for_rl(
                        model=model,
                        processor=processor,
                        model_inputs=model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    group_rewards.append(compute_reward(sampled_text, sample["answer"], reward_cfg))
                    group_sequences.append(sampled_sequences)
                    group_prompt_lens.append(prompt_len)

                rewards_t = torch.tensor(group_rewards, dtype=torch.float32, device=device)
                mean_r = rewards_t.mean()
                std_r = rewards_t.std(unbiased=False)
                advantages = ((rewards_t - mean_r) / (std_r + args.grpo_std_eps)).tolist()

                old_logprobs: List[torch.Tensor] = []
                with torch.no_grad():
                    for seq, p_len in zip(group_sequences, group_prompt_lens):
                        old_logprobs.append(
                            compute_sequence_logprob(
                                model=model,
                                model_inputs=model_inputs,
                                sampled_sequences=seq,
                                prompt_len=p_len,
                            )
                        )

                loss_terms: List[torch.Tensor] = []
                ratios: List[float] = []
                for seq, p_len, old_lp, adv in zip(
                    group_sequences, group_prompt_lens, old_logprobs, advantages
                ):
                    l_i, new_lp = ppo_loss_from_sample(
                        model=model,
                        model_inputs=model_inputs,
                        sampled_sequences=seq,
                        prompt_len=p_len,
                        old_logprob=old_lp,
                        advantage=float(adv),
                        clip_epsilon=args.ppo_clip_eps,
                        kl_coef=args.ppo_kl_coef,
                    )
                    loss_terms.append(l_i)
                    ratios.append(float(torch.exp(new_lp - old_lp).item()))

                loss = torch.stack(loss_terms).mean()
                (loss / max(1, args.grad_accum)).backward()
                for seq in group_sequences:
                    del seq

                log_reward = float(mean_r.item())
                log_baseline = float(mean_r.item())
                log_advantage = float(sum(advantages) / max(1, len(advantages)))
                log_primary_loss = float(loss.item())
                log_ratio = float(sum(ratios) / max(1, len(ratios)))

            if wandb_run is not None and step_idx % max(1, args.wandb_log_steps) == 0:
                wandb.log(
                    {
                        "train/algo": args.algo,
                        "train/reward": log_reward,
                        "train/baseline_reward": log_baseline,
                        "train/advantage": log_advantage,
                        "train/policy_loss": log_primary_loss,
                        "train/ratio": log_ratio,
                        "train/epoch": float(epoch + 1),
                        "train/step_in_epoch": int(step_idx),
                    }
                )

            if (
                torch.cuda.is_available()
                and args.empty_cache_steps > 0
                and step_idx % args.empty_cache_steps == 0
            ):
                torch.cuda.empty_cache()

            if step_idx % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if wandb_run is not None:
                    wandb.log({"train/global_step": int(global_step)})

            if step_idx % args.log_steps == 0:
                pbar.set_postfix(
                    algo=args.algo,
                    reward=f"{log_reward:.3f}",
                    baseline=f"{log_baseline:.3f}",
                    advantage=f"{log_advantage:.3f}",
                    loss=f"{log_primary_loss:.3f}",
                )

    # Flush remaining gradients.
    if any(p.grad is not None for p in model.parameters() if p.requires_grad):
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved RL LoRA adapter + processor to: {args.output_dir}")

    model.eval()
    eval_report = evaluate(
        model=model,
        processor=processor,
        samples=test_samples,
        device=device,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        vision_min_pixels=args.vision_min_pixels,
        vision_max_pixels=args.vision_max_pixels,
        limit=args.eval_limit,
    )
    eval_report["config"] = {
        "dataset": str(args.dataset),
        "images_dir": str(images_dir),
        "model_id": args.model_id,
        "output_dir": str(args.output_dir),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "epochs": args.epochs,
        "algo": args.algo,
        "baseline": args.baseline,
    }

    metrics_path = args.output_dir / "rl_test_predictions.json"
    metrics_path.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved RL test report to: {metrics_path}")
    print(
        "RL Eval -> "
        f"Exact Match: {eval_report['avg_exact_match']:.4f}, "
        f"Token F1: {eval_report['avg_token_f1']:.4f}"
    )

    if wandb_run is not None:
        wandb.log(
            {
                "eval/avg_exact_match": float(eval_report["avg_exact_match"]),
                "eval/avg_token_f1": float(eval_report["avg_token_f1"]),
                "eval/evaluated_examples": int(eval_report["evaluated_examples"]),
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
