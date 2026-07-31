import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import transformers


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
    raise FileNotFoundError(
        "Could not infer image directory. Pass --images-dir explicitly."
    )


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


class DutchVQADataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        images_dir: Path,
        processor: Any,
        max_length: int,
        vision_min_pixels: int,
        vision_max_pixels: int,
    ) -> None:
        self.processor = processor
        self.images_dir = images_dir
        self.max_length = max_length
        self.vision_min_pixels = vision_min_pixels
        self.vision_max_pixels = vision_max_pixels
        self.samples = []

        for ex in samples:
            image_name = ex.get("image")
            question = ex.get("question")
            answer = ex.get("answer")
            if not image_name or not question or answer is None:
                continue
            image_path = images_dir / str(image_name)
            if not image_path.exists():
                continue
            self.samples.append(
                {
                    "image_path": image_path,
                    "question": str(question),
                    "answer": str(answer),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_prompt(self, question: str) -> str:
        return (
            "Answer the question from the provided image. "
            "Return a short, exact answer.\n"
            f"Question: {question}"
        )

    def _build_messages(self, question: str, answer: str = "") -> List[Dict]:
        user_text = self._build_prompt(question)
        if answer:
            return [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": user_text}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
        return [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": user_text}],
            }
        ]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        prompt_messages = self._build_messages(sample["question"])
        full_messages = self._build_messages(sample["question"], sample["answer"])

        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            images_kwargs={
                "size": {
                    "shortest_edge": self.vision_min_pixels,
                    "longest_edge": self.vision_max_pixels,
                }
            },
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            images_kwargs={
                "size": {
                    "shortest_edge": self.vision_min_pixels,
                    "longest_edge": self.vision_max_pixels,
                }
            },
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[0])
        labels[:prompt_len] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in full_inputs:
                out[key] = full_inputs[key][0]
        return out


@dataclass
class VQACollator:
    processor: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        token_features = []
        for feature in features:
            tokens = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            if "mm_token_type_ids" in feature:
                tokens["mm_token_type_ids"] = feature["mm_token_type_ids"]
            token_features.append(tokens)

        batch = self.processor.tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]

        labels = []
        for feature in features:
            label = feature["labels"]
            if label.shape[0] < max_len:
                pad = torch.full((max_len - label.shape[0],), -100, dtype=label.dtype)
                label = torch.cat([label, pad], dim=0)
            labels.append(label)
        batch["labels"] = torch.stack(labels)

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features], dim=0)

        return batch


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def simple_exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def parse_semver(text: str) -> Tuple[int, int, int]:
    nums: List[int] = []
    curr = ""
    for ch in text:
        if ch.isdigit():
            curr += ch
        elif curr:
            nums.append(int(curr))
            curr = ""
        if len(nums) >= 3:
            break
    if curr and len(nums) < 3:
        nums.append(int(curr))
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def validate_torch_transformers_pair() -> None:
    tf_v = parse_semver(transformers.__version__)
    torch_v = parse_semver(torch.__version__)
    # Recent transformers versions disable PyTorch integration for old torch.
    if tf_v >= (5, 0, 0) and torch_v < (2, 4, 0):
        raise RuntimeError(
            f"Incompatible environment: transformers=={transformers.__version__} "
            f"requires newer torch support, but found torch=={torch.__version__}.\n"
            "Fix options:\n"
            "1) Upgrade torch to >=2.4 (recommended if your CUDA stack supports it), OR\n"
            "2) Downgrade transformers to a torch-2.1 compatible version, e.g.:\n"
            "   pip install \"transformers==4.44.2\" \"tokenizers<0.20\""
        )


def load_transformers_symbols():
    auto_processor_cls = getattr(transformers, "AutoProcessor", None)
    trainer_cls = getattr(transformers, "Trainer", None)
    training_args_cls = getattr(transformers, "TrainingArguments", None)
    auto_model_v2s = getattr(transformers, "AutoModelForVision2Seq", None)
    auto_model_it2t = getattr(transformers, "AutoModelForImageTextToText", None)

    if auto_processor_cls is None or trainer_cls is None or training_args_cls is None:
        raise RuntimeError(
            "Transformers model training symbols are unavailable. "
            "This usually means PyTorch support is disabled by an incompatible "
            "transformers/torch pair."
        )
    auto_model_cls = auto_model_v2s if auto_model_v2s is not None else auto_model_it2t
    if auto_model_cls is None:
        raise RuntimeError(
            "No compatible vision-language auto-model class found in transformers. "
            "Expected one of: AutoModelForVision2Seq or AutoModelForImageTextToText."
        )
    return auto_processor_cls, trainer_cls, training_args_cls, auto_model_cls


def load_vl_model(auto_model_cls, model_id: str, dtype: torch.dtype):
    model_load_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    # transformers>=5 prefers `dtype`; older versions use `torch_dtype`.
    if parse_semver(transformers.__version__) >= (5, 0, 0):
        model_load_kwargs["dtype"] = dtype
    else:
        model_load_kwargs["torch_dtype"] = dtype

    return auto_model_cls.from_pretrained(
        model_id,
        **model_load_kwargs,
    )


def generate_answer(
    model,
    processor: Any,
    image_path: Path,
    question: str,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Answer the question from the provided image. "
                        "Return a short, exact answer.\n"
                        f"Question: {question}"
                    ),
                },
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text[0].strip()


def evaluate_split(
    model,
    processor: Any,
    test_dataset: DutchVQADataset,
    max_new_tokens: int,
) -> Dict:
    correct = 0
    total = 0
    rows: List[Dict] = []
    for sample in test_dataset.samples:
        pred = generate_answer(
            model=model,
            processor=processor,
            image_path=sample["image_path"],
            question=sample["question"],
            max_new_tokens=max_new_tokens,
        )
        em = simple_exact_match(pred, sample["answer"])
        correct += int(em)
        total += 1
        rows.append(
            {
                "image": sample["image_path"].name,
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "prediction": pred,
                "exact_match": float(em),
            }
        )
    return {
        "evaluated_examples": total,
        "exact_match": (correct / total) if total else 0.0,
        "predictions": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PEFT fine-tuning + test split eval for Dutch VQA.")
    parser.add_argument("--dataset", type=Path, default=Path("Dutch/DutchVQA_gemini.json"))
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/peft_dutch_vqa"))

    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--train-max-samples", type=int, default=300)
    parser.add_argument("--test-max-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--vision-min-pixels", type=int, default=32 * 32 * 64)
    parser.add_argument("--vision-max-pixels", type=int, default=32 * 32 * 256)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_torch_transformers_pair()

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "The `peft` package is required for parameter-efficient fine-tuning. "
            "Install it with: pip install peft"
        ) from exc

    AutoProcessor, Trainer, TrainingArguments, AutoModelForVL = load_transformers_symbols()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = load_vl_model(AutoModelForVL, args.model_id, dtype)

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
    model.print_trainable_parameters()

    train_ds = DutchVQADataset(
        samples=train_raw,
        images_dir=images_dir,
        processor=processor,
        max_length=args.max_length,
        vision_min_pixels=args.vision_min_pixels,
        vision_max_pixels=args.vision_max_pixels,
    )
    test_ds = DutchVQADataset(
        samples=test_raw,
        images_dir=images_dir,
        processor=processor,
        max_length=args.max_length,
        vision_min_pixels=args.vision_min_pixels,
        vision_max_pixels=args.vision_max_pixels,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No valid training samples found. Check dataset/image paths.")
    if len(test_ds) == 0:
        raise RuntimeError("No valid test samples found. Adjust split or data paths.")

    print(f"Using images from: {images_dir}")
    print(f"Raw examples: {len(raw_data)} | train: {len(train_ds)} | test: {len(test_ds)}")

    training_args_kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "dataloader_num_workers": 2,
    }
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_params:
        training_args_kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in training_args_params:
        training_args_kwargs["evaluation_strategy"] = "no"
    else:
        raise RuntimeError(
            "Neither `eval_strategy` nor `evaluation_strategy` is supported by "
            "this transformers TrainingArguments API."
        )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs),
        train_dataset=train_ds,
        data_collator=VQACollator(processor=processor),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    base_model = load_vl_model(AutoModelForVL, args.model_id, dtype)
    eval_model = PeftModel.from_pretrained(base_model, str(args.output_dir))
    eval_model.eval()

    report = evaluate_split(
        model=eval_model,
        processor=processor,
        test_dataset=test_ds,
        max_new_tokens=args.max_new_tokens,
    )
    report["config"] = {
        "dataset": str(args.dataset),
        "images_dir": str(images_dir),
        "model_id": args.model_id,
        "output_dir": str(args.output_dir),
        "test_ratio": args.test_ratio,
        "train_max_samples": args.train_max_samples,
        "test_max_samples": args.test_max_samples,
    }

    metrics_path = args.output_dir / "test_predictions.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter + processor to: {args.output_dir}")
    print(f"Saved test report to: {metrics_path}")
    print(f"Test Exact Match: {report['exact_match']:.4f}")


if __name__ == "__main__":
    main()
