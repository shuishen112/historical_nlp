import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


def load_vqa_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def train_val_split(data: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_data = shuffled[:val_size]
    train_data = shuffled[val_size:]
    return train_data, val_data


class DutchVQASFTDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        images_dir: Path,
        processor: AutoProcessor,
        max_length: int,
        vision_min_pixels: int,
        vision_max_pixels: int,
    ):
        self.processor = processor
        self.images_dir = images_dir
        self.max_length = max_length
        self.vision_min_pixels = vision_min_pixels
        self.vision_max_pixels = vision_max_pixels
        self.samples = []

        for ex in data:
            image_name = ex.get("image")
            question = ex.get("question")
            answer = ex.get("answer")
            if not image_name or not question or answer is None:
                continue
            image_path = images_dir / image_name
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

    def _build_messages(self, question: str, answer: str = "") -> List[Dict]:
        user_text = (
            "Answer the question based on the image. "
            "Keep the answer concise and directly grounded in the image text.\n"
            f"Question: {question}"
        )
        if answer:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
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

        prompt_len = prompt_inputs["input_ids"].shape[1]
        prompt_len = min(prompt_len, labels.shape[0])
        labels[:prompt_len] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in full_inputs:
                out[key] = full_inputs[key][0]
        if "mm_token_type_ids" in full_inputs:
            out["mm_token_type_ids"] = full_inputs["mm_token_type_ids"][0]
        return out


@dataclass
class VQACollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            token_features = {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            if "mm_token_type_ids" in f:
                token_features["mm_token_type_ids"] = f["mm_token_type_ids"]
            input_features.append(token_features)
        batch = self.processor.tokenizer.pad(input_features, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            label = f["labels"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Supervised fine-tuning for Qwen3-VL on Dutch VQA.")
    parser.add_argument("--dataset", type=Path, default=Path("Dutch/DutchVQA.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("Dutch/extracted_images"))
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("qwen3_vl_2b_dutch_vqa_sft"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--vision-min-pixels", type=int, default=32 * 32 * 64)
    parser.add_argument("--vision-max-pixels", type=int, default=32 * 32 * 256)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_data = load_vqa_json(args.dataset)
    train_raw, val_raw = train_val_split(raw_data, args.val_ratio, args.seed)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_ds = DutchVQASFTDataset(
        train_raw,
        args.images_dir,
        processor,
        args.max_length,
        args.vision_min_pixels,
        args.vision_max_pixels,
    )
    val_ds = DutchVQASFTDataset(
        val_raw,
        args.images_dir,
        processor,
        args.max_length,
        args.vision_min_pixels,
        args.vision_max_pixels,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No valid training samples found. Check dataset path and image directory.")

    print(f"Loaded samples -> train: {len(train_ds)}, val: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if len(val_ds) > 0 else "no",
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    collator = VQACollator(processor=processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved fine-tuned adapter/model files to: {args.output_dir}")


if __name__ == "__main__":
    main()
