import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and normalize spaces."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def generate_answer(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    question: str,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def evaluate(
    dataset_path: Path,
    images_dir: Path,
    model_id: str,
    limit: int,
    max_new_tokens: int,
) -> Tuple[Dict, List[Dict]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    results: List[Dict] = []
    f1_sum = 0.0
    em_sum = 0.0
    evaluated = 0
    skipped_missing_image = 0
    skipped_bad_example = 0

    for idx, example in enumerate(data):
        image_name = example.get("image")
        question = example.get("question")
        answer = example.get("answer", "")
        attribute = example.get("attribute", "")

        if not image_name or not question:
            skipped_bad_example += 1
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        prediction = generate_answer(
            model=model,
            processor=processor,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        f1 = token_f1(prediction, answer)
        em = exact_match(prediction, answer)
        f1_sum += f1
        em_sum += em
        evaluated += 1

        results.append(
            {
                "index": idx,
                "image": image_name,
                "question": question,
                "gold_answer": answer,
                "prediction": prediction,
                "attribute": attribute,
                "f1": round(f1, 4),
                "exact_match": round(em, 4),
            }
        )

        print(
            f"[{evaluated}] F1={f1:.4f} EM={em:.4f} | image={image_name} | q={question}"
        )

    summary = {
        "dataset_path": str(dataset_path),
        "images_dir": str(images_dir),
        "model_id": model_id,
        "total_examples_loaded": len(data),
        "evaluated_examples": evaluated,
        "skipped_missing_image": skipped_missing_image,
        "skipped_bad_example": skipped_bad_example,
        "avg_f1": f1_sum / evaluated if evaluated else 0.0,
        "avg_exact_match": em_sum / evaluated if evaluated else 0.0,
    }
    return summary, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL model on Dutch VQA with token-level F1."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Dutch/DutchVQA.json"),
        help="Path to VQA JSON file.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("Dutch/extracted_images"),
        help="Directory containing referenced images.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only first N examples (0 = all).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per answer.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("qwen3_vl_dutch_vqa_eval.json"),
        help="Where to save detailed predictions + metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, results = evaluate(
        dataset_path=args.dataset,
        images_dir=args.images_dir,
        model_id=args.model_id,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )

    report = {"summary": summary, "results": results}
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved detailed report to: {args.output_json}")


if __name__ == "__main__":
    main()
