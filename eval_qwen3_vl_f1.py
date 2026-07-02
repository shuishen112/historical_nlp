import argparse
import json
import os
import re
import string
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import transformers
from transformers import AutoProcessor

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _pick_auto_model_loader() -> Any:
    """
    Pick the best available AutoModel loader across transformers versions.
    """
    candidates = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    ]
    for name in candidates:
        loader = getattr(transformers, name, None)
        if loader is not None:
            logger.info(f"Using transformers loader: {name}")
            return loader
    raise RuntimeError(
        "No compatible AutoModel loader found in transformers. "
        "Tried: AutoModelForImageTextToText, AutoModelForVision2Seq, "
        "AutoModelForCausalLM, AutoModel."
    )

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


def token_precision_recall(prediction: str, ground_truth: str) -> Tuple[float, float]:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return precision, recall


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def set_reproducible(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Needed by some CUDA kernels for deterministic behavior.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def generate_answer(
    model: Any,
    processor: AutoProcessor,
    images: Sequence[Image.Image],
    question: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    if not images:
        raise ValueError("No visual inputs provided for generation.")

    content: List[Dict] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})

    messages = [
        {
            "role": "user",
            "content": content,
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_beams=1,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def render_pdf_to_images(
    pdf_path: Path,
    dpi: int,
    max_pages: int,
) -> List[Image.Image]:
    # Try pypdfium2 first (usually lightweight and reliable in ML environments).
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(pdf_path))
        page_count = len(doc)
        page_indices = range(page_count if max_pages <= 0 else min(page_count, max_pages))
        images: List[Image.Image] = []
        for idx in page_indices:
            page = doc[idx]
            bitmap = page.render(scale=dpi / 72.0).to_pil()
            images.append(bitmap.convert("RGB"))
        return images
    except Exception:
        pass

    # Fallback to PyMuPDF if available.
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        images = []
        page_count = doc.page_count
        end_page = page_count if max_pages <= 0 else min(page_count, max_pages)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for idx in range(end_page):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            images.append(img.convert("RGB"))
        return images
    except Exception as exc:
        raise RuntimeError(
            f"Failed to render PDF {pdf_path}. Install pypdfium2 or PyMuPDF. Root cause: {exc}"
        ) from exc


def load_images_from_paths(paths: Sequence[Path]) -> List[Image.Image]:
    loaded: List[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            loaded.append(img.convert("RGB"))
    return loaded


def infer_doc_stem(example: Dict) -> Optional[str]:
    source_name = example.get("source_file") or example.get("image")
    if not source_name:
        return None
    return Path(str(source_name)).stem


def resolve_split_pdf_path(doc_stem: str, split_root: Path) -> Optional[Path]:
    candidates = sorted(split_root.glob(f"*{doc_stem}.pdf"))
    return candidates[0] if candidates else None


def resolve_split_jpg_paths(doc_stem: str, split_root: Path) -> List[Path]:
    dirs = sorted(
        p
        for p in split_root.iterdir()
        if p.is_dir() and doc_stem.lower() in p.name.lower()
    )
    if not dirs:
        return []
    image_paths: List[Path] = []
    for d in dirs:
        image_paths.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    return image_paths


def resolve_visual_inputs(
    example: Dict,
    input_mode: str,
    images_dir: Path,
    original_pdf_dir: Optional[Path],
    split_root: Optional[Path],
    split_jpg_max_images: int,
    pdf_dpi: int,
    pdf_max_pages: int,
) -> Tuple[List[Image.Image], Dict[str, str]]:
    image_name = example.get("image")
    source_file = example.get("source_file")
    doc_stem = infer_doc_stem(example)
    metadata: Dict[str, str] = {"input_mode": input_mode}

    if input_mode == "image":
        if not image_name:
            raise FileNotFoundError("Missing 'image' field in example.")
        image_path = images_dir / str(image_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        metadata["resolved_source"] = str(image_path)
        return load_images_from_paths([image_path]), metadata

    if input_mode == "original_pdf":
        if not source_file:
            raise FileNotFoundError("Missing 'source_file' field for original_pdf mode.")
        if original_pdf_dir is None:
            raise FileNotFoundError("--original-pdf-dir is required for original_pdf mode.")
        pdf_path = original_pdf_dir / str(source_file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Original PDF not found: {pdf_path}")
        metadata["resolved_source"] = str(pdf_path)
        return render_pdf_to_images(pdf_path, dpi=pdf_dpi, max_pages=pdf_max_pages), metadata

    if input_mode == "split_pdf":
        if not doc_stem:
            raise FileNotFoundError("Cannot infer document stem from example.")
        if split_root is None:
            raise FileNotFoundError("--split-root-dir is required for split_pdf mode.")
        split_pdf = resolve_split_pdf_path(doc_stem, split_root)
        if split_pdf is None:
            raise FileNotFoundError(f"Split PDF not found for stem '{doc_stem}' in {split_root}")
        metadata["resolved_source"] = str(split_pdf)
        return render_pdf_to_images(split_pdf, dpi=pdf_dpi, max_pages=pdf_max_pages), metadata

    if input_mode == "split_jpg":
        if not doc_stem:
            raise FileNotFoundError("Cannot infer document stem from example.")
        if split_root is None:
            raise FileNotFoundError("--split-root-dir is required for split_jpg mode.")
        jpg_paths = resolve_split_jpg_paths(doc_stem, split_root)
        if not jpg_paths:
            raise FileNotFoundError(f"Split JPGs not found for stem '{doc_stem}' in {split_root}")
        if split_jpg_max_images > 0:
            jpg_paths = jpg_paths[:split_jpg_max_images]
        metadata["resolved_source"] = ",".join(str(p) for p in jpg_paths)
        return load_images_from_paths(jpg_paths), metadata

    raise ValueError(f"Unsupported input_mode: {input_mode}")


def evaluate(
    dataset_path: Path,
    images_dir: Path,
    input_mode: str,
    original_pdf_dir: Optional[Path],
    split_root_dir: Optional[Path],
    split_jpg_max_images: int,
    pdf_dpi: int,
    pdf_max_pages: int,
    model_id: str,
    lora_adapter: Optional[Path],
    limit: int,
    max_new_tokens: int,
    seed: int,
    deterministic: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[Dict, List[Dict]]:
    set_reproducible(seed=seed, deterministic=deterministic)

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_loader = _pick_auto_model_loader()
    try:
        base_model = model_loader.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load model with an AutoModel loader. "
            "This is usually a transformers compatibility issue. "
            f"Current model_id={model_id}. Original error: {exc}"
        ) from exc
    base_model.eval()
    if lora_adapter is not None:
        if PeftModel is None:
            raise RuntimeError(
                "--lora-adapter was provided but peft is not installed. "
                "Install it with: pip install peft"
            )
        model = PeftModel.from_pretrained(base_model, str(lora_adapter))
        logger.info(f"Loaded LoRA adapter from {lora_adapter}")
    else:
        model = base_model
        logger.info(f"Loaded base model from {model_id}")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    results: List[Dict] = []
    f1_sum = 0.0
    em_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    evaluated = 0
    skipped_missing_visual_input = 0
    skipped_bad_example = 0

    for idx, example in enumerate(data):
        image_name = example.get("image") or example.get("source_file")
        question = example.get("question")
        answer = example.get("answer", "")
        attribute = example.get("attribute", "")

        if not image_name or not question:
            skipped_bad_example += 1
            continue

        try:
            images, source_meta = resolve_visual_inputs(
                example=example,
                input_mode=input_mode,
                images_dir=images_dir,
                original_pdf_dir=original_pdf_dir,
                split_root=split_root_dir,
                split_jpg_max_images=split_jpg_max_images,
                pdf_dpi=pdf_dpi,
                pdf_max_pages=pdf_max_pages,
            )
        except Exception as exc:
            logger.warning(f"Skipping idx={idx} due to visual input issue: {exc}")
            skipped_missing_visual_input += 1
            continue

        prediction = generate_answer(
            model=model,
            processor=processor,
            images=images,
            question=question,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        f1 = token_f1(prediction, answer)
        em = exact_match(prediction, answer)
        precision, recall = token_precision_recall(prediction, answer)
        f1_sum += f1
        em_sum += em
        precision_sum += precision
        recall_sum += recall
        evaluated += 1

        results.append(
            {
                "index": idx,
                "image": image_name,
                "question": question,
                "gold_answer": answer,
                "prediction": prediction,
                "attribute": attribute,
                "resolved_source": source_meta.get("resolved_source", ""),
                "input_mode": input_mode,
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "exact_match": round(em, 4),
            }
        )

        print(
            f"[{evaluated}] F1={f1:.4f} EM={em:.4f} | image={image_name} | q={question}"
        )

    summary = {
        "dataset_path": str(dataset_path),
        "images_dir": str(images_dir),
        "input_mode": input_mode,
        "original_pdf_dir": str(original_pdf_dir) if original_pdf_dir is not None else None,
        "split_root_dir": str(split_root_dir) if split_root_dir is not None else None,
        "split_jpg_max_images": split_jpg_max_images,
        "pdf_dpi": pdf_dpi,
        "pdf_max_pages": pdf_max_pages,
        "model_id": model_id,
        "lora_adapter": str(lora_adapter) if lora_adapter is not None else None,
        "seed": seed,
        "deterministic": deterministic,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "total_examples_loaded": len(data),
        "evaluated_examples": evaluated,
        "skipped_missing_visual_input": skipped_missing_visual_input,
        "skipped_bad_example": skipped_bad_example,
        "avg_f1": f1_sum / evaluated if evaluated else 0.0,
        "avg_precision": precision_sum / evaluated if evaluated else 0.0,
        "avg_recall": recall_sum / evaluated if evaluated else 0.0,
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
        help="Directory containing referenced images (used in image mode).",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="image",
        choices=["image", "original_pdf", "split_pdf", "split_jpg"],
        help="How to resolve visual input for each QA pair.",
    )
    parser.add_argument(
        "--original-pdf-dir",
        type=Path,
        default=None,
        help="Directory of original PDFs (used in original_pdf mode).",
    )
    parser.add_argument(
        "--split-root-dir",
        type=Path,
        default=None,
        help="Split folder root containing split PDFs/JPG subfolders (used in split_* modes).",
    )
    parser.add_argument(
        "--split-jpg-max-images",
        type=int,
        default=0,
        help="Max split JPG tiles per sample (0 = all).",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="PDF rendering DPI for PDF-based modes.",
    )
    parser.add_argument(
        "--pdf-max-pages",
        type=int,
        default=0,
        help="Max pages rendered from each PDF (0 = all).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--lora-adapter",
        type=Path,
        default=None,
        help="Optional path to a LoRA adapter directory/checkpoint to load for evaluation.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible evaluation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable stricter deterministic CUDA behavior.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling in generation (non-deterministic by design).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (used only when --do-sample is set).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling p (used only when --do-sample is set).",
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
        input_mode=args.input_mode,
        original_pdf_dir=args.original_pdf_dir,
        split_root_dir=args.split_root_dir,
        split_jpg_max_images=args.split_jpg_max_images,
        pdf_dpi=args.pdf_dpi,
        pdf_max_pages=args.pdf_max_pages,
        model_id=args.model_id,
        lora_adapter=args.lora_adapter,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        deterministic=args.deterministic,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
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
