import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

import logging

# Reuse the exact metrics + data resolution from the main harness so results are
# comparable across models and baselines.
from eval_base import (
    exact_match,
    resolve_visual_inputs,
    set_reproducible,
    token_f1,
    token_precision_recall,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# OCR backend (shared by both baselines)
# ---------------------------------------------------------------------------
class PaddleOCREngine:
    """Thin wrapper around PaddleOCR that is robust to the 2.x / 3.x API split.

    Returns, per image, a list of ``(text, box, confidence)`` in reading order
    (top-to-bottom, left-to-right).
    """

    def __init__(
        self,
        lang: str = "",
        use_gpu: Optional[bool] = None,
        max_side: int = 3072,
    ) -> None:
        self.lang = lang
        # None = auto-detect from Paddle's own CUDA capability at load() time.
        # IMPORTANT: do NOT key this off torch.cuda -- torch and paddle can have
        # different device support (e.g. GPU torch but a CPU-only paddlepaddle
        # build). Using the wrong device silently re-enables the buggy oneDNN
        # CPU path.
        self._use_gpu_override = use_gpu
        self.use_gpu = False  # resolved in load()
        self.max_side = max_side
        self._ocr = None

    def load(self) -> None:
        try:
            import paddle  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The `paddlepaddle` backend is not installed (only `paddleocr` was found), "
                "so PaddleOCR cannot run. Install the compute backend:\n"
                "  pip install paddlepaddle-gpu   # GPU (recommended)\n"
                "  # or: pip install paddlepaddle  # CPU only (very slow on large scans)\n"
                f"Original import error: {exc}"
            ) from exc
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr\n"
                f"Original import error: {exc}"
            ) from exc

        # Resolve GPU usage from Paddle's OWN capability (not torch's).
        if self._use_gpu_override is not None:
            self.use_gpu = bool(self._use_gpu_override)
        else:
            try:
                self.use_gpu = bool(paddle.is_compiled_with_cuda()) and (
                    paddle.device.cuda.device_count() > 0
                )
            except Exception:
                self.use_gpu = False
        device = "gpu" if self.use_gpu else "cpu"
        logger.info(
            f"[paddleocr] paddle={paddle.__version__} "
            f"compiled_with_cuda={paddle.is_compiled_with_cuda()} -> device={device}"
        )
        # On CPU, Paddle 3.x's oneDNN/PIR path hits an unimplemented kernel
        # ("ConvertPirAttribute2RuntimeAttribute not support ...") on these
        # models, so MKL-DNN MUST be disabled when running on CPU. On GPU the
        # oneDNN path is not used, so we leave it default.
        # NOTE: an unsupported `lang` (e.g. "latin" on PP-OCRv6) raises at
        # construction; we therefore also try configs without `lang` so we never
        # silently fall back to a bare PaddleOCR() that re-enables oneDNN.
        common_3x: Dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if not self.use_gpu:
            common_3x["enable_mkldnn"] = False

        has_lang = bool(self.lang)
        candidates: List[Dict[str, Any]] = []
        # 3.x, mkldnn-safe. Prefer with-lang, then without-lang (known good).
        if has_lang:
            candidates.append({**common_3x, "lang": self.lang, "device": device})
        candidates.append({**common_3x, "device": device})
        if has_lang:
            candidates.append({**common_3x, "lang": self.lang})
        candidates.append(dict(common_3x))
        # 2.x-style fallbacks (still mkldnn-safe on CPU).
        legacy_mkldnn = {} if self.use_gpu else {"enable_mkldnn": False}
        if has_lang:
            candidates.append(
                {"use_angle_cls": True, "lang": self.lang, "show_log": False, **legacy_mkldnn}
            )
        candidates.append({"use_angle_cls": True, "show_log": False, **legacy_mkldnn})

        last_exc: Optional[Exception] = None
        for kwargs in candidates:
            try:
                self._ocr = PaddleOCR(**kwargs)
                logger.info(f"[paddleocr] Initialized (device={device}) with kwargs={list(kwargs)}")
                return
            except Exception as exc:  # try the next signature
                logger.warning(f"[paddleocr] init failed for kwargs={list(kwargs)}: {exc}")
                last_exc = exc
        raise RuntimeError(f"Failed to initialize PaddleOCR. Last error: {last_exc}")

    def _maybe_downscale(self, image: Image.Image) -> Image.Image:
        if self.max_side <= 0:
            return image
        w, h = image.size
        longest = max(w, h)
        if longest <= self.max_side:
            return image
        scale = self.max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.BILINEAR)

    @staticmethod
    def _box_sort_key(box: Sequence[Sequence[float]]) -> Tuple[float, float]:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        return (min(ys), min(xs))

    def read(self, image: Image.Image) -> List[Tuple[str, List[List[float]], float]]:
        if self._ocr is None:
            raise RuntimeError("PaddleOCREngine.load() must be called first.")
        img = np.array(self._maybe_downscale(image.convert("RGB")))

        # 3.x prefers .predict / .ocr without cls kwarg; 2.x uses .ocr(img, cls=True).
        raw = None
        for call in (
            lambda: self._ocr.ocr(img, cls=True),
            lambda: self._ocr.ocr(img),
            lambda: self._ocr.predict(img),
        ):
            try:
                raw = call()
                break
            except TypeError:
                continue
        if raw is None:
            return []

        lines: List[Tuple[str, List[List[float]], float]] = []
        # Expected 2.x shape: [ [ [box, (text, conf)], ... ] ]
        try:
            page = raw[0] if raw and isinstance(raw[0], list) else raw
            for item in page:
                box = item[0]
                text, conf = item[1][0], float(item[1][1])
                lines.append((text, box, conf))
        except Exception:
            # 3.x dict-style result: {'rec_texts': [...], 'rec_boxes'/'dt_polys': [...]}
            try:
                res = raw[0] if isinstance(raw, list) else raw
                res = getattr(res, "json", res) if not isinstance(res, dict) else res
                texts = res.get("rec_texts", [])
                boxes = res.get("dt_polys", res.get("rec_boxes", [[]] * len(texts)))
                scores = res.get("rec_scores", [1.0] * len(texts))
                for t, b, s in zip(texts, boxes, scores):
                    lines.append((str(t), np.asarray(b).tolist(), float(s)))
            except Exception as exc:
                logger.warning(f"[paddleocr] Unrecognized result format: {exc}")
                return []

        lines.sort(key=lambda x: self._box_sort_key(x[1]) if x[1] else (0.0, 0.0))
        return lines

    def read_text(self, image: Image.Image) -> str:
        return "\n".join(t for t, _, _ in self.read(image) if t.strip())


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------
class BaselineRunner:
    def load(self) -> None:
        raise NotImplementedError

    def answer(
        self,
        images: Sequence[Image.Image],
        question: str,
        cache_key: Optional[str] = None,
    ) -> str:
        raise NotImplementedError


class PaddleOCRRunner(BaselineRunner):
    """PaddleOCR transcript -> (optional) text-LLM reader -> short answer."""

    def __init__(
        self,
        ocr_lang: str,
        reader_mode: str,
        reader_model_id: str,
        max_new_tokens: int,
        max_context_chars: int,
        ocr_max_side: int = 3072,
    ) -> None:
        self.ocr = PaddleOCREngine(lang=ocr_lang, max_side=ocr_max_side)
        self.reader_mode = reader_mode
        self.reader_model_id = reader_model_id
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.reader = None
        self.tokenizer = None
        self._ocr_cache: Dict[str, str] = {}

    def load(self) -> None:
        self.ocr.load()
        if self.reader_mode == "llm":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.reader_model_id, trust_remote_code=True
            )
            self.reader = AutoModelForCausalLM.from_pretrained(
                self.reader_model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.reader.eval()
            logger.info(f"[paddleocr] Loaded reader LLM: {self.reader_model_id}")
        else:
            logger.info("[paddleocr] reader_mode=ocr_only (raw transcript is the prediction)")

    def _ocr_context(
        self, images: Sequence[Image.Image], cache_key: Optional[str] = None
    ) -> str:
        # Many QA pairs reference the same page image; OCR each unique page once.
        if cache_key is not None and cache_key in self._ocr_cache:
            return self._ocr_cache[cache_key]
        text = "\n".join(self.ocr.read_text(img) for img in images)
        if self.max_context_chars > 0:
            text = text[: self.max_context_chars]
        if cache_key is not None:
            self._ocr_cache[cache_key] = text
        return text

    def answer(
        self,
        images: Sequence[Image.Image],
        question: str,
        cache_key: Optional[str] = None,
    ) -> str:
        context = self._ocr_context(images, cache_key)
        if self.reader_mode == "ocr_only":
            return context.strip()

        prompt = (
            "You are reading text extracted by OCR from a historical newspaper page. "
            "Using ONLY the OCR text below, answer the question with a short, exact span. "
            "If the answer is not present, reply with the closest span.\n\n"
            f"OCR text:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        # NOTE: on recent transformers, apply_chat_template(return_tensors="pt")
        # returns a BatchEncoding (dict-like), not a bare tensor. Use
        # return_dict=True and unpack so generate() gets input_ids + mask.
        enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v.to(self.reader.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.reader.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][input_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


class LayoutLLMRunner(BaselineRunner):
    """LayoutLLM-style baseline: OCR-grounded generative document reader.

    PaddleOCR supplies the transcript (layout-ordered), which is passed as
    context together with the page image to a generative LayoutLLM checkpoint
    (loaded via AutoProcessor + an image-text-to-text auto model). Point
    ``--layout-model-id`` at your LayoutLLM weights.
    """

    def __init__(
        self,
        model_id: str,
        ocr_lang: str,
        max_new_tokens: int,
        max_context_chars: int,
        ocr_max_side: int = 3072,
    ) -> None:
        self.model_id = model_id
        self.ocr = PaddleOCREngine(lang=ocr_lang, max_side=ocr_max_side)
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.model = None
        self.processor = None
        self._ocr_cache: Dict[str, str] = {}

    def _model_loader(self) -> Any:
        import transformers

        for name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        ):
            loader = getattr(transformers, name, None)
            if loader is not None:
                logger.info(f"[layoutllm] Using transformers loader: {name}")
                return loader
        raise RuntimeError("No compatible auto-model loader found for LayoutLLM.")

    def load(self) -> None:
        from transformers import AutoProcessor

        self.ocr.load()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        loader = self._model_loader()
        try:
            self.model = loader.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LayoutLLM model '{self.model_id}'. Set --layout-model-id "
                f"to a valid generative LayoutLLM checkpoint. Original error: {exc}"
            ) from exc
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        logger.info(f"[layoutllm] Loaded model: {self.model_id}")

    def _ocr_context(
        self, images: Sequence[Image.Image], cache_key: Optional[str] = None
    ) -> str:
        if cache_key is not None and cache_key in self._ocr_cache:
            return self._ocr_cache[cache_key]
        text = "\n".join(self.ocr.read_text(img) for img in images)
        if self.max_context_chars > 0:
            text = text[: self.max_context_chars]
        if cache_key is not None:
            self._ocr_cache[cache_key] = text
        return text

    def answer(
        self,
        images: Sequence[Image.Image],
        question: str,
        cache_key: Optional[str] = None,
    ) -> str:
        context = self._ocr_context(images, cache_key)
        instruction = (
            "You are a document understanding assistant. Use the page image and its "
            "OCR text to answer with a short, exact answer.\n"
            f"OCR text:\n{context}\n\n"
            f"Question: {question}"
        )
        content: List[Dict] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return text[0].strip()


def build_runner(args: argparse.Namespace) -> BaselineRunner:
    if args.baseline == "paddleocr":
        return PaddleOCRRunner(
            ocr_lang=args.ocr_lang,
            reader_mode=args.reader_mode,
            reader_model_id=args.reader_model_id,
            max_new_tokens=args.max_new_tokens,
            max_context_chars=args.max_context_chars,
            ocr_max_side=args.ocr_max_side,
        )
    if args.baseline == "layoutllm":
        return LayoutLLMRunner(
            model_id=args.layout_model_id,
            ocr_lang=args.ocr_lang,
            max_new_tokens=args.max_new_tokens,
            max_context_chars=args.max_context_chars,
            ocr_max_side=args.ocr_max_side,
        )
    raise ValueError(f"Unsupported baseline: {args.baseline}")


# ---------------------------------------------------------------------------
# Evaluation loop (matches eval_base.py output format)
# ---------------------------------------------------------------------------
def evaluate(args: argparse.Namespace, runner: BaselineRunner) -> Tuple[Dict, List[Dict]]:
    set_reproducible(seed=args.seed, deterministic=False)

    with args.dataset.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]

    runner.load()

    results: List[Dict] = []
    f1_sum = em_sum = precision_sum = recall_sum = 0.0
    evaluated = 0
    skipped_missing_visual_input = 0
    skipped_bad_example = 0

    for idx, example in enumerate(data):
        image_name = example.get("image") or example.get("source_file")
        question = example.get("question")
        answer = example.get("answer", "")
        category = example.get("category") or example.get("attribute", "")
        category_id = example.get("category_id", "")

        if not image_name or not question:
            skipped_bad_example += 1
            continue

        try:
            images, source_meta = resolve_visual_inputs(
                example=example,
                input_mode=args.input_mode,
                images_dir=args.images_dir,
                original_pdf_dir=args.original_pdf_dir,
                split_root=args.split_root_dir,
                split_jpg_max_images=args.split_jpg_max_images,
                pdf_dpi=args.pdf_dpi,
                pdf_max_pages=args.pdf_max_pages,
            )
        except Exception as exc:
            logger.warning(f"Skipping idx={idx} due to visual input issue: {exc}")
            skipped_missing_visual_input += 1
            continue

        try:
            cache_key = source_meta.get("resolved_source") or str(image_name)
            prediction = runner.answer(images, question, cache_key=cache_key)
        except Exception as exc:
            logger.warning(f"Generation failed at idx={idx}: {exc}")
            prediction = ""

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
                "category": category,
                "category_id": category_id,
                "resolved_source": source_meta.get("resolved_source", ""),
                "input_mode": args.input_mode,
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "exact_match": round(em, 4),
            }
        )
        print(f"[{evaluated}] F1={f1:.4f} EM={em:.4f} | image={image_name} | q={question}")

    # Per-category_id aggregation (same shape as eval_base.py).
    per_cat: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cid = r.get("category_id") or "UNKNOWN"
        b = per_cat.setdefault(
            cid,
            {"category": r.get("category", ""), "count": 0,
             "f1_sum": 0.0, "em_sum": 0.0, "precision_sum": 0.0, "recall_sum": 0.0},
        )
        b["count"] += 1
        b["f1_sum"] += r["f1"]
        b["em_sum"] += r["exact_match"]
        b["precision_sum"] += r["precision"]
        b["recall_sum"] += r["recall"]

    category_id_summary: Dict[str, Dict[str, Any]] = {}
    for cid in sorted(per_cat):
        b = per_cat[cid]
        n = b["count"]
        category_id_summary[cid] = {
            "category": b["category"],
            "count": n,
            "avg_f1": round(b["f1_sum"] / n, 4) if n else 0.0,
            "avg_precision": round(b["precision_sum"] / n, 4) if n else 0.0,
            "avg_recall": round(b["recall_sum"] / n, 4) if n else 0.0,
            "avg_exact_match": round(b["em_sum"] / n, 4) if n else 0.0,
        }

    summary = {
        "dataset_path": str(args.dataset),
        "images_dir": str(args.images_dir),
        "input_mode": args.input_mode,
        "baseline": args.baseline,
        "ocr_lang": args.ocr_lang,
        "reader_mode": args.reader_mode if args.baseline == "paddleocr" else None,
        "reader_model_id": args.reader_model_id
        if (args.baseline == "paddleocr" and args.reader_mode == "llm")
        else None,
        "layout_model_id": args.layout_model_id if args.baseline == "layoutllm" else None,
        "seed": args.seed,
        "total_examples_loaded": len(data),
        "evaluated_examples": evaluated,
        "skipped_missing_visual_input": skipped_missing_visual_input,
        "skipped_bad_example": skipped_bad_example,
        "avg_f1": f1_sum / evaluated if evaluated else 0.0,
        "avg_precision": precision_sum / evaluated if evaluated else 0.0,
        "avg_recall": recall_sum / evaluated if evaluated else 0.0,
        "avg_exact_match": em_sum / evaluated if evaluated else 0.0,
        "per_category_id": category_id_summary,
    }
    return summary, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR / LayoutLLM baselines on VQA with token-level F1."
    )
    parser.add_argument("--dataset", type=Path, default=Path("Dutch/DutchVQA_gemini.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("Dutch/extracted_images of Dutch"))
    parser.add_argument(
        "--input-mode",
        type=str,
        default="image",
        choices=["image", "original_pdf", "split_pdf", "split_jpg"],
    )
    parser.add_argument("--original-pdf-dir", type=Path, default=None)
    parser.add_argument("--split-root-dir", type=Path, default=None)
    parser.add_argument("--split-jpg-max-images", type=int, default=0)
    parser.add_argument("--pdf-dpi", type=int, default=200)
    parser.add_argument("--pdf-max-pages", type=int, default=0)

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["paddleocr", "layoutllm"],
        help="Which baseline to evaluate.",
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default="",
        help="PaddleOCR recognition language. Leave empty to use the default PP-OCR model "
        "(handles Latin-script text well for Dutch/Spanish/French/English). NOTE: 'latin' is "
        "NOT a valid code on PP-OCRv6; use e.g. 'en', 'fr', 'german' if you pass one explicitly.",
    )
    # PaddleOCR reader options
    parser.add_argument(
        "--reader-mode",
        type=str,
        default="llm",
        choices=["llm", "ocr_only"],
        help="paddleocr only: 'llm' reads an answer from OCR text; 'ocr_only' returns the transcript.",
    )
    parser.add_argument(
        "--reader-model-id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="paddleocr + reader-mode llm: text LLM used to extract the answer from OCR text.",
    )
    # LayoutLLM options
    parser.add_argument(
        "--layout-model-id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="layoutllm only: HF id/path of the generative LayoutLLM checkpoint.",
    )

    parser.add_argument(
        "--ocr-max-side",
        type=int,
        default=3072,
        help="Downscale each page so its longest side <= this many pixels before OCR "
        "(0 = no resize). The raw scans are ~4096x7900; downscaling greatly speeds up OCR.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=6000,
        help="Truncate the OCR transcript to this many characters (0 = no limit).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=Path("eval_baseline.json"))
    parser.add_argument("--output-summary", type=Path, default=Path("eval_baseline_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = build_runner(args)
    summary, results = evaluate(args, runner)

    report = {"summary": summary, "results": results}
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    per_category_id = summary.get("per_category_id", {})
    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        if key == "per_category_id":
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if per_category_id:
        print("\n=== F1 by category_id ===")
        header = f"{'category_id':<12} {'n':>4} {'F1':>8} {'EM':>8} {'category'}"
        print(header)
        print("-" * (len(header) + 8))
        for cid, stats in per_category_id.items():
            print(
                f"{cid:<12} {stats['count']:>4} {stats['avg_f1']:>8.4f} "
                f"{stats['avg_exact_match']:>8.4f} {stats['category']}"
            )

    print(f"\nSaved detailed report to: {args.output_json}")
    pd.DataFrame(results).to_csv(args.output_summary, index=False)
    print(f"Saved summary to: {args.output_summary}")


if __name__ == "__main__":
    main()
