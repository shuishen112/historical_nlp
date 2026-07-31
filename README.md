# Historical Newspaper VQA

Code for building and evaluating vision–language models (VLMs) on **visual question
answering (VQA) over historical newspaper scans**, across three languages
(**Dutch**, **English-French**, **Spanish**). The repository covers the full
pipeline: data collection, automatic QA generation, base-model and baseline
evaluation, parameter-efficient (LoRA) supervised fine-tuning, and reinforcement
learning (GRPO / PPO). It also includes a small, unit-tested reference package
(`historian_llm/`) implementing the core equations, and a temporal word-embedding
model (TWEC).

---

## 1. Repository layout

```
.
├── download_mediastream_pdfs.py     # collect raw newspaper PDFs
├── extract_qa_gemini.py             # generate QA pairs with Gemini (10 categories)
├── eval_base.py                     # evaluate base VLMs (Qwen3-VL, LLaVA)
├── eval_baselines.py                # OCR baselines: PaddleOCR (+reader), LayoutLLM
├── eval_qwen3_vl_f1.py              # older single-model Qwen3-VL eval (kept for reference)
├── peft_dutch_vqa_train_eval.py     # supervised LoRA fine-tuning + test-split eval
├── rl_lora_dutch_vqa.py             # RL (REINFORCE / PPO / GRPO) LoRA fine-tuning
├── historian_llm/                   # unit-tested reference implementation (see analysis.md)
├── tests/                           # pytest suite for historian_llm
├── twec.py                          # Temporal Word Embeddings with Compass
├── run_eval_base.sh                 # eval base VLMs on the three languages
├── run_eval_baselines.sh            # eval PaddleOCR / LayoutLLM baselines
├── run_eval_rl.sh                   # eval an RL LoRA adapter on the three languages
├── train_peft.sh                    # launch supervised LoRA training
├── train_rl_peft.sh / train_ppo_peft.sh   # launch RL (GRPO / PPO) training
├── requirements.txt                 # Python dependencies
├── analysis.md / requirements.md    # methodology + reference-package spec
├── Dutch/ · English-French/ · Spanish/     # per-language data + generated QA
└── Racialized_Motherhood/           # PDF/column-split case study data
```

Each language folder contains the generated QA file (e.g.
`Dutch/DutchVQA_gemini.json`) and an image directory
(e.g. `Dutch/extracted_images of Dutch/`).

---

## 2. Installation

The code was developed and tested with the conda environment **`mttl_upgrade`**
(`transformers 5.13`, `torch 2.8`, CUDA 12.8). Any environment works as long as:

- **`transformers >= 4.57`** (required for Qwen3-VL support), and
- optional extras are installed for the features you use.

```bash
pip install -r requirements.txt

# Extra deps, only if you run the corresponding component:
pip install paddleocr paddlepaddle        # OCR baselines (paddlepaddle-gpu for GPU OCR)
pip install google-genai                  # Gemini QA generation
pip install peft                          # LoRA fine-tuning / adapter eval
```

> The default `paddlepaddle` wheel is CPU-only; OCR then runs on CPU. Install
> `paddlepaddle-gpu` (matching your CUDA) if you want GPU-accelerated OCR.

---

## 3. Data collection

Download raw newspaper PDFs from the Mediastream source.

```bash
# Preview one year (no download)
python3 download_mediastream_pdfs.py --years 1816 --dry-run

# Download one / several years
python3 download_mediastream_pdfs.py --years 1816
python3 download_mediastream_pdfs.py --years 1816,1820-1825
```

Options: `--paper-id` (defaults to the URL's ID), `--output-dir` (default
`mediastream_pdfs`), `--delay 0.5` (pause between requests), `--no-skip-existing`
(re-download existing files).

---

## 4. QA generation

Generate QA pairs from newspaper images/PDFs with Gemini. Each source page yields
one Q&A pair per category, across **10 categories**: Text Extraction (TE), Named
Entity Extraction (NE), Temporal Information (TI), Monetary/Quantitative (MQ),
Occupation & Labor (OL), Person Description (PD), Location Extraction (LE),
Advertisement Type (AT), Layout & Document Structure (LD), Historical Reasoning (HR).

```bash
export GEMINI_API_KEY="<your-key>"        # do NOT commit your key

# One dataset / all datasets
python extract_qa_gemini.py --dataset Dutch
python extract_qa_gemini.py --all

# Quick test run
python extract_qa_gemini.py --dataset Dutch --max-images 20 --model gemini-3.5-flash
```

Output is written to `<Language>/<Language>VQA_gemini.json`. Each record contains
`dataset`, `source_file`, `image`, `category_id`, `category`, `question`, `answer`.

---

## 5. Evaluation

All evaluators report **token-level F1**, **exact match (EM)**, precision and
recall, an overall summary, and a **per-`category_id` breakdown**. Detailed
per-example predictions are saved to JSON and a CSV summary.

### 5.1 Base VLMs (Qwen3-VL, LLaVA)

`eval_base.py` has a pluggable model-handler interface (`--model-type`).

```bash
python eval_base.py \
  --dataset "Dutch/DutchVQA_gemini.json" \
  --images-dir "Dutch/extracted_images of Dutch" \
  --input-mode image \
  --model-type qwen3_vl \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --output-json eval_base_qwen3_dutch_vqa.json \
  --output-summary eval_base_qwen3_dutch_vqa_summary.csv

# Or run all three languages:
bash run_eval_base.sh
```

Key flags: `--model-type {auto,qwen3_vl,llava}`, `--lora-adapter <path>` (evaluate a
LoRA/RL adapter), `--limit N` (first N examples), `--input-mode
{image,original_pdf,split_pdf,split_jpg}`.

### 5.2 OCR baselines (PaddleOCR, LayoutLLM)

`eval_baselines.py` reuses the same metrics/data loading.

- **PaddleOCR**: OCRs the page, then a text LLM "reader" extracts a short answer
  (`--reader-mode llm`). Use `--reader-mode ocr_only` for the raw-transcript lower
  bound.
- **LayoutLLM**: OCR-grounded — the transcript plus the page image are fed to a
  generative LayoutLLM checkpoint (`--layout-model-id`).

```bash
# PaddleOCR + reader (single language)
python eval_baselines.py \
  --baseline paddleocr \
  --dataset "Dutch/DutchVQA_gemini.json" \
  --images-dir "Dutch/extracted_images of Dutch" \
  --input-mode image \
  --reader-mode llm --reader-model-id Qwen/Qwen2.5-3B-Instruct \
  --output-json eval_paddleocr_dutch_vqa.json \
  --output-summary eval_paddleocr_dutch_vqa_summary.csv

# All three languages x both baselines:
bash run_eval_baselines.sh
```

Notes:
- Newspaper scans are large (~30 MP); `--ocr-max-side` (default 3072) downscales
  before OCR, and OCR results are cached per page.
- OCR language: leave `--ocr-lang` empty to use the default PP-OCR model (handles
  Latin-script text). `latin` is **not** a valid code on PP-OCRv6; pass `en`, `fr`,
  `german`, `es`, etc. if you want a language-specific model.

### 5.3 RL adapter evaluation

```bash
bash run_eval_rl.sh    # evaluates outputs/rl_lora_dutch_vqa_run1 on all three languages
```

---

## 6. Training

Backbone: `Qwen/Qwen3-VL-2B-Instruct`. Adapters: LoRA (`r=16`, `alpha=32`,
`dropout=0.05`) on the attention projections (`q/k/v/o`).

### 6.1 Supervised LoRA fine-tuning

```bash
python peft_dutch_vqa_train_eval.py \
  --dataset "Spanish/SpanishVQA_gemini.json" \
  --images-dir "Spanish/extracted_images of Spanish" \
  --output-dir "outputs/peft_spanish_vqa_run1" \
  --test-ratio 0.2 --train-max-samples 200 --test-max-samples 80 \
  --epochs 2 --batch-size 1 --grad-accum 8

# or: bash train_peft.sh
```

### 6.2 Reinforcement learning (GRPO / PPO)

Sequence-level reward `R_auto = clip(1 − CER, 0, 1)` (optionally blended with token
F1, and an optional learned Historian Reward Model). Clipped surrogate objective.

```bash
bash train_rl_peft.sh     # GRPO
bash train_ppo_peft.sh    # PPO
```

Both call `rl_lora_dutch_vqa.py`; see the script for the full hyperparameter set
(`--algo`, `--grpo-group-size`, `--ppo-clip-eps`, `--reward-mode`, `--f1-weight`,
`--baseline`, etc.).

---

## 7. Reference package + tests (`historian_llm/`)

A small, CPU-only, unit-tested implementation of the core equations in
`analysis.md` (visual cross-attention, LoRA, the reward stack, and PPO/GRPO
objectives). See `requirements.md` for the full specification.

```bash
python -m pytest tests/ -q
python -c "import historian_llm"
```

---

## 8. TWEC — Temporal Word Embeddings with Compass

`twec.py` implements temporal word embeddings that use a shared "compass" to align
embeddings across time periods, for diachronic analysis of the newspaper corpus.

---

## 9. Notes

- **Do not commit API keys.** Set `GEMINI_API_KEY` via your shell environment
  rather than hardcoding it in any script.
- Qwen3-VL requires `transformers >= 4.57`.
- GPU is strongly recommended for model inference/training; the OCR step runs on
  CPU unless `paddlepaddle-gpu` is installed.

