---
title: Invoice Intelligence Demo
emoji: 🧾
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Invoice Intelligence System

OCR + layout-aware transformer (LayoutLMv3) pipeline that turns invoice
images into structured JSON — invoice number, date, vendor name, and
total amount. Ships two front ends: a FastAPI JSON service (`api/app.py`)
and a Gradio demo UI (`app.py`) for Hugging Face Spaces.

## Live demo on Hugging Face Spaces

This repo doubles as a Hugging Face Space. The YAML block at the very
top of this file is what Spaces reads to configure the build —
`sdk: gradio` + `app_file: app.py` means it runs `app.py` directly with
no Docker build step, which works on the free **CPU Basic** tier (and
also on **ZeroGPU** if that's what your account defaults to — this app
is CPU-light enough that either works fine).

> Note: Hugging Face's free-tier SDK availability has been in flux lately
> (Docker now requires a paid plan; some accounts are seeing Gradio
> restricted to ZeroGPU-only). If Gradio Spaces aren't available on your
> account either, ZeroGPU is still free — it just has a daily GPU-minute
> quota, which is irrelevant here since this app barely touches the GPU.

**Important caveat about the deployed demo:** it runs the *base pretrained*
LayoutLMv3 weights, not a fine-tuned one. The token-classification head
(the part that decides "this word is an invoice number" vs "this word is
a total") is randomly initialized until you train it — so the deployed
demo currently proves the pipeline runs end-to-end (OCR → layout model →
structured JSON), but the actual field values it returns won't be
reliably correct yet. The UI shows a note saying so. See
[Training](#training) below to fix that.

### Deploying this repo to a Space

1. Create a new Space at https://huggingface.co/new-space with **Gradio**
   as the SDK (any name, e.g. `invoice-intelligence-demo`).
2. Locally:
   ```bash
   git clone https://github.com/Jis-4/deep-learning-project.git
   cd deep-learning-project
   git remote add space https://huggingface.co/spaces/<your-username>/invoice-intelligence-demo
   git push space main
   ```
3. The Space installs `requirements.txt` and runs `app.py`. First load
   takes a few minutes (downloading the ~500MB LayoutLMv3 weights +
   EasyOCR models). Once it's up:
   `https://huggingface.co/spaces/<your-username>/invoice-intelligence-demo`
4. Upload an invoice image in the UI and click "Extract fields".

The FastAPI service (`api/app.py`) still works the same way locally or
in Docker if you have Docker/a paid Space tier available — see
[Running the API](#running-the-api) below.

```
invoice image  →  OCR (EasyOCR)  →  words + bounding boxes
               →  LayoutLMv3 token classification
               →  field extraction  →  JSON
```

## Why LayoutLMv3 instead of plain OCR + regex

OCR alone gives you text, not meaning. It doesn't know that the number
near the bottom-right is the total, or that a date at the top is the
invoice date rather than a due date. LayoutLMv3 is a layout-aware
transformer: it sees the text *and* where each word sits on the page, so
it can learn that "total" tends to sit near the bottom right, "invoice
number" near the header, and so on.

## Project structure

```
invoice-intelligence/
├── app.py                       # Gradio demo UI (Hugging Face Spaces entry point)
├── data/
│   ├── sample_invoice_1.png   # synthetic sample invoice for smoke-testing
│   └── annotations.csv        # matching training annotations
├── src/
│   ├── config.py               # label schema + hyperparameters
│   ├── ocr.py                  # EasyOCR wrapper (+ a mock for quick tests)
│   ├── dataset.py               # PyTorch Dataset reading annotations.csv
│   ├── model.py                 # LayoutLMv3ForTokenClassification wrapper
│   ├── utils.py                 # checkpoint save/load
│   └── infer.py                 # inference + field-extraction logic
├── api/
│   └── app.py                   # FastAPI service (/health, /extract)
├── tests/
│   └── test_pipeline.py         # fast tests that don't need a GPU or model download
├── train.py                     # training entry point
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick sanity check (no model download required)

```bash
python -m pytest tests/ -v
```

This checks that `data/annotations.csv` is well-formed and that the
field-extraction grouping logic works, without needing to download the
~500MB LayoutLMv3 weights.

## Training

The repo ships with one synthetic labeled example
(`data/sample_invoice_1.png` + `data/annotations.csv`) purely so the
pipeline is runnable end-to-end out of the box. For a model that's
actually useful, add real invoices:

1. Drop invoice images into `data/`.
2. For each image, run OCR (`src/ocr.run_ocr`) to get words + boxes, then
   hand-label each word with one of: `O`, `INVOICE_NUMBER`,
   `TOTAL_AMOUNT`, `DATE`, `VENDOR_NAME` (see `src/config.py`).
3. Append a row to `data/annotations.csv` with the image path, pipe-joined
   words, the box list, and the label-id list.
4. Aim for at least a few dozen labeled invoices before training — one
   example will run but won't generalize.

Then train:

```bash
python train.py
```

Checkpoints are written to `checkpoints/model_epoch_N.pt` after every
epoch.

## Running the API

```bash
export CHECKPOINT_PATH=checkpoints/model_epoch_3.pt   # optional, once you've trained
uvicorn api.app:app --reload --port 8000
```

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@data/sample_invoice_1.png"
```

```json
{
  "fields": {
    "invoice_number": "INV-1092",
    "date": "12/05/2026",
    "vendor_name": "ABC Pvt Ltd",
    "total_amount": "1499.00"
  },
  "ocr_word_count": 11
}
```

If `CHECKPOINT_PATH` isn't set, the API still runs — it serves
predictions from the base pretrained LayoutLMv3 weights, so you can
confirm the pipeline works before you've trained anything. Extraction
quality is only meaningful after fine-tuning on labeled invoices.

## Docker

```bash
docker build -t invoice-intelligence .
docker run -p 8000:8000 invoice-intelligence
```

## Notes on "real-time"

The `/extract` endpoint runs OCR + a forward pass through the model per
request — typically a few hundred milliseconds to a couple of seconds on
CPU depending on image size, faster on GPU. For higher throughput,
batch requests or move the model to GPU (`device_map="cuda"`).

## Resume line

Built an invoice intelligence system using OCR and LayoutLMv3 to extract
key fields such as invoice number, date, vendor, and total amount, and
deployed it as a FastAPI service.
