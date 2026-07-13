"""
FastAPI service exposing the invoice extraction pipeline.

Run locally:
    uvicorn api.app:app --reload --port 8000

Then:
    curl -X POST "http://localhost:8000/extract" -F "file=@data/sample_invoice_1.png"

If CHECKPOINT_PATH is not set (or the file doesn't exist), the API serves
predictions from the base pretrained LayoutLMv3 weights, which is enough
to prove the pipeline runs end-to-end. For real field extraction quality,
train first (python train.py) and point CHECKPOINT_PATH at the resulting
checkpoint.
"""

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import LayoutLMv3Processor

from src import config
from src.infer import predict_invoice
from src.model import InvoiceExtractor
from src.utils import load_checkpoint

app = FastAPI(
    title="Invoice Intelligence API",
    description="OCR + LayoutLMv3 powered invoice field extraction.",
    version="1.0.0",
)

_processor = LayoutLMv3Processor.from_pretrained(config.MODEL_NAME, apply_ocr=False)
_model = InvoiceExtractor(num_labels=config.NUM_LABELS)

_checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
_model_loaded_from_checkpoint = False
if _checkpoint_path and os.path.exists(_checkpoint_path):
    _model = load_checkpoint(_model, _checkpoint_path)
    _model_loaded_from_checkpoint = True
_model.eval()

_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}


@app.get("/")
async def root():
    return {
        "service": "Invoice Intelligence API",
        "using_fine_tuned_checkpoint": _model_loaded_from_checkpoint,
        "try_it": "Open /docs for an interactive upload form, or POST an image to /extract.",
        "note": (
            None
            if _model_loaded_from_checkpoint
            else (
                "Running the base pretrained LayoutLMv3 weights (not fine-tuned on invoices). "
                "The pipeline works end-to-end, but field values won't be reliably accurate "
                "until a checkpoint from train.py is provided via CHECKPOINT_PATH."
            )
        ),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.MODEL_NAME,
        "using_fine_tuned_checkpoint": _model_loaded_from_checkpoint,
    }


@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a PNG or JPEG image.")

    contents = await file.read()

    suffix = ".png" if file.content_type == "image/png" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        fields, words, _predictions = predict_invoice(tmp_path, _processor, _model)
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    finally:
        os.remove(tmp_path)

    response = {"fields": fields, "ocr_word_count": len(words)}
    if not _model_loaded_from_checkpoint:
        response["note"] = (
            "Base pretrained weights only (not fine-tuned) — field values are not yet reliable. "
            "See README for how to train."
        )
    return response
