"""
Inference: turn an invoice image into structured fields.

predict_invoice() runs OCR, feeds words+boxes+image into the model, then
extract_fields() maps the token-level predictions back onto the original
words (using the tokenizer's word_ids alignment) and groups them by field.
"""

from typing import Optional

import torch
from PIL import Image

from . import config
from .ocr import mock_ocr_text, run_ocr

_LABEL_TO_FIELD = {
    "INVOICE_NUMBER": "invoice_number",
    "TOTAL_AMOUNT": "total_amount",
    "DATE": "date",
    "VENDOR_NAME": "vendor_name",
}


def predict_invoice(image_path: str, processor, model, use_real_ocr: bool = True):
    """
    Run the full pipeline on a single invoice image.

    Returns:
        fields: dict of extracted field -> value (or None)
        words: the OCR words used
        predictions: raw per-token label ids
    """
    words, boxes = run_ocr(image_path) if use_real_ocr else mock_ocr_text()

    if not words:
        return {field: None for field in _LABEL_TO_FIELD.values()}, [], []

    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze(0).tolist()

    fields = extract_fields(predictions, words, encoding)
    return fields, words, predictions


def extract_fields(predictions, words, encoding=None) -> dict:
    """
    Convert per-token predictions into structured fields.

    LayoutLMv3's tokenizer can split one OCR word into several sub-word
    tokens, so we use encoding.word_ids() to map each predicted token back
    to its original word, then take the first predicted label per word
    (a simple but effective heuristic for this token-classification setup).
    """
    grouped = {field: [] for field in _LABEL_TO_FIELD.values()}

    if encoding is not None and hasattr(encoding, "word_ids"):
        word_ids = encoding.word_ids(batch_index=0)
    else:
        # No fast-tokenizer alignment available (e.g. mock predictions) -
        # fall back to a 1:1 index mapping.
        word_ids = list(range(len(predictions)))

    seen_words = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen_words or word_idx >= len(words):
            continue
        if token_idx >= len(predictions):
            continue

        label = config.ID2LABEL.get(predictions[token_idx], "O")
        field = _LABEL_TO_FIELD.get(label)
        if field:
            grouped[field].append(words[word_idx])
            seen_words.add(word_idx)

    return {field: (" ".join(vals) if vals else None) for field, vals in grouped.items()}
