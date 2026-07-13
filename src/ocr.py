"""
OCR helpers.

`run_ocr` performs real OCR with EasyOCR and returns (words, boxes) with
boxes normalized to the 0-1000 scale that LayoutLMv3 expects.

`mock_ocr_text` returns a small hard-coded example so the rest of the
pipeline (dataset -> model -> inference) can be smoke-tested without
installing OCR dependencies or having a real scanned invoice on hand.
"""

from typing import List, Tuple

import numpy as np
from PIL import Image

_reader = None  # lazy-loaded EasyOCR reader (loading it is slow)


def get_reader():
    global _reader
    if _reader is None:
        import easyocr  # imported lazily so the module can be imported without easyocr installed

        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def run_ocr(image_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Run OCR on an invoice image.

    Returns:
        words: list of recognized word/phrase strings
        boxes: list of [x0, y0, x1, y1] boxes, normalized to 0-1000
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    reader = get_reader()
    results = reader.readtext(np.array(image))

    words: List[str] = []
    boxes: List[List[int]] = []

    for bbox, text, _confidence in results:
        text = text.strip()
        if not text:
            continue

        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        norm_box = [
            max(0, min(1000, int(1000 * x0 / width))),
            max(0, min(1000, int(1000 * y0 / height))),
            max(0, min(1000, int(1000 * x1 / width))),
            max(0, min(1000, int(1000 * y1 / height))),
        ]

        words.append(text)
        boxes.append(norm_box)

    return words, boxes


def mock_ocr_text() -> Tuple[List[str], List[List[int]]]:
    """Fallback sample OCR output for quick testing without a real image or OCR engine."""
    data = [
        ("Invoice", [50, 40, 120, 70]),
        ("No:", [130, 40, 160, 70]),
        ("INV-1092", [170, 40, 280, 70]),
        ("Date:", [50, 90, 100, 120]),
        ("12/05/2026", [110, 90, 250, 120]),
        ("ABC", [50, 140, 100, 170]),
        ("Pvt", [110, 140, 150, 170]),
        ("Ltd", [160, 140, 200, 170]),
        ("Total", [50, 300, 100, 330]),
        ("1499.00", [110, 300, 190, 330]),
    ]
    words = [w for w, _ in data]
    boxes = [b for _, b in data]
    return words, boxes
