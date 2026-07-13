"""
Generates a synthetic sample invoice image + matching data/annotations.csv
so the rest of the pipeline (tests, training, inference) can be run
end-to-end without needing a real scanned invoice.

Run once after cloning:
    python scripts/generate_sample_data.py
"""

import csv
import os

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

W, H = 600, 400

# label ids — must match src/config.py LABEL_LIST
O, INVOICE_NUMBER, TOTAL_AMOUNT, DATE, VENDOR_NAME = 0, 1, 2, 3, 4

LINES = [
    (50, 40, [("Invoice", O), ("No:", O), ("INV-1092", INVOICE_NUMBER)]),
    (50, 90, [("Date:", O), ("12/05/2026", DATE)]),
    (50, 140, [("Vendor:", O), ("ABC", VENDOR_NAME), ("Pvt", VENDOR_NAME), ("Ltd", VENDOR_NAME)]),
    (50, 300, [("Total:", O), ("1499.00", TOTAL_AMOUNT)]),
]


def _load_font(size=20):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def generate():
    os.makedirs(DATA_DIR, exist_ok=True)

    img = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(20)

    words, boxes, labels = [], [], []

    for x, y, tokens in LINES:
        cursor_x = x
        for word, label_id in tokens:
            bbox = draw.textbbox((cursor_x, y), word, font=font)
            draw.text((cursor_x, y), word, fill="black", font=font)
            norm = [
                max(0, min(1000, int(1000 * bbox[0] / W))),
                max(0, min(1000, int(1000 * bbox[1] / H))),
                max(0, min(1000, int(1000 * bbox[2] / W))),
                max(0, min(1000, int(1000 * bbox[3] / H))),
            ]
            words.append(word)
            boxes.append(norm)
            labels.append(label_id)
            cursor_x = bbox[2] + 10

    draw.line((50, 250, 550, 250), fill="black", width=1)

    image_path = os.path.join(DATA_DIR, "sample_invoice_1.png")
    img.save(image_path)

    annotations_path = os.path.join(DATA_DIR, "annotations.csv")
    with open(annotations_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "words", "boxes", "labels"])
        writer.writerow(
            [
                "data/sample_invoice_1.png",
                "|".join(words),
                str(boxes),
                str(labels),
            ]
        )

    print(f"Wrote {image_path}")
    print(f"Wrote {annotations_path}")


if __name__ == "__main__":
    generate()
