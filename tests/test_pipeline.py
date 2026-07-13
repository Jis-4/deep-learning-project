"""
Lightweight tests that don't require torch/transformers to be installed,
so they can run anywhere (including CI without GPU/model downloads).

Run with: python -m pytest tests/ -v
"""

import ast
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.infer import extract_fields  # noqa: E402


def test_annotations_csv_is_well_formed():
    df = pd.read_csv(config.ANNOTATIONS_FILE)
    row = df.iloc[0]

    words = row["words"].split("|")
    boxes = ast.literal_eval(row["boxes"])
    labels = ast.literal_eval(row["labels"])

    assert len(words) == len(boxes) == len(labels)
    assert all(0 <= label_id < config.NUM_LABELS for label_id in labels)
    assert os.path.exists(os.path.join(config.BASE_DIR, row["image_path"]))


def test_extract_fields_groups_words_by_label():
    words = ["Invoice", "No:", "INV-1092", "Total:", "1499.00"]
    # predictions aligned 1:1 with words (as if encoding.word_ids() were identity)
    predictions = [
        config.LABEL2ID["O"],
        config.LABEL2ID["O"],
        config.LABEL2ID["INVOICE_NUMBER"],
        config.LABEL2ID["O"],
        config.LABEL2ID["TOTAL_AMOUNT"],
    ]

    fields = extract_fields(predictions, words, encoding=None)

    assert fields["invoice_number"] == "INV-1092"
    assert fields["total_amount"] == "1499.00"
    assert fields["date"] is None
    assert fields["vendor_name"] is None


if __name__ == "__main__":
    test_annotations_csv_is_well_formed()
    test_extract_fields_groups_words_by_label()
    print("All lightweight tests passed.")
