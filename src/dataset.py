"""
PyTorch Dataset for training LayoutLMv3 on invoice token classification.

Expects a CSV (see data/annotations.csv) with columns:
    image_path : path to the invoice image, relative to the repo root
    words      : OCR words, pipe-separated, e.g. "Invoice|No:|INV-1092"
    boxes      : stringified list of [x0, y0, x1, y1] boxes, 0-1000 scale
                 e.g. "[[50,40,120,70],[130,40,160,70]]"
    labels     : stringified list of label ids (see src/config.py LABEL2ID),
                 aligned 1:1 with the words column
                 e.g. "[0,0,1]"
"""

import ast
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from . import config


class InvoiceDataset(Dataset):
    def __init__(self, csv_path=None, processor=None, max_length=config.MAX_LENGTH, base_dir=None):
        self.csv_path = csv_path or config.ANNOTATIONS_FILE
        self.df = pd.read_csv(self.csv_path)
        self.processor = processor
        self.max_length = max_length
        self.base_dir = base_dir or config.BASE_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.base_dir, image_path)
        image = Image.open(image_path).convert("RGB")

        words = row["words"].split("|")
        boxes = ast.literal_eval(row["boxes"])
        labels = ast.literal_eval(row["labels"])

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}
