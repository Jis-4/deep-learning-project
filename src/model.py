"""
Model definition: a thin wrapper around LayoutLMv3ForTokenClassification.

Kept as its own nn.Module (rather than using the HF model directly) so
it's easy to swap in a different backbone later, or add extra heads,
without touching train.py / api/app.py.
"""

import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification

from . import config


class InvoiceExtractor(nn.Module):
    def __init__(self, num_labels: int = config.NUM_LABELS):
        super().__init__()
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=num_labels,
            id2label=config.ID2LABEL,
            label2id=config.LABEL2ID,
        )

    def forward(self, **inputs):
        return self.model(**inputs)
