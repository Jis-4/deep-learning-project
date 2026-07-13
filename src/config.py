"""
Central configuration for the Invoice Intelligence System.

Keeping every tunable value in one place means train.py, api/app.py and
src/infer.py all agree on the same label set, model checkpoint format,
and hyperparameters.
"""

import os

# ---------------------------------------------------------------------------
# Label schema (token classification labels used by LayoutLMv3)
# ---------------------------------------------------------------------------
LABEL_LIST = ["O", "INVOICE_NUMBER", "TOTAL_AMOUNT", "DATE", "VENDOR_NAME"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-5
EPOCHS = 3
BATCH_SIZE = 2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
