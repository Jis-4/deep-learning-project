"""
Training entry point.

Usage:
    python train.py

Reads data/annotations.csv, fine-tunes LayoutLMv3ForTokenClassification on
the invoice field-extraction task, and writes a checkpoint per epoch to
checkpoints/.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor

from src import config
from src.dataset import InvoiceDataset
from src.model import InvoiceExtractor
from src.utils import save_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # apply_ocr=False because we supply our own words/boxes (from src/ocr.py or the CSV)
    processor = LayoutLMv3Processor.from_pretrained(config.MODEL_NAME, apply_ocr=False)
    model = InvoiceExtractor(num_labels=config.NUM_LABELS).to(device)

    train_dataset = InvoiceDataset(config.ANNOTATIONS_FILE, processor)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")

        ckpt_path = save_checkpoint(model, epoch + 1, optimizer)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
