"""Small shared helpers used by train.py, src/infer.py and api/app.py."""

import os

import torch

from . import config


def save_checkpoint(model, epoch, optimizer=None, path=None) -> str:
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = path or os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")

    state = {"model_state_dict": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, path)
    return path


def load_checkpoint(model, path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
