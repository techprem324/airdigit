"""The digit-recognition CNN, plus helpers to save and load trained weights."""
from pathlib import Path

import torch
import torch.nn as nn

from preprocess import MEAN, STD  # noqa: F401  (re-exported for train.py)


class DigitCNN(nn.Module):
    """Four 3x3 conv layers (with batch norm), two poolings and a small dense head.

    `width` scales every layer. width=1 is ~470k parameters and is plenty for
    MNIST; width=2 or 3 is slower to train but can squeeze out a little more
    accuracy.
    """

    def __init__(self, num_classes: int = 10, width: int = 1):
        super().__init__()
        c1, c2, hidden = 32 * width, 64 * width, 128 * width
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, hidden), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def save_checkpoint(path, model: DigitCNN, width: int, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {"width": width}, "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint written by train.py. Returns (model in eval mode, meta dict).

    Plain state_dict files from older versions of the project also load, as width 1.
    """
    blob = torch.load(path, map_location=device, weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        width = int(blob.get("config", {}).get("width", 1))
        state, meta = blob["state_dict"], blob.get("meta", {})
    else:
        width, state, meta = 1, blob, {}
    model = DigitCNN(width=width).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, meta
