"""Train the digit CNN on MNIST with heavy augmentation (so it copes with shaky air-drawn digits).

Usage:  python train.py --epochs 10
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN, MEAN, STD


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += loss_fn(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=8),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

    train_ds = datasets.MNIST("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    model = DigitCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl)
    )

    hist = {"train_loss": [], "test_loss": [], "test_acc": []}
    best = 0.0
    Path("models").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        tl, ta = evaluate(model, test_dl, device, loss_fn)
        hist["train_loss"].append(run_loss / n)
        hist["test_loss"].append(tl)
        hist["test_acc"].append(ta)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {run_loss / n:.4f} | "
              f"test loss {tl:.4f} | test acc {ta * 100:.2f}% | {time.time() - t0:.1f}s")
        if ta > best:
            best = ta
            torch.save(model.state_dict(), "models/digit_cnn.pt")

    print(f"Best test accuracy: {best * 100:.2f}%  -> saved to models/digit_cnn.pt")

    # Training curves for the README
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["test_loss"], label="test")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot([a * 100 for a in hist["test_acc"]], color="green")
    ax[1].set_title("Test accuracy (%)"); ax[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig("assets/training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
