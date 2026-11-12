import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
) -> list[float]:

    device = next(model.parameters()).device
    train_loss = 0
    for ep in range(1, epochs + 1):
        # training
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            train_loss += loss.item()
            loss.backward()
            opt.step()

        # validation
        model.eval()
        val_loss, corr = 0, 0
        total = 0
        val_losses = []
        val_acc = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item()
                pred = logits.argmax(dim=1)
                corr += (pred == y).sum().item()
                total += y.numel()

        val_losses.append(val_loss / len(val_loader))
        val_acc.append(corr / total)

        print(f"Epoch {ep}: Avg Training Loss = {train_loss / len(val_loader)} | Avg Validation Loss = {val_loss / len(val_loader)} | Accuracy = {corr / total}")

    return val_losses, val_acc
