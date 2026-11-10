import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from src.attacks import Attack, FSGM


def evaluate(model: nn.Module, attacks: list[Attack], loader: DataLoader):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()

    for att in attacks:
        x, y = next(iter(loader))
        x = x.to(device)
        y = y.to(device)
        pred, pred_adv = att.attack(model, loss_fn, x, y)

        total = y.numel()
        corr = (pred == y).sum().item()
        adv_succ = torch.logical_and((pred == y), (pred != pred_adv)).sum()

        print(
            f"Attack: {type(att)} | Correct predictions: {corr / total * 100}% | Successfull attacks: {adv_succ / corr * 100}%"
        )


