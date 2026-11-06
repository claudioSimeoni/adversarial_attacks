import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np


def train(model : nn.Module, loss_fn, optimizer, loader, epochs):
    device = next(model.parameters()).device
    model.train()
    loss_list = []

    for _ in range(epochs):
        epoch_loss = 0

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            # single batch
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(epoch_loss / len(loader))

    return loss_list
