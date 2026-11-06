import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from core.attacks import Attack, FSGM, IFSGM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, X):
        return self.model(X)


train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

#################################################################################

model = MLP()
state = torch.load("../model.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()


def evaluate(model: MLP, attacks: list[Attack], loader: DataLoader):
    loss_fn = nn.CrossEntropyLoss()
    for att in attacks:
        X, y = next(iter(loader))
        X = X.to(device)
        y = y.to(device)
        pred_cl, pred_adv = att.attack(model, loss_fn, X, y)

        total = len(y)
        corr = (pred_cl == y).sum()
        print(pred_cl, pred_adv)
        adv_succ = torch.logical_and((pred_cl == y), (pred_cl != pred_adv)).sum()

        print(
            f"Attack: {type(att)} | Correct predictions: {corr / total * 100}% | Successfull attacks: {adv_succ / corr * 100}%"
        )


loader = DataLoader(test_data, batch_size=100, shuffle=True)
evaluate(model, [FSGM(0.05)], loader)
