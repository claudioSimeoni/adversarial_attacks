import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.attacks import FSGM, IFSGM, PGD
from src.evaluation import evaluate
from src.models import MLP
from src.training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MLP([784, 128, 10], nn.Linear, nn.ReLU)
state = torch.load("trained_models/model.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
evaluate(model, [FSGM(0.05), IFSGM(0.05, 10), PGD(0.05, 10)], test_loader)
