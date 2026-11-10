import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.attacks import FSGM
from src.evaluation import evaluate
from src.models import MLP
from src.training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# TODO: here we might want to divide the training data into training and validation
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

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = MLP([784, 128, 10], nn.Linear, nn.ReLU)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr = 0.01)
epochs = 1
train(model, nn.CrossEntropyLoss(), opt, train_loader, test_loader, epochs)

torch.save(model.state_dict(), "trained_models/mlp.pth")