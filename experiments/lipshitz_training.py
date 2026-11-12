import torch
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

from src.models import MLP, LipshitzLayerSN, LipshitzLayerAOL, GroupSort
from src.training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

n_samples = 10000
n_train = n_samples * 4 // 5
x, y = datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=42)

x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
x_test = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.long)
y_test = torch.tensor(y[n_train:], dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


model = MLP([2, 64, 64, 2], nn.Linear, nn.ReLU)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

lipshitz_model = MLP([2, 64, 64, 2], LipshitzLayerSN, GroupSort)
lipshitz_model.to(device)
opt_lip = torch.optim.Adam(lipshitz_model.parameters(), lr=0.01)

lipshitzAOL_model = MLP([2, 64, 64, 2], LipshitzLayerAOL, GroupSort)
lipshitzAOL_model.to(device)
opt_lipAOL = torch.optim.Adam(lipshitzAOL_model.parameters(), lr=0.01)

epochs = 5

train(model, nn.CrossEntropyLoss(), opt, train_loader, test_loader, epochs)
train(lipshitz_model, nn.CrossEntropyLoss(), opt_lip, train_loader, test_loader, epochs)
train(lipshitzAOL_model, nn.CrossEntropyLoss(), opt_lipAOL, train_loader, test_loader, epochs)

torch.save(model.state_dict(), "trained_models/mlp_moon.pth")
torch.save(lipshitz_model.state_dict(), "trained_models/lipshitz_mlp_moon.pth")
torch.save(lipshitzAOL_model.state_dict(), "trained_models/lipshitzAOL_mlp_moon.pth")
