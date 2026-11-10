import torch
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

from src.models import MLP, LipshitzLayer
from src.training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x, y = datasets.make_moons(n_samples=2000, noise=0.1, random_state=42)

fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:, 1])

plt.show()
quit()
x_test = torch.tensor(x[:1600, :], dtype=torch.float32)
x_train = torch.tensor(x[1600:, :], dtype=torch.float32)
y_test = torch.tensor(y[:1600], dtype=torch.long)
y_train = torch.tensor(y[1600:], dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


model = MLP([[784, 128]], [128, 10])
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr = 0.01)
epochs = 1
train(model, nn.CrossEntropyLoss(), opt, train_loader, test_loader, epochs)

torch.save(model.state_dict(), "trained_models/mlp.pth")