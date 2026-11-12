from torch import nn
from torch.nn.utils import spectral_norm
import torch


class LipshitzLayer(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer = spectral_norm(nn.Linear(input, output))

    def forward(self, x):
        return self.layer(x)


class GroupSort(nn.Module):
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(shape[0], x[0].numel() // self.group_size, self.group_size)
        x = x.sort(dim=2).values
        return x.reshape(shape)


class MLP(nn.Module):
    def __init__(self, layer_sizes, layer_type, activation):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(layer_type(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation())
        layers.append(layer_type(layer_sizes[-2], layer_sizes[-1]))

        self.model = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, X):
        return self.model(X)
