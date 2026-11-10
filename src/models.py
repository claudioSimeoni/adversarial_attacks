from torch import nn
from torch.nn.utils import spectral_norm


class LipshitzLayer(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer = nn.Linear(input, output)

    def forward(self, X):
        return self.layer(X) / spectral_norm(self.layer)


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