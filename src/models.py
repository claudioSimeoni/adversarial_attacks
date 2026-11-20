from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch


class SNLinear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer = spectral_norm(nn.Linear(input, output))

    def forward(self, x):
        return self.layer(x)


class AOLLinear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(output, input))
        self.bias = nn.Parameter(torch.zeros(output))
        nn.init.orthogonal_(self.P)

    def forward(self, x):
        G = self.P.T @ self.P
        d = torch.rsqrt(G.abs().sum(dim=0).clamp_min(1e-12))
        W = self.P * d

        return F.linear(x, W, self.bias)


class CaleyLinear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(output, input))
        self.bias = nn.Parameter(torch.zeros(output))
        nn.init.orthogonal_(self.P)

    def forward(self, x):
        P = self.P

        vertical = P.shape[0] > P.shape[1]
        if(not vertical):
            P = P.T

        n = P.shape[1]
        Y = P[:n, :]
        Z = P[n:, :]
        M = Y - Y.T + (Z.T @ Z)
        I = torch.eye(n, device=P.device, dtype=P.dtype)

        inv = (I + M).inverse()
        caley = torch.cat((inv @ (I - M), 2 * Z @ inv), 0)

        if(not vertical):
            caley = caley.T

        return F.linear(x, caley, self.bias)


class GroupSort(nn.Module):
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], x[0].numel() // self.group_size, self.group_size)
        x = x.sort(dim=2).values
        return x.reshape(shape)


class HingeLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, logits : torch.Tensor, label):
        corr_logits = logits.gather(1, label.unsqueeze(1)).squeeze(1)
        remain_max = logits.scatter(1, label.unsqueeze(1), float("-inf"))
        remain_max = remain_max.max(dim=1).values

        return F.relu(self.margin - (corr_logits - remain_max)).mean()


class MLP(nn.Module):
    def __init__(self, layer_sizes, layer_type, activation, name=None):
        super().__init__()
        self.name = name

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(layer_type(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation())
        layers.append(layer_type(layer_sizes[-2], layer_sizes[-1]))

        self.model = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, num_classes, conv, activation, linear, name=None):
        super().__init__()
        self.name = name
        self.model = nn.Sequential(
            conv(1, 64, 3, padding=1),
            activation(),
            nn.MaxPool2d(2),
            conv(64, 64, 3, padding=1),
            activation(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            linear(64 * 7 * 7, 128),
            activation(),
            linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class AOLConv2d(nn.Conv2d):
    def constraint(self, P : torch.Tensor):
        K = P.permute(1, 0, 2, 3)

        d = F.conv2d(K, K, padding=(P.shape[2] - 1, P.shape[3] - 1))
        d = d.abs().sum(dim=(0, 2, 3))
        d = d.clamp_min(1e-12).rsqrt()

        return P * d[None, :, None, None]

    def forward(self, x):
        W = self.constraint(self.weight)
        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)

# out in k0 k1