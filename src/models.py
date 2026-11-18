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
        part = G.abs().sum(dim=0)
        d = 1 / torch.sqrt(part)
        W = self.P * d

        return F.linear(x, W, self.bias)


class CaleyLinear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(output, input))
        self.bias = nn.Parameter(torch.zeros(output))
        nn.init.orthogonal_(self.P)
        self.vertical = output >= input

    def forward(self, x):
        output, input = self.P.shape
        P = self.P

        if(not self.vertical):
            P = P.T
            output, input = input, output
            
        Y = P[:input, :]
        Z = P[input:, :]
        M = Y - Y.T + (Z.T @ Z)
        eye = torch.eye(input).to(x.device)

        caley = torch.zeros_like(P)

        inv = (eye + M).inverse()
        caley[:input, :] =  inv @ (eye - M)
        caley[input:, :] = 2 * Z @ inv

        if(not self.vertical):
            caley = caley.T
            self.P = nn.Parameter(P.T)
        else:
            self.P = nn.Parameter(P)

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
        top2 = logits.topk(k=2, dim=1, sorted=True).values
        col0 = torch.zeros_like(top2[:, 0])
        colm = torch.ones_like(top2[:, 0]) * self.margin
        loss = torch.max(col0, colm - (torch.abs(top2[:, 1] - top2[:, 0])))
        return loss.mean()



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
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)