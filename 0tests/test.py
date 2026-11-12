import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np


x = [1, 2, 3]


def prix(l):
    l[0] = 0


prix(x)
print(x)

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

z = np.concatenate(x[:, None], y[None, :])
print(z)
