import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np


x = torch.tensor([[1, 2, 3], [3, 4, 5]])
y = x.T
z = x
k = x[:, [0, 2]]
z[0, 0] = 3
y[0, 1] = 17
prixx = [1, 3, 4]
brux = prixx

k[0, 0] = 1234123

print(y, x.T, k)

