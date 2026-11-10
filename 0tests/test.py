import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


x = torch.tensor([1, 2, 3])
print(x.clamp([[0, 0.5], [3, 4], [2, 5]]))