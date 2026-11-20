import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

from src.models import MLP, SNLinear, AOLLinear, CaleyLinear, GroupSort, CNN, AOLConv2d
from src.training import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# dataset
train_data = datasets.MNIST(root="data", train=True, download=True, 
                            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
train_data, val_data = random_split(train_data, [0.8, 0.2])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(train_data, batch_size=64, shuffle=False)

# models 1. Linear MLP | 2. L <= 1 Lipschitz MLPs with different architectures
exp3_mlp = MLP([784, 128, 128, 10], nn.Linear, nn.ReLU)
exp3_mlp.to(device)
opt = torch.optim.Adam(exp3_mlp.parameters(), lr=0.001)

exp3_cnn = CNN(10, nn.Conv2d, nn.ReLU, nn.Linear)
exp3_cnn.to(device)
opt_cnn = torch.optim.Adam(exp3_cnn.parameters(), lr=0.001)

exp3_cnn_AOL = CNN(10, AOLConv2d, GroupSort, AOLLinear)
exp3_cnn_AOL.to(device)
opt_cnn_AOL = torch.optim.Adam(exp3_cnn_AOL.parameters(), lr=0.001)

# training
epochs = 5

# train(exp3_mlp, nn.CrossEntropyLoss(), opt, train_loader, val_loader, epochs)
# torch.save(exp3_mlp.state_dict(), "trained_models/exp3_mlp.pth")

# train(exp3_cnn, nn.CrossEntropyLoss(), opt_cnn, train_loader, val_loader, epochs)
# torch.save(exp3_cnn.state_dict(), "trained_models/exp3_cnn.pth")

# train(exp3_cnn_AOL, nn.CrossEntropyLoss(), opt_cnn_AOL, train_loader, val_loader, epochs)
# torch.save(exp3_cnn_AOL.state_dict(), "trained_models/exp3_cnn_AOL.pth")