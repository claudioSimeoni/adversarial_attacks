import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn import datasets

from src.models import MLP, SNLinear, AOLLinear, CaleyLinear, GroupSort
from src.training import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# dataset
n_samples = 10000
n_train = n_samples * 4 // 5
x, y = datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=42)

x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
x_val = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.long)
y_val = torch.tensor(y[n_train:], dtype=torch.long)

val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# models 1. Linear MLP | 2. L <= 1 Lipschitz MLPs with different architectures
exp2_mlp = MLP([2, 64, 64, 2], nn.Linear, nn.ReLU)
exp2_mlp.to(device)
opt = torch.optim.Adam(exp2_mlp.parameters(), lr=0.001)

exp2_SN_mlp = MLP([2, 64, 64, 2], SNLinear, GroupSort)
exp2_SN_mlp.to(device)
opt_SN = torch.optim.Adam(exp2_SN_mlp.parameters(), lr=0.001)

exp2_AOL_mlp = MLP([2, 64, 64, 2],  AOLLinear, GroupSort)
exp2_AOL_mlp.to(device)
opt_AOL = torch.optim.Adam(exp2_AOL_mlp.parameters(), lr=0.001)

exp2_Caley_mlp = MLP([2, 64, 64, 2], CaleyLinear, GroupSort)
exp2_Caley_mlp.to(device)
opt_Caley = torch.optim.Adam(exp2_Caley_mlp.parameters(), lr=0.001)

# training
epochs = 50

# train(exp2_mlp, nn.CrossEntropyLoss(), opt, train_loader, val_loader, epochs)
# torch.save(exp2_mlp.state_dict(), "trained_models/exp2_mlp.pth")

# train(exp2_SN_mlp, nn.CrossEntropyLoss(), opt_SN, train_loader, val_loader, epochs)
# torch.save(exp2_SN_mlp.state_dict(), "trained_models/exp2_SN_mlp.pth")

# train(exp2_AOL_mlp, nn.CrossEntropyLoss(), opt_AOL, train_loader, val_loader, epochs)
# torch.save(exp2_AOL_mlp.state_dict(), "trained_models/exp2_AOL_mlp.pth")

# train(exp2_Caley_mlp, nn.CrossEntropyLoss(), opt_Caley, train_loader, val_loader, epochs)
# torch.save(exp2_Caley_mlp.state_dict(), "trained_models/exp2_Caley_mlp.pth")
