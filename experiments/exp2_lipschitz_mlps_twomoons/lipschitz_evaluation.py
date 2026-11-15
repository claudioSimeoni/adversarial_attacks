import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from sklearn import datasets

from src.attacks import FSGM, IFSGM, PGD
from src.evaluation import evaluate, certify_l2
from src.models import MLP, SNLinear, AOLLinear, GroupSort
from src.utils import load_models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# plotting utilities
def show_boundary(model: MLP, lb, ub, ax):
    xgrid = np.linspace(lb, ub, 100)
    ygrid = np.linspace(lb, ub, 100)

    grid = [[i, j] for i in xgrid for j in ygrid]
    grid = torch.tensor(grid, dtype=torch.float32).to(device)
    pred = torch.argmax(model(grid), dim=1)

    xplt = grid[:, 0].cpu().numpy().reshape(len(xgrid), len(ygrid))
    yplt = grid[:, 1].cpu().numpy().reshape(len(xgrid), len(ygrid))
    zplt = pred.cpu().numpy().reshape(len(xgrid), len(ygrid))

    ax.contourf(xplt, yplt, zplt, levels=[-0.5, 0.5, 1.5], cmap="magma", alpha=0.6)


def show_lipshitz_constraints(model: MLP, pts, ax, L):
    pred = model(pts)
    radii = torch.abs(pred[:, 0] - pred[:, 1]) / (np.sqrt(2) * L)

    for i in range(len(pts)):
        ax.add_patch(
            Circle(
                pts[i].cpu().detach().numpy(),
                radius=radii[i].cpu().detach().numpy(),
                color="white",
                alpha=0.1,
            )
        )

# dataset
x, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(x, y)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# loading models
exp2_mlp = MLP([2, 64, 64, 2], nn.Linear, nn.ReLU, name=None)
exp2_SN_mlp = MLP([2, 64, 64, 2], SNLinear, GroupSort, name="SN Linear")
exp2_AOL_mlp = MLP([2, 64, 64, 2], AOLLinear, GroupSort, name="AOL Linear")
load_models([exp2_mlp, exp2_SN_mlp, exp2_AOL_mlp], ["exp2_mlp.pth", "exp2_SN_mlp.pth", "exp2_AOL_mlp.pth"], device)

# plotting
fig, ax = plt.subplots(1, 3)
show_boundary(exp2_mlp, -3, 4, ax[0])
show_boundary(exp2_SN_mlp, -3, 4, ax[1])
show_boundary(exp2_AOL_mlp, -3, 4, ax[2])
show_lipshitz_constraints(exp2_SN_mlp, x.to(device), ax[1], 1)
show_lipshitz_constraints(exp2_AOL_mlp, x.to(device), ax[2], 1)
ax[0].scatter(x[:, 0], x[:, 1], color="black")
ax[1].scatter(x[:, 0], x[:, 1], color="black")
ax[2].scatter(x[:, 0], x[:, 1], color="black")
plt.show()

# evaluation
epsilon = 0.5
steps = 20
ce = nn.CrossEntropyLoss()

evaluate(exp2_mlp, [FSGM(epsilon, ce), IFSGM(epsilon, ce, steps), PGD(epsilon, ce, steps)], test_loader)
evaluate(exp2_SN_mlp, [FSGM(epsilon, ce), IFSGM(epsilon, ce, steps), PGD(epsilon, ce, steps)], test_loader)
evaluate(exp2_AOL_mlp, [FSGM(epsilon, ce), IFSGM(epsilon, ce, steps), PGD(epsilon, ce, steps)], test_loader)
certify_l2(exp2_AOL_mlp, epsilon, test_loader)


