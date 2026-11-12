import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from sklearn import datasets

from src.attacks import FSGM, IFSGM, PGD
from src.evaluation import evaluate, evaluate_l2_lipshitz
from src.models import MLP, LipshitzLayerSN, LipshitzLayerAOL, GroupSort
from src.utils import load_models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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


# loading models
model = MLP([2, 64, 64, 2], nn.Linear, nn.ReLU)
lipshitz_model = MLP([2, 64, 64, 2], LipshitzLayerAOL, GroupSort)
load_models([model, lipshitz_model], ["mlp_moon.pth", "lipshitzAOL_mlp_moon.pth"], device)

# preparing the ds
x, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(x, y)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# plotting
fig, ax = plt.subplots(1, 2)
show_boundary(model, -3, 4, ax[0])
show_boundary(lipshitz_model, -3, 4, ax[1])
show_lipshitz_constraints(lipshitz_model, x.to(device), ax[1], 1)
ax[0].scatter(x[:, 0], x[:, 1], color="black")
ax[1].scatter(x[:, 0], x[:, 1], color="black")

# evaluation
epsilon = 0.5
steps = 20

adv_linear = evaluate(model, [FSGM(epsilon), IFSGM(epsilon, steps), PGD(epsilon, steps, norm=2)], test_loader)
adv_lipshitz = evaluate_l2_lipshitz(
    lipshitz_model,
    [FSGM(epsilon), IFSGM(epsilon, steps), PGD(epsilon, steps, norm=2)],
    test_loader,
)

adv_linear_fsgm = adv_linear[0].cpu()
adv_lipshitz_fsgm = adv_lipshitz[0].cpu()

# plotting fsgm adversarial data
ax[0].scatter(adv_linear_fsgm[:, 0], adv_linear_fsgm[:, 1], color="grey")
ax[1].scatter(adv_lipshitz_fsgm[:, 0], adv_lipshitz_fsgm[:, 1], color="grey")
plt.show()
