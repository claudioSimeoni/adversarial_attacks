from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, X):
        return self.model(X)
