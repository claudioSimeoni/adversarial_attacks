import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, X):
        return self.model(X)


train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

#################################################################################

# TODO: split dataset into validation and training, maybe plot both to check overfitting?
# TODO: try and benchmark different architectures / loss function / optimizer?


fig, ax = plt.subplots()


def train(model, loss_fn, optimizer, loader, epochs):
    model.train()
    loss_list = []

    for _ in range(epochs):
        epoch_loss = 0

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            # single batch
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(epoch_loss / len(loader))

    return loss_list


def evaluate(model, test_loader):
    model.eval()
    succ = 0
    total = 0
    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            succ += (torch.argmax(logits, dim=1) == y).sum().item()
            total += len(y)

    return succ / total * 100


batch_sizes = [32]
lrs = [0.001]
model = None

for i in batch_sizes:
    for j in lrs:
        torch.manual_seed(42)
        epochs = 5
        model = MLP().to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), j)

        loader = DataLoader(train_data, batch_size=i, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=i, shuffle=False)

        loss_list = train(model, loss_fn, optimizer, loader, epochs)
        ax.plot(
            [i for i in range(1, epochs + 1)],
            loss_list,
            label=f"batch size = {i}, lr = {j}",
        )

        succ_rate = evaluate(model, test_loader)
        print(f"Batch size = {i}, Learning rate = {j}, Success rate = {succ_rate}%")

torch.save(model.state_dict(), "model.pth")


ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)
plt.show()
