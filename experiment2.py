import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, X):
        return self.model(X)


train_data = datasets.FashionMNIST(
    root = "data", 
    train = True,
    download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root = "data", 
    train = False,
    download = True,
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

#################################################################################

# reading the parameters of the trained model
model = MLP()
state = torch.load("model.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()


t = 1000
eps_vars = [0.1, 0.05, 0.01]
fig, ax = plt.subplots(len(eps_vars), 2)
for eind, epsilon in enumerate(eps_vars):
    adv_success = 0
    correct_pred = 0
    for i in range(t):
        idx = torch.randint(len(test_data), size=(1,)).item()
        img, label = train_data[idx]
        img = img.to(device)
        img.requires_grad = True


        logits = model(img)
        label = torch.tensor([label], device=device)
        loss = loss_fn(logits, label)

        img_grad = torch.autograd.grad(loss, img)[0]
        new_img = (img + torch.sign(img_grad) * epsilon).clamp(0, 1).detach()

        adv_logits = model(new_img)
        pred_class = torch.argmax(logits)
        pred_adv = torch.argmax(adv_logits)

        if(i == 0): ax[eind][0].imshow(img.cpu().squeeze().detach().numpy(), cmap="grey")
        if(i == 0): ax[eind][1].imshow(new_img.cpu().squeeze().detach().numpy(), cmap="grey")

        if(pred_class == label): correct_pred += 1
        if(pred_class == label and pred_class != pred_adv): adv_success += 1
        # print(f"Label: {label}, Predicted Class: {pred_class}, Predicted adv: {pred_adv}")

    print(f"Epsilon = {epsilon}, Success rate = {correct_pred / t * 100}%, Adversarial working attacks = {adv_success / correct_pred * 100}%")

plt.show()




