import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np


class Attack:
    def __init__(self):
        pass

    def attack(self, model, loss_fn, img, label):
        pass


class FSGM(Attack):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack(self, model, loss_fn, img, label):
        device = next(model.parameters()).device
        img = img.to(device)
        img.requires_grad = True

        logits = model(img)
        label = torch.tensor([label], device=device)
        loss = loss_fn(logits, label)

        img_grad = torch.autograd.grad(loss, img)[0]
        new_img = (img + torch.sign(img_grad) * self.epsilon).clamp(0, 1).detach()

        adv_logits = model(new_img)
        pred_class = torch.argmax(logits)
        pred_adv = torch.argmax(adv_logits)
        return pred_class, pred_adv

