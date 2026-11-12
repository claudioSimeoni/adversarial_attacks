import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from src.attacks import Attack, FSGM


def evaluate(model: nn.Module, attacks: list[Attack], loader: DataLoader):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()

    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    adv_list = []

    logits = model(x)
    pred = torch.argmax(logits, dim=1)
    corr_pred = (pred == y).sum().item()
    total = y.numel()

    print(f"Model: {type(model)} | Clean Accuracy: {corr_pred / total * 100}%")
    
    for att in attacks:
        adv_x = att.attack(model, loss_fn, x, y)
        adv_list.append(adv_x)

        adv_logits = model(adv_x)
        pred_adv = torch.argmax(adv_logits, dim=1)

        adv_succ = torch.logical_and((pred == y), (pred != pred_adv)).sum().item()
        adv_corr_pred = (pred_adv == y).sum().item()

        att_succ_rate = adv_succ / corr_pred * 100
        emp_rob_acc = adv_corr_pred / total * 100

        print(f"Attack: {type(att)} | Att Succ Rate: {att_succ_rate}% | Empirical Robust Acc: {emp_rob_acc}%")

    return adv_list


def evaluate_l2_lipshitz(model: nn.Module, attacks: list[Attack], loader: DataLoader):
    device = next(model.parameters()).device
    x, y = next(iter(loader))
    x = x.to(device)
    logits = model(x)

    adv_list = evaluate(model, attacks, loader)

    top2 = torch.topk(logits, k=2, dim=1).values
    radii = torch.abs(top2[:,0] - top2[:, 1]) / (np.sqrt(2) * 1)
    cert_l2_acc = (radii <= attacks[2].epsilon).sum().item() / y.numel() * 100 # this is not general at all, TODO: remove or change this function?

    print(f"Certified L2 Accuracy of the lipshitz model: {cert_l2_acc}%")
    
    return adv_list
