import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from src.attacks import Attack

from dataclasses import dataclass


@dataclass
class AttackMetrics:
    name: str
    asr: float
    robust_accuracy: float

@dataclass
class EvaluationMetrics:
    model: str
    attacks: list[AttackMetrics]
    clean_accuracy: float
    adv_examples: list[list[torch.Tensor]]


def print_metrics(res : EvaluationMetrics):
    print(f"Model: {res.model} | Clean Accuracy: {res.clean_accuracy:.2%}")
    for i in range(len(res.attacks)):
        print(f"Attack: {res.attacks[i].name} | ASR: {res.attacks[i].asr:.2%} | "
              f"Robust Accuracy: {res.attacks[i].robust_accuracy:.2%}")
    print()


def certify_l2(model: nn.Module, epsilon : float, loader: DataLoader):
    device = next(model.parameters()).device

    cert_total = 0
    total = 0
    model_name = type(model).__name__ + (f" ({model.name})" if model.name else "") 
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
    
        top2 = torch.topk(logits, k=2, dim=1).values
        radii = torch.abs(top2[:,0] - top2[:, 1]) / (np.sqrt(2) * 1)
        corr_pred = logits.argmax(dim=1) == y.to(device)
        cert_total += torch.logical_and(radii >= epsilon, corr_pred).sum().item()
        total += y.numel()

    print(f"Certified L2 Accuracy of {model_name}: {cert_total / total:.2%}")
    return cert_total / total


def evaluate(model: nn.Module, attacks: list[Attack], loader: DataLoader, return_examples=True):
    device = next(model.parameters()).device
    res = EvaluationMetrics(
        model="",
        attacks=[AttackMetrics("", 0.0, 0.0) for i in attacks],
        clean_accuracy=0.0,
        adv_examples=[[] for i in attacks]
    )

    # running attacks
    total = 0
    corr_pred = 0
    adv_succ = [0 for i in range(len(attacks))]
    adv_corr_pred = [0 for i in range(len(attacks))]

    for batch_ind, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        corr_pred += (pred == y).sum().item()
        total += y.numel()

        for i, att in enumerate(attacks):
            adv_x = att.attack(model, x, y)

            adv_logits = model(adv_x)
            pred_adv = torch.argmax(adv_logits, dim=1)

            adv_succ[i] += torch.logical_and((pred == y), (pred != pred_adv)).sum().item()
            adv_corr_pred[i] += (pred_adv == y).sum().item()

            # if return_examples = True, we only return the first batch of adversarial examples 
            if(batch_ind == 0 and return_examples):
                res.adv_examples[i].append(adv_x)

    # saving and printing metrics
    res.model = type(model).__name__ + (f" ({model.name})" if model.name else "") 
    res.clean_accuracy = corr_pred / total

    for i in range(len(attacks)):
        res.attacks[i].name = type(attacks[i]).__name__
        res.attacks[i].asr = adv_succ[i] / corr_pred
        res.attacks[i].robust_accuracy = adv_corr_pred[i] / total
    print_metrics(res)

    return res


