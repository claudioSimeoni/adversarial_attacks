import torch
import matplotlib.pyplot as plt


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
        label = label.to(device)
        img = img.to(device)
        img.requires_grad = True

        logits = model(img)
        loss = loss_fn(logits, label)

        img_grad = torch.autograd.grad(loss, img)[0]
        new_img = (img + torch.sign(img_grad) * self.epsilon).clamp(0, 1).detach()

        adv_logits = model(new_img)
        pred_classes = torch.argmax(logits, dim=1)
        pred_adv_classes = torch.argmax(adv_logits, dim=1)
        return pred_classes, pred_adv_classes


class PGD(Attack):
    def __init__(self, epsilon, steps):
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = epsilon / steps

    def attack(self, model, loss_fn, img, label):
        device = next(model.parameters()).device
        label = label.to(device)
        img = img.to(device)

        clamp_min, clamp_max = img-self.epsilon, img+self.epsilon
        pred_class = torch.argmax(model(img), dim=1)

        img = img + (torch.rand(*img.shape).to(device) * 2 * self.epsilon - self.epsilon) 
        img.requires_grad = True

        for _ in range(self.steps):
            logits = model(img)
            loss = loss_fn(logits, label)

            img_grad = torch.autograd.grad(loss, img)[0]
            img = (
                (img + torch.sign(img_grad) * self.alpha).clamp(min=clamp_min, max=clamp_max).detach()
            )
            img.requires_grad = True

        adv_logits = model(img)
        pred_adv = torch.argmax(adv_logits, dim=1)
        return pred_class, pred_adv

class IFSGM(Attack):
    def __init__(self, epsilon, steps):
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = epsilon / steps

    def attack(self, model, loss_fn, img, label):
        device = next(model.parameters()).device
        label = label.to(device)
        img = img.to(device)
        img.requires_grad = True

        clamp_min, clamp_max = img-self.epsilon, img+self.epsilon
        pred_class = torch.argmax(model(img), dim=1)

        for _ in range(self.steps):
            logits = model(img)
            loss = loss_fn(logits, label)

            img_grad = torch.autograd.grad(loss, img)[0]
            img = (
                (img + torch.sign(img_grad) * self.alpha).clamp(min=clamp_min, max=clamp_max).detach()
            )
            img.requires_grad = True

        adv_logits = model(img)
        pred_adv = torch.argmax(adv_logits, dim=1)
        return pred_class, pred_adv
