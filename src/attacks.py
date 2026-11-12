import torch
import matplotlib.pyplot as plt


class Attack:
    def __init__(self):
        pass

    def attack(self, model, loss_fn, img, label):
        pass


class FSGM(Attack):
    def __init__(self, epsilon, bounds=(None, None)):
        self.epsilon = epsilon
        self.bounds = bounds

    def attack(self, model, loss_fn, img, label):
        device = next(model.parameters()).device
        label = label.to(device)
        img = img.to(device)
        img.requires_grad = True

        logits = model(img)
        loss = loss_fn(logits, label)

        img_grad = torch.autograd.grad(loss, img)[0]
        img = (img + torch.sign(img_grad) * self.epsilon).detach()

        if self.bounds != (None, None):
            img = img.clamp(*self.bounds)

        return img


class PGD(Attack):
    def __init__(self, epsilon, steps, bounds=(None, None), norm=None):
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = 2 * epsilon / steps
        self.bounds = bounds
        self.norm = norm

    def attack(self, model, loss_fn, img, label):
        device = next(model.parameters()).device
        label = label.to(device)
        img = img.to(device)

        # random img initialization
        img0 = img.detach().clone()
        img = img.detach() + (torch.rand(*img.shape).to(device) * 2 * self.epsilon - self.epsilon)

        for _ in range(self.steps):
            img.requires_grad = True

            logits = model(img)
            loss = loss_fn(logits, label)
            img_grad = torch.autograd.grad(loss, img)[0]

            img_adv = img + torch.sign(img_grad) * self.alpha
            delta = img_adv - img0

            # projecting with respect to the norm
            if not self.norm:
                img = (img0 + delta.clamp(min=-self.epsilon, max=self.epsilon)).detach()
            else:
                delta_norm = torch.linalg.vector_norm(delta.flatten(start_dim=1), ord=self.norm, dim=1, keepdim=True)
                img = (img0 + delta * self.epsilon / delta_norm.clamp(min=self.epsilon)).detach()

            if self.bounds != (None, None):
                img = img.clamp(*self.bounds)

        return img


class IFSGM(Attack):
    def __init__(self, epsilon, steps, bounds=(None, None)):
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = epsilon / steps
        self.bounds = bounds

    def attack(self, model, loss_fn, img, label):
        device = next(model.parameters()).device
        label = label.to(device)
        img = img.to(device)
        img0 = img.detach().clone()

        for _ in range(self.steps):
            img.requires_grad = True

            logits = model(img)
            loss = loss_fn(logits, label)
            img_grad = torch.autograd.grad(loss, img)[0]

            img_adv = img + torch.sign(img_grad) * self.alpha
            delta = img_adv - img0

            # clipping
            img = (img0 + delta.clamp(min=-self.epsilon, max=self.epsilon)).detach()

            if self.bounds != (None, None):
                img = img.clamp(*self.bounds)

        return img
