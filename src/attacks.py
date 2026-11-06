import torch


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
        label = label.to(device)
        loss = loss_fn(logits, label)

        img_grad = torch.autograd.grad(loss, img)[0]
        new_img = (img + torch.sign(img_grad) * self.epsilon).clamp(0, 1).detach()

        adv_logits = model(new_img)
        pred_classes = torch.argmax(logits, dim=1)
        pred_adv_classes = torch.argmax(adv_logits, dim=1)
        return pred_classes, pred_adv_classes

