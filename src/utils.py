import torch


def load_models(models, names, device):
    for model, name in zip(models, names):
        state = torch.load(f"trained_models/{name}", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
