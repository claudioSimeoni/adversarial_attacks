import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

from src.models import MLP, SNLinear, AOLLinear, CaleyLinear, GroupSort, CNN, AOLConv2d, HingeLoss
from src.training import train
from src.utils import load_models
from src.evaluation import evaluate
from src.attacks import FSGM, IFSGM, PGD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# dataset
test_data = datasets.MNIST(root="data", train=False, download=True, 
                            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# loading models
exp3_mlp = MLP([784, 128, 128, 10], nn.Linear, nn.ReLU, name=None)
exp3_cnn = CNN(10, nn.Conv2d, nn.ReLU, nn.Linear, name=None)
exp3_cnn_AOL = CNN(10, AOLConv2d, GroupSort, AOLLinear, name="AOL CNN")
load_models([exp3_mlp, exp3_cnn, exp3_cnn_AOL], 
            ["exp3_mlp.pth", "exp3_cnn.pth", "exp3_cnn_AOL.pth"], device)

# evaluation
epsilon = 0.05
steps = 20
ce = nn.CrossEntropyLoss()
hl = HingeLoss(1)

# evaluate(exp3_mlp, [FSGM(epsilon, ce), IFSGM(epsilon, ce, steps), PGD(epsilon, ce, steps)], test_loader)
# evaluate(exp3_cnn, [FSGM(epsilon, ce), IFSGM(epsilon, ce, steps), PGD(epsilon, ce, steps)], test_loader)
# evaluate(exp3_cnn_AOL, [FSGM(epsilon, ce), IFSGM(epsilon, ce, steps), PGD(epsilon, ce, steps)], test_loader)
evaluate(exp3_cnn_AOL, [FSGM(epsilon, hl), IFSGM(epsilon, hl, steps), PGD(epsilon, hl, steps)], test_loader)
# certify_l2(exp2_AOL_mlp, epsilon, test_loader)
