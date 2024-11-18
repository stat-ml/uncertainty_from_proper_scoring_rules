"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn
import torch.nn.functional as F
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

## Are taken from example from https://github.com/IntelLabs/bayesian-torch/tree/main
BNN_PRIORS = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",
        "moped_enable": True,
        "moped_delta": 0.5,
}
    
def ResNet18Flipout(n_classes):
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, n_classes)
    dnn_to_bnn(model, BNN_PRIORS)
    return model

def test():
    n_classes = 10
    net = ResNet18Flipout(n_classes)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
