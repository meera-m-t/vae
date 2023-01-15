import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, Module
from torch import nn


class BCELoss(Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, y_pred, y_true):
        pred = y_pred.view(-1).double()
        truth = y_true.view(-1).double()

        # BCE loss
        bce_loss = BCEWithLogitsLoss()(pred, truth).double()

        return bce_loss


     

class VAELoss(Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self,model, x):
        x_hat = model(x)
        vae_loss = ((x - x_hat)**2).sum() + model.encoder.kl
        return vae_loss











