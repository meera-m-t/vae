import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, Module

class customLoss(nn.Module):
    def __init__(self, beta=-0.5):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.beta = beta
    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar 
    def forward(self, outputs,  x):        
        x_recon =outputs[0];mu = outputs[1]; logvar=outputs[2]
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = self.beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD