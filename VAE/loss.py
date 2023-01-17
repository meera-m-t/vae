import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, Module


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

    def forward(self, model, x):
        x_hat = model(x)
        vae_loss = ((x - x_hat) ** 2).sum() + 0.5 * model.encoder.kl  # (beta)
        return vae_loss


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar 
    def forward(self, outputs,  x):
        x_recon =outputs[0];mu = outputs[1]; logvar=outputs[2]
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD