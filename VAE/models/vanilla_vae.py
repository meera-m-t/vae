import time

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torchsummary import summary

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, input_dims):
        super(VariationalEncoder, self).__init__()
        self.linear = nn.Linear(input_dims, 784)
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear2(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims, input_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, input_dims)
        self.input_dims = input_dims

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        return z.reshape((-1, 1, self.input_dims))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, input_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, input_dims)
        self.decoder = Decoder(latent_dims, input_dims)
        self._save_name = f"latent_dims-{latent_dims}-{int(time.time())}"

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_save_dir(self, num_training=None):
        if not num_training:
            return self._save_name
        else:
            return f"{self._save_name}-dataset_size-{num_training}"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VariationalAutoencoder(latent_dims=3, input_dims=9)
    summ = summary(model=vae.to(device), input_size=(9))
    print(summ)
