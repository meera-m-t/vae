import os
import time  # Python Module
import numpy as np  # Third party pacakge
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from smt.sampling_methods import LHS
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Dataset_LHS(Dataset):
    def __init__(self,  split=0, n_samples=1000, n_dims=2,  data=None):
        self.data = data
        self.n_dims = n_dims
        metadata = self.create_dataset(n_samples)
        self.num_dimensions = n_dims
        train_valid_metadata, self.test_metadata = train_test_split(
            metadata, test_size=0.1, shuffle=True, random_state=42
        )
        self.train_metadata, self.valid_metadata = train_test_split(
            train_valid_metadata, test_size=0.1, shuffle=True, random_state=42
        )
        # self.transform = ToTensor()
        self.split = split

    def create_dataset(self, n_samples, n_dims=2):
        n = 10000*n_dims
        nt = 128
        f = 3.0                  # frequency in Hz
        t = np.linspace(0,1,nt)  # time stamps in s
        x = np.zeros((n,nt))
        x = np.random.uniform(-np.pi, np.pi, size=n)
        x = x.reshape(10000, n_dims)
        print(x.shape)
        normalized_vector = x / np.linalg.norm(x)
        return normalized_vector

    def _get_metadata(self):
        if self.split == 0:
            metadata = self.train_metadata
        elif self.split == 1:
            metadata = self.valid_metadata
        else:
            metadata = self.test_metadata
        return metadata

    def __getitem__(self, index):
        metadata = self._get_metadata()
        lhs = metadata[index]
        lhs = torch.from_numpy(lhs)
        return lhs

    def __len__(self):
        metadata = self._get_metadata()
        return len(metadata)

    def plot_dist(self, save_name: str) -> None:
        if self.num_dimensions > 3:
            raise ValueError("Cannot plot in more than 3 dimension")
        if self.num_dimensions == 2:
            self._plot_2d(save_name)
        else:
            self._plot_3d(save_name)

    def _plot_2d(self, save_name: str) -> None:
        data = self.data
        if data is None:          
            to_plot =  self._get_metadata()
        else:          
            to_plot = data.detach().cpu().numpy()

        plt.scatter(to_plot[:, 0], to_plot[:, 1])
        plt.xlabel("Var:0")
        plt.ylabel("Var:1")
        plt.savefig(save_name)

    def _plot_3d(self, save_name: str) -> None:
        data = self.data
        if data is None:  
            to_plot =  self._get_metadata()
        else:           
            to_plot = data.detach().cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2])
        ax.set_xlabel("Var:0")
        ax.set_ylabel("Var:1")
        ax.set_zlabel("Var:2")
        plt.savefig(save_name)



