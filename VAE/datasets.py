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
    def __init__(self,  split=0, n_samples=10000, n_dims=3,  data=None):
        self.data = data
        self.n_dims = n_dims
        metadata = self.create_dataset(n_samples)
        self.num_dimensions = n_dims
        self.train_metadata = metadata   

    def create_dataset(self, n_samples, n_dims=3):
        n = n_samples*n_dims
        nt = 128
        f = 3.0                  # frequency in Hz
        t = np.linspace(0,1,nt)  # time stamps in s
        x = np.zeros((n,nt))
        x = np.random.uniform(-np.pi, np.pi, size=n)     
        print(min(x), max(x), "x min and max values") 
        # means = x.mean(axis=0)
        # stds = x.std(axis=0)
        # normalized_data = (x - means) / stds
        # x_norm = (x - x.min()) / (x.max() - x.min())       
        # x_norm = 2.0*x_norm - 1.0 
        return x.reshape(n_samples, n_dims) 

    def _get_metadata(self):
        metadata = self.train_metadata
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
            # x_norm = (x - x.min()) / (x.max() - x.min())       
            # x_norm = 2.0*x_norm - 1.0           
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
            print(data.shape, "data shape")         
            to_plot = data.detach().cpu().numpy() 
            print(min(data.flatten()), max(data.flatten()), "x min and max reconstruct values") 
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], s=0.5)
        ax.set_xlabel("Var:0")
        ax.set_ylabel("Var:1")
        ax.set_zlabel("Var:2")
        plt.savefig(save_name)



