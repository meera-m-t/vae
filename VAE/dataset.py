import pandas as pd
import torch
import torchvision.transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2
import numpy as np
from smt.sampling_methods import LHS
from torchvision.transforms import (
    ToTensor,
)

class Dataset_LHS(Dataset):
    def __init__(
        self,  split=0, n_samples=1000
    ):
        metadata = self.create_dataset(n_samples)
        train_valid_metadata, self.test_metadata = train_test_split(
            metadata, test_size=0.1, shuffle=True, random_state=42
        )
        self.train_metadata, self.valid_metadata = train_test_split(
            train_valid_metadata, test_size=0.1, shuffle=True, random_state=42
        )                  
        # self.transform = ToTensor()
        self.split = split
    def create_dataset(self, n_samples, n_dims=9):
        # Create the dataset
        x1 = np.random.uniform([0,400])
        x2 = np.random.uniform([100, 300])
        x3 = np.random.uniform([0, 400])
        x4  = np.random.uniform([100, 300])
        x5 = np.random.uniform([0, 400])
        x6= np.random.uniform([100, 300])
        x7 = np.random.uniform([0, 400])
        x8 = np.random.uniform([100, 300])
        x9 = np.random.uniform([0, 400])
        x10 = np.random.uniform([100, 300])
        x11 = np.random.uniform([0, 400])
        x12 = np.random.uniform([100, 300])
        x13 = np.random.uniform([0, 400])
        x14 = np.random.uniform([100, 300])
        x15 = np.random.uniform([0, 400])
        x16 = np.random.uniform([100, 300])
        x17 = np.random.uniform([203, 225])
        x18 = np.random.uniform([350, 500])
        x19 = np.random.uniform([300, 500])
        x20 = np.random.uniform([300, 500])
        xlimits = np.array([x1, x2, x3, x4, x5,x6, x7, x8,x9])
        sampling = LHS(xlimits=xlimits)
        x = sampling(n_samples)
        normalized_vector = x / np.linalg.norm(x)
        # n_train = int(0.7 * n_samples)
        # n_test = int(0.15 * n_samples)
        # x_train = x[:n_train]
        # x_test = x[n_train:n_train + n_test]
        # x_val = x[n_train + n_test:]
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

if __name__ == "__main__":   
    dset = Dataset_LHS(split=0, n_samples=10000)
    for j in range(len(dset)):
        name = dset.train_metadata[j][0]
        lhs = dset[j]
        print(lhs.shape, lhs)
    print(len(dset))    
