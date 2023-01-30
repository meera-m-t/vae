import argparse
from os import makedirs
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image


class SimpleLogger:
    def __init__(self, name):
        self.name = name

    def log(self, data):
        print(f"{self.name} - {data}")


def make_dirs(dirs):
    for dir_ in dirs:
        makedirs(dir_, exist_ok=True)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def change_range(set_, new_min=0, new_max=1):
    set_ = set_.detach().cpu().numpy()
    set_ = set_.reshape(set_.shape[0] * set_.shape[1])
    OldRange = max(set_) - min(set_)
    NewRange = new_max - new_min
    new_value = lambda i: (((i - min(set_)) * NewRange) / OldRange) + new_min
    new_value = np.vectorize(new_value)
    print(new_value[0])
    new_value = new_value.__doc__
    return torch.tensor(
        new_value(set_).reshape(set_.shape[0], set_.shape[1]), decice="cuda"
    )
