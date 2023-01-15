
from os import makedirs
from typing import List, Optional
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


class SimpleLogger:
    def __init__(self, name):
        self.name = name

    def log(self, data):
        print(f"{self.name} - {data}")


def make_dirs(dirs):
    for dir_ in dirs:
        makedirs(dir_, exist_ok=True)







