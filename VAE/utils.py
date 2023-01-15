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
