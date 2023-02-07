import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .gaussianize import Gaussianize


def change_range(set_, new_min, new_max):
    set_ = set_.reshape(set_.shape[0] * set_.shape[1])
    OldRange = max(set_) - min(set_)
    NewRange = new_max - new_min
    new_value = lambda i: (((i - min(set_)) * NewRange) / OldRange) + new_min
    new_value = np.vectorize(new_value)
    return new_value(set_).reshape(set_.shape[0], set_.shape[1])


class check_overlab_distrbution(nn.Module):
    def __init__(self):
        super(check_overlab_distrbution, self).__init__()

    def convert_to_normal(self, U):
        out = Gaussianize()
        out.fit(U)  # Learn the parameters for the transformation
        y = out.transform(U)  # Transform x to y, where y should be normally distributed
        x_prime = out.inverse_transform(
            y
        )  # Inverting this transform should recover the data
        U = U.detach().cpu().numpy()
        print(np.allclose(x_prime, U), "check if the data is recovered")
        return y

    def draw_distrbution(self, orig_dis, predict_dis, dir_save):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        temp = ax1.hist(orig_dis)
        ax1.set_title("orignal_distribution")
        temp = ax2.hist(predict_dis)
        ax2.set_title("Predicted_distribution")
        temp = ax3.hist(orig_dis)
        temp = ax3.hist(predict_dis)
        ax3.set_title("overlap distribution")

        if not os.path.exists(f"{dir_save}/images/"):
            os.makedirs(f"{dir_save}/images/")
        plt.savefig(f"{dir_save}/images/distrbution.png")

    def forward(self, x, y, dir_save):
        orig_dis = self.convert_to_normal(x)
        print(orig_dis.shape, "orig_dis shape")
        predict_dis = self.convert_to_normal(y)
        print(predict_dis.shape, "predict_dis shape")
        self.draw_distrbution(orig_dis, predict_dis, dir_save)
