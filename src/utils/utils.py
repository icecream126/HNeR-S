import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from numbers import Integral
import math
from torch import nn
import cv2
from math import sin, cos, sqrt, atan2, radians
import healpy as hp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def col_to_cartesian(points):
    theta, phi = points[..., 0], points[..., 1]

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)


def mse2psnr(mse):
    return -10.0 * torch.log10(mse)


def to_cartesian(points):
    theta, phi = points[..., 0], points[..., 1]

    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)


EPSILON = 1e-5


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )


class MinMaxScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, max=None, min=None):
        self.max = max
        self.min = min

    def fit(self, X):
        self.max = torch.max(X, dim=0)[0]
        self.min = torch.min(X, dim=0)[0]

    def transform(self, X):
        return (X - self.min) / (self.max - self.min + EPSILON)

    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min

    def match_device(self, tensor):
        if self.max.device != tensor.device:
            self.max = self.max.to(tensor.device)
            self.min = self.min.to(tensor.device)

    def copy(self):
        return MinMaxScalerTorch(
            maxs=self.max.clone().detach(), min=self.min.clone().detach()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max: {self.max.tolist()}, "
            f"min: {self.min.tolist()})"
        )
