import os

import sys

import numpy as np

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

torch.set_default_dtype(torch.float32)

def fft2c(data):
    """2D FFT with ortho normalization, returns real/imag stacked"""
    data = np.fft.fft2(data, norm="ortho")
    return np.stack((data.real, data.imag), axis=-1)
