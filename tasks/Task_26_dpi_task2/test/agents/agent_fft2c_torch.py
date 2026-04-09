import os

import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

torch.set_default_dtype(torch.float32)

def fft2c_torch(img):
    """2D FFT for torch tensors, returns real/imag stacked on last dim"""
    x = img.unsqueeze(-1)
    x = torch.cat([x, torch.zeros_like(x)], -1)
    xc = torch.view_as_complex(x)
    kc = torch.fft.fft2(xc, norm="ortho")
    return torch.view_as_real(kc)
