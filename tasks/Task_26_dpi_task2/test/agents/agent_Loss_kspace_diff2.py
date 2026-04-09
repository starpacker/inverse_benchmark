import os

import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

torch.set_default_dtype(torch.float32)

def Loss_kspace_diff2(sigma):
    """K-space L2 loss function"""
    def func(y_true, y_pred):
        return torch.mean((y_pred - y_true)**2, (1, 2, 3)) / (sigma)**2
    return func
