import os

import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

torch.set_default_dtype(torch.float32)

def Loss_TV(y_pred):
    """Total variation loss"""
    return torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) + \
           torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2))
