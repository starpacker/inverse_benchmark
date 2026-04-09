import numpy as np

import torch

import torch.nn.functional as F

def compute_pad_size(width: int):
    return int(np.ceil((np.sqrt(2)*width - width)/2))

def pad_proj(proj: torch.Tensor, mode: str = 'constant', value: float = 0):
    pad_size = compute_pad_size(proj.shape[-2])  
    return F.pad(proj, [0,0,pad_size,pad_size], mode=mode, value=value)
