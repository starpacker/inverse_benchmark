import numpy as np

import torch

import torch.nn.functional as F

def compute_pad_size(width: int):
    return int(np.ceil((np.sqrt(2)*width - width)/2))

def pad_object(object: torch.Tensor, mode='constant'):
    pad_size = compute_pad_size(object.shape[-2]) 
    if mode=='back_project':
        object = F.pad(object.unsqueeze(0), [0,0,0,0,pad_size,pad_size], mode='replicate').squeeze()
        object = F.pad(object, [0,0,pad_size,pad_size], mode='constant')
        return object
    else:
        return F.pad(object, [0,0,pad_size,pad_size,pad_size,pad_size], mode=mode)
