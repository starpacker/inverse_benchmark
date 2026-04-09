import os

import sys

import numpy as np

import torch

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

torch.set_default_dtype(torch.float32)

def resize_array(arr, new_size):
    """Resize 2D array using PIL instead of cv2"""
    img = Image.fromarray(arr.astype(np.float32), mode='F')
    img_resized = img.resize((new_size, new_size), Image.BILINEAR)
    return np.array(img_resized)
