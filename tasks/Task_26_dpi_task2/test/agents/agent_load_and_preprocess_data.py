import os
import sys
import numpy as np
import torch
import pickle
from PIL import Image

# Ensure local modules can be found if necessary
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

torch.set_default_dtype(torch.float32)

def resize_array(arr, new_size):
    """Resize 2D array using PIL instead of cv2"""
    img = Image.fromarray(arr.astype(np.float32), mode='F')
    img_resized = img.resize((new_size, new_size), Image.BILINEAR)
    return np.array(img_resized)

def resize_mask(arr, new_size):
    """Resize mask array using nearest neighbor"""
    img = Image.fromarray(arr.astype(np.uint8))
    img_resized = img.resize((new_size, new_size), Image.NEAREST)
    return np.array(img_resized)

def fft2c(data):
    """2D FFT with ortho normalization, returns real/imag stacked"""
    data = np.fft.fft2(data, norm="ortho")
    return np.stack((data.real, data.imag), axis=-1)

def load_and_preprocess_data(impath, maskpath, npix, sigma):
    """
    Load MRI data and mask, preprocess for reconstruction.
    """
    # Load target image from pickle
    with open(impath, 'rb') as f:
        obj = pickle.load(f)
        img_true = obj['target']
    
    # Resize image to target size
    img_true = resize_array(img_true, npix)
    
    # Compute k-space and add noise
    kspace = fft2c(img_true)
    kspace = kspace + np.random.normal(size=kspace.shape) * sigma
    
    # Load and process mask
    mask = np.load(maskpath)
    if mask.shape[0] != npix:
        mask = resize_mask(mask, npix)
    
    # Ensure center region is fully sampled
    center_size = 8
    center_start = npix // 2 - center_size
    center_end = npix // 2 + center_size
    mask[center_start:center_end, center_start:center_end] = 1
    
    # FFT shift mask and stack for real/imag components
    mask = np.fft.fftshift(mask)
    mask = np.stack((mask, mask), axis=-1)
    
    # Compute flux (total intensity)
    flux = np.sum(img_true)
    
    return {
        'img_true': img_true,
        'kspace': kspace,
        'mask': mask,
        'flux': flux,
        'npix': npix,
        'sigma': sigma
    }