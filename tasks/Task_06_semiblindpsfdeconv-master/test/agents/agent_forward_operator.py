import logging

import numpy as np

from numpy.fft import rfft2, irfft2

import scipy.ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def unpad(img, npad):
    return img[npad:-npad, npad:-npad]

def _centered(arr, newshape):
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def forward_operator(image, psf_grid, grid_weights):
    """
    Simulates spatially variant blur.
    image: 2D numpy array
    psf_grid: list of PSF kernels (numpy arrays)
    grid_weights: list of weight maps (same length as psf_grid)
    """
    blurred_image = np.zeros_like(image)
    
    pad_width = np.max(psf_grid[0].shape)
    img_padded = np.pad(image, pad_width, mode='reflect')
    
    fft_shape_calc = np.array(img_padded.shape) + np.array(psf_grid[0].shape) - 1
    fsize = [scipy.fftpack.next_fast_len(int(d)) for d in fft_shape_calc]
    fslice = tuple([slice(0, int(sz)) for sz in fft_shape_calc])
    
    img_f = rfft2(img_padded, fsize)
    
    for i, psf in enumerate(psf_grid):
        psf_f = rfft2(psf, fsize)
        convolved = irfft2(np.multiply(psf_f, img_f), fsize)[fslice].real
        convolved = _centered(convolved, img_padded.shape)
        convolved = unpad(convolved, pad_width)
        
        blurred_image += convolved * grid_weights[i]
        
    return blurred_image
