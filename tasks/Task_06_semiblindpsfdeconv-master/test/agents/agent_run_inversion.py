import logging

import numpy as np

from numpy.fft import rfft2, irfft2

from scipy.interpolate import griddata

import scipy.ndimage

import torch

from functools import reduce

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scale(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    out = v / norm
    return out * (1/np.max(np.abs(out)))

def normalize(v):
    norm = v.sum()
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm

def gaussian_kernel(size, fwhmx=3, fwhmy=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return normalize(np.exp(-4 * np.log(2) * (((x - x0) ** 2) / fwhmx**2 + ((y - y0) ** 2) / fwhmy**2)))

def unpad(img, npad):
    return img[npad:-npad, npad:-npad]

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0
    return c

def _centered(arr, newshape):
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def divergence(F):
    return reduce(np.add, np.gradient(F))

def compute_grid_in_memory(psf_map_shape, input_image_shape):
    """
    Computes interpolation grid coefficients.
    psf_map_shape: tuple (rows, cols)
    input_image_shape: tuple (H, W)
    """
    grid_z1 = []
    grid_x, grid_y = np.mgrid[0:input_image_shape[0], 0:input_image_shape[1]]
    xmax = np.linspace(0, input_image_shape[0], psf_map_shape[0])
    ymax = np.linspace(0, input_image_shape[1], psf_map_shape[1])

    total_patches = psf_map_shape[0] * psf_map_shape[1]
    
    for i in range(total_patches):
        points = []
        values = []
        for x in xmax:
            for y in ymax:
                points.append(np.asarray([x, y]))
                values.append(0.0)

        values[i] = 1.0
        points = np.asarray(points)
        values = np.asarray(values)
        grid_z1.append(griddata(points, values, (grid_x, grid_y), method='linear', rescale=True))
    
    return grid_z1

def run_inversion(blurred_img, cnn_model, step=64, iterations=15, lbd=0.05):
    """
    Estimates local PSFs using CNN and runs Spatially Variant Richardson-Lucy.
    """
    # --- Step A: PSF Estimation using Sliding Window ---
    size = 128
    num_classes = 2
    
    # Pad or crop logic similar to sliding window in original
    # We create a weight image to handle overlapping windows
    weight_image = np.zeros((blurred_img.shape[0], blurred_img.shape[1], num_classes))
    tile_dataset = []
    coords = []
    
    # Extract patches
    for x_end in range(size, blurred_img.shape[0] + 1, step):
        for y_end in range(size, blurred_img.shape[1] + 1, step):
            a = blurred_img[x_end - size:x_end, y_end - size:y_end]
            # Neural network expects locally normalized patches (max=1)
            a = scale(a) 
            tile_dataset.append(a[:])
            coords.append((x_end, y_end))
            weight_image[x_end - size:x_end, y_end - size:y_end] += 1.0

    if not tile_dataset:
        # Fallback for very small images
        return blurred_img 

    tile_dataset = np.asarray(tile_dataset)
    tile_dataset = np.reshape(tile_dataset, (tile_dataset.shape[0], 1, size, size))
    
    # Batch inference
    max_size = tile_dataset.shape[0]
    batch_size = 8
    it = 0
    output_npy = np.zeros((tile_dataset.shape[0], num_classes))
    input_tensor = torch.FloatTensor(tile_dataset)

    with torch.no_grad():
        while max_size > 0:
            num_batch = min(batch_size, max_size)
            batch_input = input_tensor.narrow(0, it, num_batch).to(device)
            out = cnn_model(batch_input)
            output_npy[it:it+num_batch] = out.data.cpu().numpy()
            it += num_batch
            max_size -= num_batch

    # Aggregate predictions
    estimated_map_full = np.zeros((blurred_img.shape[0], blurred_img.shape[1], num_classes))
    
    for i, (x_end, y_end) in enumerate(coords):
        estimated_map_full[x_end - size:x_end, y_end - size:y_end] += output_npy[i, :]

    estimated_map_full = estimated_map_full / weight_image
    
    # --- Step B: Downsample and Filter Map ---
    map_downsampled = estimated_map_full[::step, ::step]
    output_filtered = [scipy.ndimage.median_filter(map_downsampled[:,:,i], size=(2,2), mode='reflect') for i in range(2)]
    
    grid_h, grid_w = output_filtered[0].shape
    
    # --- Step C: Generate Grid PSFs ---
    rec_psf_list = []
    for i in range(grid_h * grid_w):
        r, c = i // grid_w, i % grid_w
        fx = max(0.1, output_filtered[0][r, c])
        fy = max(0.1, output_filtered[1][r, c])
        psf = gaussian_kernel(31, fx, fy)
        rec_psf_list.append(psf)
        
    rec_psf_list = np.asarray(rec_psf_list)
    
    # --- Step D: Compute Reconstruction Weights ---
    # Need shape of the grid for interpolation logic
    rec_grid_weights = compute_grid_in_memory((grid_h, grid_w), blurred_img.shape)
    
    # --- Step E: Richardson-Lucy Deconvolution ---
    img_list = []
    # Mask image for each PSF region
    for i in range(len(rec_psf_list)):
        img_list.append(np.multiply(rec_grid_weights[i], blurred_img))
    
    # Pre-processing for RL
    min_values = []
    processed_img_list = []
    
    pad_width_rl = np.max(rec_psf_list[0].shape)
    for img in img_list:
        img_padded = np.pad(img, pad_width_rl, mode='reflect')
        min_val = np.min(img)
        img_padded = img_padded - min_val
        min_values.append(min_val)
        processed_img_list.append(img_padded)
        
    latent_estimate = [img.copy() for img in processed_img_list]
    error_estimate = [img.copy() for img in processed_img_list]
    
    # FFT Pre-calc
    fft_shape_calc = np.array(processed_img_list[0].shape) + np.array(rec_psf_list[0].shape) - 1
    fsize = [scipy.fftpack.next_fast_len(int(d)) for d in fft_shape_calc]
    fslice = tuple([slice(0, int(sz)) for sz in fft_shape_calc])
    
    psf_f = []
    psf_flipped_f = []
    for i in range(len(latent_estimate)):
        psf_f.append(rfft2(rec_psf_list[i], fsize))
        _psf_flipped = np.flip(rec_psf_list[i], axis=0)
        _psf_flipped = np.flip(_psf_flipped, axis=1)
        psf_flipped_f.append(rfft2(_psf_flipped, fsize))
        
    # Iterations
    for it in range(iterations):
        regularization = np.ones(processed_img_list[0].shape)
        
        for idx in range(len(latent_estimate)):
            # Forward
            est_conv = irfft2(np.multiply(psf_f[idx], rfft2(latent_estimate[idx], fsize)))[fslice].real
            est_conv = _centered(est_conv, processed_img_list[idx].shape)
            
            relative_blur = div0(processed_img_list[idx], est_conv)
            
            # Backward
            error_est = irfft2(np.multiply(psf_flipped_f[idx], rfft2(relative_blur, fsize)), fsize)[fslice].real
            error_est = _centered(error_est, processed_img_list[idx].shape)
            
            # TV
            div_val = divergence(latent_estimate[idx] / (np.linalg.norm(latent_estimate[idx], ord=1) + 1e-10))
            regularization += 1.0 - (lbd * div_val)
            
            latent_estimate[idx] = np.multiply(latent_estimate[idx], error_est)
            
        # Update Latent
        for idx in range(len(latent_estimate)):
            latent_estimate[idx] = np.divide(latent_estimate[idx], regularization/float(len(latent_estimate)))
            
    # Unpad and Combine
    final_output = np.zeros_like(blurred_img)
    for idx, img in enumerate(latent_estimate):
        img_res = img + min_values[idx]
        img_res = unpad(img_res, pad_width_rl)
        
        # We sum up the contributions. In original code: np.sum(latent_estimate, axis=0)
        # However, RL logic with masked inputs usually sums up the partial deconvolutions.
        if idx == 0:
            final_output = img_res
        else:
            final_output += img_res
            
    return np.clip(final_output, 0, 1)
