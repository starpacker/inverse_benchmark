```python
import os
import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2

# ==============================================================================
# Helper Functions (Inlined from utils.py)
# ==============================================================================

def forward_operator(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI).
    Multiple encoded frames are collapsed into a single measurement.
    Phi: Sensing matrix (Mask)
    '''
    return np.sum(x * Phi, axis=2)

def backward_operator(y, Phi):
    '''
    Transpose of the forward model.
    '''
    return np.multiply(np.repeat(y[:,:,np.newaxis], Phi.shape[2], axis=2), Phi)

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def shift(inputs, step):
    '''
    Simulate the dispersion effect (spatial shift).
    '''
    [row, col, nC] = inputs.shape
    output = np.zeros((row, col+(nC-1)*step, nC))
    for i in range(nC):
        output[:, i*step:i*step+row, i] = inputs[:, :, i]
    return output

def shift_back(inputs, step):
    '''
    Reverse the dispersion effect.
    '''
    [row, col, nC] = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = np.roll(inputs[:, :, i], (-1)*step*i, axis=1)
    output = inputs[:, 0:col-step*(nC-1), :]
    return output

def TV_denoiser(x, _lambda, n_iter_max):
    '''
    Total Variation Denoiser (Chambolle's algorithm).
    '''
    dt = 0.25
    N = x.shape
    idx = np.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    iux = np.arange(-1, N[0]-1)
    iux[0] = 0
    ir = np.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    il = np.arange(-1, N[1]-1)
    il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x * _lambda
        z1 = z[:, ir, :] - z
        z2 = z[idx, :, :] - z
        denom_2d = 1 + dt * np.sqrt(np.sum(z1**2 + z2**2, 2))
        denom_3d = np.tile(denom_2d[:, :, np.newaxis], (1, 1, N[2]))
        p1 = (p1 + dt * z1) / denom_3d
        p2 = (p2 + dt * z2) / denom_3d
        divp = p1 - p1[:, il, :] + p2 - p2[iux, :, :]
    
    u = x - divp / _lambda
    return u

# ==============================================================================
# Solvers (Inlined from dvp_linear_inv_cassi.py)
# ==============================================================================

def gap_denoise(y, Phi, _lambda=1, iter_max=20, tv_weight=6, tv_iter_max=5, X_orig=None):
    '''
    Generalized Alternating Projection (GAP) for SCI reconstruction.
    Using TV as the denoiser here for simplicity and robustness without external weights.
    '''
    # Initialization
    x = backward_operator(y, Phi) # A^T(y)
    y1 = np.zeros_like(y)
    
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum == 0] = 1
    
    psnr_all = []
    
    print(f"Starting GAP-TV Reconstruction for {iter_max} iterations...")
    
    for k in range(iter_max):
        # 1. Data Projection (GAP)
        yb = forward_operator(x, Phi)
        y1 = y1 + (y - yb)
        x = x + _lambda * (backward_operator((y1 - yb) / Phi_sum, Phi))
        
        # 2. Denoising (Prior)
        # Shift back to image domain for denoising
        x_img = shift_back(x, step=1)
        
        # TV Denoising
        x_img = TV_denoiser(x_img, tv_weight, n_iter_max=tv_iter_max)
        
        # Shift forward to measurement domain
        x = shift(x_img, step=1)
        
        # Metrics
        if X_orig is not None:
            current_psnr = psnr(X_orig, x_img)
            psnr_all.append(current_psnr)
            if (k+1) % 1 == 0:
                print(f"  Iteration {k+1}/{iter_max}, PSNR: {current_psnr:.2f} dB")
                
    return x_img, psnr_all

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    # Configuration
    datname = 'kaist_crop256_01'
    matfile = f'Dataset/{datname}.mat'
    maskfile = 'mask/mask256.mat'
    
    if not os.path.exists(matfile) or not os.path.exists(maskfile):
        print(f"Error: Data files not found. Please ensure {matfile} and {maskfile} exist.")
        exit(1)
        
    # 1. Load Data
    print("Loading data...")
    # Load Truth
    truth = sio.loadmat(matfile)['img']
    # Normalize truth to [0, 1] if not already
    if truth.max() > 1:
        truth = truth / 255.0
        
    r, c, nC = truth.shape
    step = 1
    
    # Load and Prepare Mask
    mask256 = sio.loadmat(maskfile)['mask']
    mask = np.zeros((r, c + step * (nC - 1)))
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
    
    for i in range(nC):
        mask_3d[:, i:i+256, i] = mask256
        
    # Generate Measurement (Simulation)
    print("Simulating measurement...")
    truth_shift = shift(truth, step=step)
    meas = np.sum(mask_3d * truth_shift, 2)
    
    # Save Measurement
    plt.imsave('measurement.png', meas, cmap='gray')
    
    # 2. Reconstruction (GAP-TV)
    # Using small iteration count as requested
    n_iter = 5
    print(f"Running reconstruction (GAP-TV) with {n_iter} iterations...")
    
    recon_img, psnrs = gap_denoise(
        meas, 
        mask_3d, 
        _lambda=1, 
        iter_max=n_iter, 
        tv_weight=6, 
        tv_iter_max=5, 
        X_orig=truth
    )
    
    # 3. Save Results
    print("Saving results...")
    sio.savemat('recon_result.mat', {'img': recon_img})
    
    # Save spectral channels as image grid
    fig = plt.figure(figsize=(10, 10))
    # Plot first 9 channels
    for i in range(min(9, nC)):
        plt.subplot(3, 3, i+1)
        plt.imshow(recon_img[:, :, i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'Band {i+1}')
    plt.savefig('recon_channels.png')
    plt.close()
    
    # Save PSNR plot
    plt.figure()
    plt.plot(psnrs)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('Reconstruction Convergence')
    plt.savefig('psnr_curve.png')
    plt.close()
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
```