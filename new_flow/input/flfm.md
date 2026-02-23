```python
import torch
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from pathlib import Path
import time

# --- 1. Explicit Data Generation (Preprocessing) ---

def generate_psf(
    shape=(20, 128, 128), 
    pixel_size=6.5, 
    wavelength=0.520, 
    na=0.8, 
    ref_index=1.33
):
    """
    Generates a simulated 3D PSF using a Gaussian approximation of the Airy disk,
    following the logic from the FLFM notebook.
    """
    nz, ny, nx = shape
    psf = torch.zeros(shape, dtype=torch.float32)
    
    # Parameters from notebook approximation
    # omega_0 approx 0.42 * lambda * sqrt((n/NA)^2 - 1)
    # This is a simplification; for a full demo we use a 3D Gaussian where
    # sigma increases with distance from focal plane (z_c).
    
    z_c = nz // 2
    y_c = ny // 2
    x_c = nx // 2
    
    # Grid
    y = torch.arange(ny) - y_c
    x = torch.arange(nx) - x_c
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R2 = X**2 + Y**2
    
    # Simple model: Gaussian beam width w(z) = w0 * sqrt(1 + (z/zR)^2)
    # But for this sandbox, let's just use a fixed Gaussian per slice 
    # but slightly blurring away from center to simulate defocus.
    
    # Using the formula from notebook for sigma (converted to pixels)
    # pixelSpacing is roughly pixel_size / M. Let's assume M=1 for simplicity in sandbox.
    # sigma_0 = (0.42 * wavelength * np.sqrt((ref_index / na)**2 - 1)) / (pixel_size/20) # scaling
    
    # Heuristic sigma for visualization
    sigma0 = 2.0 
    
    for z in range(nz):
        # Defocus factor
        dist = abs(z - z_c)
        sigma = sigma0 + 0.1 * dist # Broadens away from center
        
        arg = -R2 / (2 * sigma**2)
        layer = torch.exp(arg)
        psf[z] = layer
        
    # Normalize PSF
    psf = psf / psf.sum()
    return psf

def generate_ground_truth(shape=(20, 128, 128)):
    """
    Generates a sparse 3D ground truth object (e.g., fluorescent beads).
    """
    obj = torch.zeros(shape, dtype=torch.float32)
    
    # Place some random beads
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    
    num_beads = 10
    for _ in range(num_beads):
        z = torch.randint(0, shape[0], (1,))
        y = torch.randint(10, shape[1]-10, (1,))
        x = torch.randint(10, shape[2]-10, (1,))
        obj[z, y, x] = 100.0
        
    return obj

def preprocess_data(gt_obj, psf):
    """
    Simulates the forward process to get the 'observed' image (measurement).
    Preprocessing involves:
    1. Generating measurement (Forward Project GT).
    2. Adding noise.
    3. Computing FFT of PSF for the inversion.
    """
    # 1. Forward Project to get 2D measurement (or 3D stack if FLFM is treated as 3D deconv)
    # In FLFM notebook, they do "3D-to-2D deconv" logic but the code RL_3Ddeconv 
    # iterates over slices. 
    # Actually, FLFM usually captures a 2D image (lenslet array) encoding 3D info.
    # But the notebook's RL_3Ddeconv function signature is:
    # def RL_3Ddeconv(img, FFT_PSF, rec_vol, FFT_PSF_inv_norm, n_iter):
    # It takes a 2D image 'img' and 3D 'FFT_PSF'.
    
    # Let's simulate the 2D measurement.
    # Measurement = sum_z ( Conv2D(Obj_z, PSF_z) )
    
    nz, ny, nx = gt_obj.shape
    measurement = torch.zeros((ny, nx), dtype=torch.float32, device=gt_obj.device)
    
    # Explicit Forward Model for Simulation
    for z in range(nz):
        layer_obj = gt_obj[z]
        layer_psf = psf[z]
        
        # Convolve using FFT
        # Pad to avoid wrap-around artifacts (simplified here: no padding, just circular conv)
        O_fft = torch.fft.rfft2(layer_obj)
        P_fft = torch.fft.rfft2(layer_psf)
        
        convolved = torch.fft.irfft2(O_fft * P_fft, s=(ny, nx))
        measurement += convolved
        
    # 2. Add Noise (Poisson + Gaussian)
    # Simple Gaussian for now
    noise = torch.randn_like(measurement) * 0.1
    measurement = torch.relu(measurement + noise) # Non-negative
    
    # 3. Precompute PSF FFTs for Deconvolution
    # The reconstruction needs the FFT of the PSF stack.
    # Shape: [nz, ny, nx]
    psf_fft = torch.fft.rfft2(psf, dim=(-2, -1))
    
    # Also need time-reversed PSF FFT (complex conjugate of FFT(PSF) usually, or FFT(flip(PSF)))
    # For real PSF, FFT(flip(PSF)) is complex conjugate of FFT(PSF) * phase shift?
    # flfm/backend/torch.py uses:
    # psft_fft = torch.fft.rfft2(torch.flip(psf, (-2, -1)))
    psft_fft = torch.fft.rfft2(torch.flip(psf, dims=(-2, -1)), dim=(-2, -1))
    
    return measurement, psf_fft, psft_fft

# --- 2. Explicit Forward and Backward Operators ---

def forward_operator(estimate, psf_fft):
    """
    Computes the forward projection of the current 3D estimate.
    Projected_Image = Sum_z ( Conv2D(Estimate_z, PSF_z) )
    
    Args:
        estimate: 3D Tensor [nz, ny, nx]
        psf_fft: 3D Tensor [nz, ny, nx/2+1] (FFT of PSF)
    Returns:
        proj_image: 2D Tensor [ny, nx]
    """
    # Convolution in Freq domain: Multiply
    est_fft = torch.fft.rfft2(estimate, dim=(-2, -1))
    product = est_fft * psf_fft
    
    # Inverse FFT
    layers = torch.fft.irfft2(product, dim=(-2, -1), s=estimate.shape[-2:])
    
    # Sum over Z to get 2D image
    proj_image = layers.sum(dim=0)
    
    return proj_image

def backward_operator(error_image, psft_fft):
    """
    Computes the backward projection of the error image.
    BackProj_z = Corr2D(Error, PSF_z) 
               = Conv2D(Error, Flip(PSF_z))
    
    Args:
        error_image: 2D Tensor [ny, nx]
        psft_fft: 3D Tensor [nz, ny, nx/2+1] (FFT of Time-Reversed PSF)
    Returns:
        back_proj: 3D Tensor [nz, ny, nx]
    """
    # Expand error to match z-dim for broadcasting (or just repeat)
    # Actually we just broadcast the 2D error fft against all z slices of psf
    
    err_fft = torch.fft.rfft2(error_image, dim=(-2, -1))
    
    # Multiply: [ny, nx/2+1] * [nz, ny, nx/2+1] -> [nz, ny, nx/2+1]
    product = err_fft.unsqueeze(0) * psft_fft
    
    # Inverse FFT
    back_proj = torch.fft.irfft2(product, dim=(-2, -1), s=error_image.shape[-2:])
    
    return back_proj

# --- 3. Run Inversion (Richardson-Lucy) ---

def run_inversion(measurement, psf_fft, psft_fft, num_iter=20, device='cpu'):
    """
    Runs the Richardson-Lucy deconvolution loop.
    """
    print(f"Starting Inversion on {device}...")
    
    nz = psf_fft.shape[0]
    ny = measurement.shape[0]
    nx = measurement.shape[1]
    
    # Initial Guess: Flat 3D object
    estimate = torch.ones((nz, ny, nx), device=device) * measurement.mean()
    
    # Move inputs to device
    measurement = measurement.to(device)
    psf_fft = psf_fft.to(device)
    psft_fft = psft_fft.to(device)
    
    # Loop
    for i in range(num_iter):
        t0 = time.time()
        
        # 1. Forward Projection
        # H x
        proj = forward_operator(estimate, psf_fft)
        
        # 2. Compute Ratio (Error)
        # y / H x
        # Avoid division by zero
        ratio = measurement / (proj + 1e-8)
        
        # 3. Backward Projection
        # H^T (y / H x)
        update_factor = backward_operator(ratio, psft_fft)
        
        # 4. Update
        # x_{k+1} = x_k * H^T (...)
        estimate = estimate * update_factor
        
        # Enforce non-negativity (RL implies it if start > 0, but good to be safe)
        estimate = torch.relu(estimate)
        
        if i % 5 == 0:
            print(f"Iter {i}/{num_iter}: Time={time.time()-t0:.4f}s")
            
    return estimate

# --- 4. Evaluation ---

def evaluate_results(reconstruction, ground_truth):
    """
    Computes MSE and PSNR.
    """
    mse = torch.mean((reconstruction - ground_truth)**2)
    max_val = ground_truth.max()
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    return mse.item(), psnr.item()

# --- Main ---

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Generate Data
    print("Generating simulated data...")
    psf = generate_psf().to(device)
    gt_obj = generate_ground_truth().to(device)
    
    # 2. Preprocess (Simulate measurement)
    print("Preprocessing (Simulating measurement)...")
    measurement, psf_fft, psft_fft = preprocess_data(gt_obj, psf)
    
    # Save inputs
    plt.figure()
    plt.imshow(measurement.cpu().numpy(), cmap='gray')
    plt.title("Simulated Measurement")
    plt.savefig("measurement.png")
    plt.close()
    
    # 3. Run Inversion
    print("Running Inversion...")
    reconstruction = run_inversion(measurement, psf_fft, psft_fft, num_iter=30, device=device)
    
    # 4. Evaluate
    print("Evaluating...")
    mse, psnr = evaluate_results(reconstruction, gt_obj)
    print(f"Result: MSE={mse:.6f}, PSNR={psnr:.2f} dB")
    
    # 5. Visualize
    # Max projection along Z
    recon_mip = reconstruction.max(dim=0)[0].cpu().numpy()
    gt_mip = gt_obj.max(dim=0)[0].cpu().numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(gt_mip, cmap='inferno')
    ax[0].set_title("Ground Truth (MIP)")
    ax[1].imshow(recon_mip, cmap='inferno')
    ax[1].set_title(f"Reconstruction (MIP)\nPSNR={psnr:.2f}dB")
    plt.savefig("result_comparison.png")
    print("Saved result_comparison.png")

if __name__ == "__main__":
    main()
```
