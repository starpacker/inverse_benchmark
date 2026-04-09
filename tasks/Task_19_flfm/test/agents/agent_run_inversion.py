import torch
import time

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
    # 1. Convert estimate to frequency domain
    est_fft = torch.fft.rfft2(estimate, dim=(-2, -1))
    
    # 2. Convolve (Element-wise multiplication)
    product = est_fft * psf_fft
    
    # 3. Convert back to spatial domain
    # Note: We specify 's' to ensure output shape matches input spatial dims
    layers = torch.fft.irfft2(product, dim=(-2, -1), s=estimate.shape[-2:])
    
    # 4. Sum along Z-axis to simulate sensor integration
    proj_image = layers.sum(dim=0)
    
    return proj_image

def run_inversion(measurement, psf_fft, psft_fft, shape, num_iter=20, device='cpu'):
    """
    Runs the Richardson-Lucy deconvolution loop.
    
    Args:
        measurement: 2D Tensor [ny, nx] - observed image
        psf_fft: 3D Tensor [nz, ny, nx/2+1] - FFT of PSF
        psft_fft: 3D Tensor [nz, ny, nx/2+1] - FFT of time-reversed PSF
        shape: tuple (nz, ny, nx) - shape of the 3D volume
        num_iter: int - number of iterations
        device: str - computation device
    
    Returns:
        estimate: 3D Tensor [nz, ny, nx] - reconstructed volume
    """
    print(f"Starting Inversion on {device}...")
    
    nz, ny, nx = shape
    
    # Initial Guess: Flat 3D object
    # We initialize with the mean value to scale the energy correctly
    estimate = torch.ones((nz, ny, nx), device=device) * measurement.mean()
    
    # Ensure inputs are on correct device
    measurement = measurement.to(device)
    psf_fft = psf_fft.to(device)
    psft_fft = psft_fft.to(device)
    
    for i in range(num_iter):
        t0 = time.time()
        
        # 1. Forward Projection: H * x
        # Simulate the image based on current estimate
        proj = forward_operator(estimate, psf_fft)
        
        # 2. Compute Ratio (Error): y / (H * x)
        # Small epsilon added to denominator for numerical stability
        ratio = measurement / (proj + 1e-8)
        
        # 3. Backward Projection: H^T * (y / H * x)
        # This distributes the 2D error map back into 3D space
        err_fft = torch.fft.rfft2(ratio, dim=(-2, -1))
        
        # Broadcast 2D error spectrum against 3D PSF spectrum
        product = err_fft.unsqueeze(0) * psft_fft
        
        # Inverse FFT to get the multiplicative update factor
        update_factor = torch.fft.irfft2(product, dim=(-2, -1), s=(ny, nx))
        
        # 4. Update: x_{k+1} = x_k * UpdateFactor
        estimate = estimate * update_factor
        
        # Enforce non-negativity constraint (Physics constraint)
        estimate = torch.relu(estimate)
        
        if i % 5 == 0:
            elapsed = time.time() - t0
            print(f"Iter {i}/{num_iter}: Time={elapsed:.4f}s")
    
    return estimate