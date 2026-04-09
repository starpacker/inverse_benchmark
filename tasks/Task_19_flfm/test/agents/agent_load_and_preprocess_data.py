import torch

def load_and_preprocess_data(
    shape=(20, 128, 128),
    pixel_size=6.5,
    wavelength=0.520,
    na=0.8,
    ref_index=1.33,
    device='cpu'
):
    """
    Generates simulated data including PSF, ground truth object, and measurement.
    Returns all necessary data for the inversion.
    """
    nz, ny, nx = shape
    
    # --- Generate PSF ---
    # We approximate the PSF as a stack of 2D Gaussians.
    # The width (sigma) of the Gaussian increases as we move away from the center z-plane.
    psf = torch.zeros(shape, dtype=torch.float32)
    
    z_c = nz // 2
    y_c = ny // 2
    x_c = nx // 2
    
    # Create spatial grid
    y = torch.arange(ny) - y_c
    x = torch.arange(nx) - x_c
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R2 = X**2 + Y**2
    
    sigma0 = 2.0 # Base width at focal plane
    
    for z in range(nz):
        dist = abs(z - z_c)
        # Linear broadening of PSF with defocus
        sigma = sigma0 + 0.1 * dist
        arg = -R2 / (2 * sigma**2)
        layer = torch.exp(arg)
        psf[z] = layer
    
    # Normalize energy
    psf = psf / psf.sum()
    psf = psf.to(device)
    
    # --- Generate Ground Truth Object ---
    # Create a sparse object with random bright beads
    gt_obj = torch.zeros(shape, dtype=torch.float32)
    
    torch.manual_seed(42) # Fix seed for reproducibility
    
    num_beads = 10
    for _ in range(num_beads):
        z_idx = torch.randint(0, shape[0], (1,))
        # Avoid edges to prevent boundary artifacts
        y_idx = torch.randint(10, shape[1] - 10, (1,))
        x_idx = torch.randint(10, shape[2] - 10, (1,))
        gt_obj[z_idx, y_idx, x_idx] = 100.0
    
    gt_obj = gt_obj.to(device)
    
    # --- Simulate Measurement (Forward Model) ---
    # M = sum_z (Obj_z * PSF_z)
    measurement = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    
    for z in range(nz):
        layer_obj = gt_obj[z]
        layer_psf = psf[z]
        
        # Convolution via FFT
        O_fft = torch.fft.rfft2(layer_obj)
        P_fft = torch.fft.rfft2(layer_psf)
        
        convolved = torch.fft.irfft2(O_fft * P_fft, s=(ny, nx))
        measurement += convolved
    
    # Add Gaussian noise and clip negative values
    noise = torch.randn_like(measurement) * 0.1
    measurement = torch.relu(measurement + noise)
    
    # --- Precompute PSF FFTs ---
    # These are cached for the iterative reconstruction loop
    
    # Forward kernel
    psf_fft = torch.fft.rfft2(psf, dim=(-2, -1))
    
    # Adjoint kernel (flipped PSF)
    # Flipping is required because correlation is convolution with a flipped kernel
    psft_fft = torch.fft.rfft2(torch.flip(psf, dims=(-2, -1)), dim=(-2, -1))
    
    return {
        'measurement': measurement,
        'psf_fft': psf_fft,
        'psft_fft': psft_fft,
        'ground_truth': gt_obj,
        'psf': psf,
        'shape': shape,
        'device': device
    }