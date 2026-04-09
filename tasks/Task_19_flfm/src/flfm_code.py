import torch
import numpy as np
import matplotlib.pyplot as plt
import time


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
    psf = torch.zeros(shape, dtype=torch.float32)
    
    z_c = nz // 2
    y_c = ny // 2
    x_c = nx // 2
    
    y = torch.arange(ny) - y_c
    x = torch.arange(nx) - x_c
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R2 = X**2 + Y**2
    
    sigma0 = 2.0
    
    for z in range(nz):
        dist = abs(z - z_c)
        sigma = sigma0 + 0.1 * dist
        arg = -R2 / (2 * sigma**2)
        layer = torch.exp(arg)
        psf[z] = layer
    
    psf = psf / psf.sum()
    psf = psf.to(device)
    
    # --- Generate Ground Truth Object ---
    gt_obj = torch.zeros(shape, dtype=torch.float32)
    
    torch.manual_seed(42)
    
    num_beads = 10
    for _ in range(num_beads):
        z_idx = torch.randint(0, shape[0], (1,))
        y_idx = torch.randint(10, shape[1] - 10, (1,))
        x_idx = torch.randint(10, shape[2] - 10, (1,))
        gt_obj[z_idx, y_idx, x_idx] = 100.0
    
    gt_obj = gt_obj.to(device)
    
    # --- Simulate Measurement (Forward Model) ---
    measurement = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    
    for z in range(nz):
        layer_obj = gt_obj[z]
        layer_psf = psf[z]
        
        O_fft = torch.fft.rfft2(layer_obj)
        P_fft = torch.fft.rfft2(layer_psf)
        
        convolved = torch.fft.irfft2(O_fft * P_fft, s=(ny, nx))
        measurement += convolved
    
    # Add noise
    noise = torch.randn_like(measurement) * 0.1
    measurement = torch.relu(measurement + noise)
    
    # --- Precompute PSF FFTs ---
    psf_fft = torch.fft.rfft2(psf, dim=(-2, -1))
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
    est_fft = torch.fft.rfft2(estimate, dim=(-2, -1))
    product = est_fft * psf_fft
    
    layers = torch.fft.irfft2(product, dim=(-2, -1), s=estimate.shape[-2:])
    
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
    estimate = torch.ones((nz, ny, nx), device=device) * measurement.mean()
    
    # Ensure inputs are on correct device
    measurement = measurement.to(device)
    psf_fft = psf_fft.to(device)
    psft_fft = psft_fft.to(device)
    
    for i in range(num_iter):
        t0 = time.time()
        
        # 1. Forward Projection: H x
        proj = forward_operator(estimate, psf_fft)
        
        # 2. Compute Ratio (Error): y / H x
        ratio = measurement / (proj + 1e-8)
        
        # 3. Backward Projection: H^T (y / H x)
        # BackProj_z = Corr2D(Error, PSF_z) = Conv2D(Error, Flip(PSF_z))
        err_fft = torch.fft.rfft2(ratio, dim=(-2, -1))
        product = err_fft.unsqueeze(0) * psft_fft
        update_factor = torch.fft.irfft2(product, dim=(-2, -1), s=(ny, nx))
        
        # 4. Update: x_{k+1} = x_k * H^T (...)
        estimate = estimate * update_factor
        
        # Enforce non-negativity
        estimate = torch.relu(estimate)
        
        if i % 5 == 0:
            elapsed = time.time() - t0
            print(f"Iter {i}/{num_iter}: Time={elapsed:.4f}s")
    
    return estimate


def evaluate_results(reconstruction, ground_truth, save_figures=True):
    """
    Computes MSE and PSNR, and optionally saves visualization figures.
    
    Args:
        reconstruction: 3D Tensor [nz, ny, nx] - reconstructed volume
        ground_truth: 3D Tensor [nz, ny, nx] - ground truth volume
        save_figures: bool - whether to save visualization figures
    
    Returns:
        metrics: dict containing MSE and PSNR values
    """
    # Compute MSE
    mse = torch.mean((reconstruction - ground_truth) ** 2)
    
    # Compute PSNR
    max_val = ground_truth.max()
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    mse_val = mse.item()
    psnr_val = psnr.item()
    
    print(f"Result: MSE={mse_val:.6f}, PSNR={psnr_val:.2f} dB")
    
    if save_figures:
        # Max projection along Z
        recon_mip = reconstruction.max(dim=0)[0].cpu().numpy()
        gt_mip = ground_truth.max(dim=0)[0].cpu().numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gt_mip, cmap='inferno')
        ax[0].set_title("Ground Truth (MIP)")
        ax[1].imshow(recon_mip, cmap='inferno')
        ax[1].set_title(f"Reconstruction (MIP)\nPSNR={psnr_val:.2f}dB")
        plt.savefig("result_comparison.png")
        plt.close()
        print("Saved result_comparison.png")
    
    return {
        'mse': mse_val,
        'psnr': psnr_val
    }


if __name__ == '__main__':
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define parameters
    volume_shape = (20, 128, 128)
    num_iterations = 30
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(
        shape=volume_shape,
        pixel_size=6.5,
        wavelength=0.520,
        na=0.8,
        ref_index=1.33,
        device=device
    )
    
    measurement = data['measurement']
    psf_fft = data['psf_fft']
    psft_fft = data['psft_fft']
    ground_truth = data['ground_truth']
    
    # Save measurement visualization
    plt.figure()
    plt.imshow(measurement.cpu().numpy(), cmap='gray')
    plt.title("Simulated Measurement")
    plt.savefig("measurement.png")
    plt.close()
    print("Saved measurement.png")
    
    # 2. Run inversion
    print("Running Inversion...")
    reconstruction = run_inversion(
        measurement=measurement,
        psf_fft=psf_fft,
        psft_fft=psft_fft,
        shape=volume_shape,
        num_iter=num_iterations,
        device=device
    )
    
    # 3. Evaluate results
    print("Evaluating results...")
    metrics = evaluate_results(
        reconstruction=reconstruction,
        ground_truth=ground_truth,
        save_figures=True
    )
    
    print(f"Final MSE: {metrics['mse']:.6f}")
    print(f"Final PSNR: {metrics['psnr']:.2f} dB")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")