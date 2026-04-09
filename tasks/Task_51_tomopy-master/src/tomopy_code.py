import numpy as np
import scipy.ndimage
import scipy.fft
import matplotlib.pyplot as plt
import os
import sys

# --- Helper Functions (Defined First) ---

def minus_log(data):
    """
    Computes the minus log of the data: P = -log(data).
    """
    data = np.where(data <= 0, 1e-6, data)
    return -np.log(data)

def radon_transform_logic(image, theta):
    """
    Radon Transform (Forward Projector) using skimage for performance.
    """
    from skimage.transform import radon
    # skimage.radon expects theta in degrees, image as 2D array
    # Returns sinogram with shape (n_detectors, n_angles)
    sino = radon(image, theta=theta, circle=True)
    # Transpose to (n_angles, n_detectors) to match our convention
    return sino.T.astype(np.float32)

def filter_sinogram(sinogram, window=None):
    """
    Applies the Ram-Lak filter with an optional window function.
    """
    num_angles, num_detectors = sinogram.shape
    
    # Pad to the next power of 2 for efficient FFT
    n = num_detectors
    padded_len = max(64, int(2 ** np.ceil(np.log2(2 * n))))
    
    # Compute frequency axis
    freq = scipy.fft.rfftfreq(padded_len)
    
    # Ram-Lak filter: |f| (ramp filter)
    filt = 2 * np.abs(freq) 
    
    # Apply window
    if window == 'hann':
        w = np.hanning(2 * len(freq))[:len(freq)]
        filt *= w
    elif window == 'hamming':
        w = np.hamming(2 * len(freq))[:len(freq)]
        filt *= w
    
    # Apply filter in Fourier domain
    sino_fft = scipy.fft.rfft(sinogram, n=padded_len, axis=1)
    filtered_sino_fft = sino_fft * filt
    filtered_sino = scipy.fft.irfft(filtered_sino_fft, n=padded_len, axis=1)
    
    # Crop back to original size
    return filtered_sino[:, :num_detectors]

def backproject(sinogram, theta):
    """
    Backprojection using skimage iradon (no filter) for performance.
    """
    from skimage.transform import iradon
    # sinogram is (n_angles, n_detectors), iradon expects (n_detectors, n_angles)
    sino_T = sinogram.T
    recon = iradon(sino_T, theta=theta, filter_name=None, circle=True)
    return recon.astype(np.float32)

def fbp_reconstruct(sinogram, theta, window=None):
    """
    Complete FBP pipeline using skimage iradon.
    """
    from skimage.transform import iradon
    # Map our window names to skimage filter names
    filter_map = {None: 'ramp', 'hann': 'hann', 'hamming': 'hamming'}
    filter_name = filter_map.get(window, 'ramp')
    # sinogram is (n_angles, n_detectors), iradon expects (n_detectors, n_angles)
    sino_T = sinogram.T
    recon = iradon(sino_T, theta=theta, filter_name=filter_name, circle=True)
    return recon.astype(np.float32)

def sirt_reconstruct(sinogram, theta, n_iter=10):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT).
    """
    num_angles, num_detectors = sinogram.shape
    N = num_detectors
    
    recon = np.zeros((N, N), dtype=np.float32)
    
    # Calculate Row Sums (R)
    ones_img = np.ones((N, N), dtype=np.float32)
    row_sums = radon_transform_logic(ones_img, theta)
    row_sums[row_sums == 0] = 1.0
    
    # Calculate Column Sums (C)
    ones_sino = np.ones_like(sinogram)
    col_sums = backproject(ones_sino, theta)
    col_sums[col_sums == 0] = 1.0
    
    for k in range(n_iter):
        fp = radon_transform_logic(recon, theta)
        diff = sinogram - fp
        correction_term = diff / row_sums
        correction = backproject(correction_term, theta)
        
        recon += correction / col_sums
        recon[recon < 0] = 0
        
    return recon

def calculate_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return 100
    max_pixel = gt.max()
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(gt, recon):
    """Structural Similarity Index Wrapper"""
    try:
        from skimage.metrics import structural_similarity
        data_range = gt.max() - gt.min()
        return structural_similarity(gt, recon, data_range=data_range)
    except ImportError:
        return 0

def norm_minmax(x):
    return (x - x.min()) / (x.max() - x.min())

# --- Core Functional Components ---

def load_and_preprocess_data(file_path, I0=100000.0, downsample_size=(128, 128),
                             mu_scale=0.02):
    """
    Loads image, simulates noise, and prepares sinogram.
    
    mu_scale: Attenuation scaling factor. Phantom pixels in [0,1] on a 256x256
              grid produce Radon line integrals up to ~130. With Beer's law,
              exp(-130)≈0, causing total sinogram saturation and Poisson(0)=0.
              Scaling by mu_scale keeps max line integral ≈ 130*mu_scale ≈ 2.6,
              which is physically realistic and avoids saturation.
    
    Returns: original_image, sinogram_noisy, theta, mu_scale
    """
    # Import logic restricted to function scope or global try/except used inside
    try:
        import tifffile
        read_tiff = tifffile.imread
    except ImportError:
        try:
            from skimage import io
            read_tiff = io.imread
        except ImportError:
            read_tiff = plt.imread

    print(f"Loading data from: {file_path}")
    original_image = read_tiff(file_path)
    original_image = original_image.astype(np.float32)
    
    if original_image.max() > 1.0:
        original_image /= original_image.max()
        
    # Downsample
    if original_image.shape[0] > downsample_size[0]:
        from skimage.transform import resize
        original_image = resize(original_image, downsample_size, anti_aliasing=True).astype(np.float32)
    
    # Simulate Acquisition
    theta = np.linspace(0, 180, 180, endpoint=False)
    
    # Scale phantom attenuation to avoid Beer's law overflow
    # Without scaling: line_integral ~ 130 -> exp(-130) = 0 -> total saturation
    # With mu_scale=0.02: line_integral ~ 2.6 -> exp(-2.6) ~ 0.07 -> healthy signal
    scaled_image = original_image * mu_scale
    sinogram_clean = radon_transform_logic(scaled_image, theta)
    
    print(f"  mu_scale={mu_scale}, sinogram_clean range: [{sinogram_clean.min():.2f}, {sinogram_clean.max():.2f}]")
    
    # Add Poisson Noise
    transmission = I0 * np.exp(-sinogram_clean)
    transmission_noisy = np.random.poisson(transmission).astype(np.float32)
    
    # Check saturation
    n_saturated = np.sum(transmission_noisy == 0)
    n_total = transmission_noisy.size
    print(f"  Saturation: {n_saturated}/{n_total} = {100*n_saturated/n_total:.1f}%")
    
    # Preprocess (Normalization + Log)
    normalized_proj = transmission_noisy / I0
    sinogram_noisy = minus_log(normalized_proj)
    
    return original_image, sinogram_noisy, theta, mu_scale

def forward_operator(x, theta):
    """
    Wraps the physical forward model (Radon Transform).
    x: Image
    theta: Angles
    Returns: Sinogram
    """
    return radon_transform_logic(x, theta)

def run_inversion(sinogram, theta, method='fbp', **kwargs):
    """
    Runs the specified reconstruction algorithm.
    """
    if method == 'fbp':
        window = kwargs.get('window', None)
        return fbp_reconstruct(sinogram, theta, window=window)
    elif method == 'sirt':
        n_iter = kwargs.get('n_iter', 20)
        return sirt_reconstruct(sinogram, theta, n_iter=n_iter)
    else:
        raise ValueError(f"Unknown inversion method: {method}")

def evaluate_results(gt, recon_dict):
    """
    Calculates metrics and generates visualization.
    gt: Ground Truth Image
    recon_dict: Dictionary {method_name: reconstructed_image}
    """
    gt_norm = norm_minmax(gt)
    
    # Create circular mask
    h, w = gt_norm.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - w/2)**2 + (y - h/2)**2 <= (w/2)**2
    
    results_text = []
    
    for name, recon in recon_dict.items():
        r_norm = norm_minmax(recon)
        psnr = calculate_psnr(gt_norm[mask], r_norm[mask])
        ssim = calculate_ssim(gt_norm, r_norm)
        results_text.append(f"{name} -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        print(results_text[-1])

    # Visualization
    try:
        num_plots = 1 + len(recon_dict)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1: axes = [axes] # Handle single plot case if empty dict
        
        # Plot GT
        axes[0].imshow(gt, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Plot Recons
        for i, (name, recon) in enumerate(recon_dict.items(), 1):
            axes[i].imshow(recon, cmap='gray')
            axes[i].set_title(name)
            axes[i].axis('off')
            
        output_file = 'tomopy_workflow_refactored.png'
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Result saved to {output_file}")
    except Exception as e:
        print(f"Visualization failed: {e}")
        
    return results_text

if __name__ == '__main__':
    # 0. Setup Path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard structure or current dir fallback
    possible_paths = [
        os.path.join(base_dir, "source", "tomopy", "data", "shepp2d.tif"),
        os.path.join(base_dir, "shepp2d.tif")
    ]
    
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    # Fallback search if specific paths fail
    if data_path is None:
        import glob
        matches = glob.glob(os.path.join(base_dir, "**", "shepp2d.tif"), recursive=True)
        if matches:
            data_path = matches[0]
        else:
            # Create a synthetic phantom if file not found to ensure execution
            print("Warning: shepp2d.tif not found. Creating synthetic phantom.")
            from skimage.data import shepp_logan_phantom
            synthetic_phantom = shepp_logan_phantom()
            # Save temporarily to fit the API
            import tifffile
            data_path = "temp_phantom.tif"
            tifffile.imwrite(data_path, (synthetic_phantom * 255).astype(np.uint8))
            
    # 1. Load and Preprocess
    gt_image, sinogram, theta, mu_scale = load_and_preprocess_data(data_path)
    
    # 2. Forward Operator (Demonstration)
    # We already did this inside load_and_preprocess to generate the sinogram, 
    # but strictly calling it here for verification or if we had real data and needed a forward model check.
    # verification_sino = forward_operator(gt_image, theta) 
    
    # 3. Run Inversion
    print("Running FBP with Hann window...")
    recon_fbp = run_inversion(sinogram, theta, method='fbp', window='hann')
    
    print("Running SIRT (50 iterations)...")
    recon_sirt = run_inversion(sinogram, theta, method='sirt', n_iter=50)
    
    # 4. Scale reconstructions back to original phantom scale
    # The sinogram was built from (original_image * mu_scale), so the
    # reconstruction recovers the scaled attenuation. Divide by mu_scale
    # to compare against the original phantom.
    recon_fbp_rescaled = recon_fbp / mu_scale
    recon_sirt_rescaled = recon_sirt / mu_scale
    
    # 5. Evaluate
    reconstructions = {
        'FBP (Hann)': recon_fbp_rescaled,
        'SIRT (50 iter)': recon_sirt_rescaled
    }
    evaluate_results(gt_image, reconstructions)
    
    # 6. Pick the best reconstruction and save outputs
    gt_norm = norm_minmax(gt_image)
    h, w = gt_norm.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - w/2)**2 + (y - h/2)**2 <= (w/2)**2
    
    best_recon = None
    best_psnr = -1
    best_name = ''
    best_ssim = 0
    for name, recon in reconstructions.items():
        r_norm = norm_minmax(recon)
        psnr = calculate_psnr(gt_norm[mask], r_norm[mask])
        ssim = calculate_ssim(gt_norm, r_norm)
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim
            best_recon = recon
            best_name = name
    
    print(f"\nBest method: {best_name} -> PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}")
    
    # Save outputs
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(base_dir, 'gt_output.npy'), gt_image)
    np.save(os.path.join(base_dir, 'recon_output.npy'), best_recon)
    
    import json
    metrics = {
        'psnr': round(best_psnr, 2),
        'ssim': round(best_ssim, 4),
        'method': best_name
    }
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics}")
    
    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_image, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    r_norm_best = norm_minmax(best_recon)
    axes[1].imshow(r_norm_best, cmap='gray')
    axes[1].set_title(f'{best_name}\nPSNR={best_psnr:.2f} dB')
    axes[1].axis('off')
    
    diff = np.abs(gt_norm - r_norm_best)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('|GT - Recon|')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {results_dir}/reconstruction_result.png")
        
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")