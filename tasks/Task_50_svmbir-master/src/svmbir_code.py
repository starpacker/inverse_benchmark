import numpy as np
import scipy.ndimage
import sys
import time
import matplotlib.pyplot as plt

# Try importing optional dependencies for metrics/transforms
try:
    from skimage.transform import radon, iradon
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using slower fallback implementations.")

# ==============================================================================
# Helper Functions (Phantoms, Math, Priors)
# ==============================================================================

def _gen_ellipse(x_grid, y_grid, x0, y0, a, b, gray_level, theta=0):
    """Generates a single ellipse mask scaled by gray_level."""
    c = np.cos(theta)
    s = np.sin(theta)
    x_rot = (x_grid - x0) * c + (y_grid - y0) * s
    y_rot = -(x_grid - x0) * s + (y_grid - y0) * c
    mask = (x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2) <= 1.0
    return mask * gray_level

def _gen_shepp_logan(num_rows, num_cols):
    """Generates the Shepp-Logan phantom."""
    sl_paras = [
        {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': -0.98},
        {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': -0.02},
        {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.01}
    ]
    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    x_grid, y_grid = np.meshgrid(axis_x, axis_y)
    image = np.zeros_like(x_grid)
    for el in sl_paras:
        image += _gen_ellipse(x_grid, y_grid, el['x0'], el['y0'], el['a'], el['b'], 
                              el['gray_level'], el['theta'] / 180.0 * np.pi)
    return image

def _qggmrf_derivative(delta, sigma_x, p, q, T):
    """
    Computes the derivative of the Q-GGMRF potential function w.r.t delta.
    Approximation logic preserved from input code.
    """
    abs_d = np.abs(delta) + 1e-6 
    sign_d = np.sign(delta)
    u = abs_d / T
    
    # rho(x) = |x|^p / (1 + |x/T|^(p-q))
    # Using the derivation logic provided in the input context:
    num = abs_d ** p
    den = 1 + u ** (p - q)
    
    d_num = p * abs_d ** (p - 1) * sign_d
    d_den = (p - q) * (u ** (p - q - 1)) * (1.0/T) * sign_d
    
    grad = (d_num * den - num * d_den) / (den ** 2)
    return grad / (sigma_x ** p)

def _compute_prior_gradient(image, sigma_x, p, q, T):
    """Computes the gradient of the Q-GGMRF prior (Markov Random Field)."""
    total_grad = np.zeros_like(image)
    
    # Right neighbor interaction: x[r,c] - x[r,c+1]
    d = image - np.roll(image, -1, axis=1)
    d[:, -1] = 0 # Boundary
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    # Left neighbor interaction: x[r,c] - x[r,c-1]
    d = image - np.roll(image, 1, axis=1)
    d[:, 0] = 0
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    # Down neighbor interaction: x[r,c] - x[r+1,c]
    d = image - np.roll(image, -1, axis=0)
    d[-1, :] = 0
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    # Up neighbor interaction: x[r,c] - x[r-1,c]
    d = image - np.roll(image, 1, axis=0)
    d[0, :] = 0
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    return total_grad

def _back_project_internal(sinogram, angles, image_shape):
    """
    Adjoint of the forward projector.
    """
    if HAS_SKIMAGE:
        # skimage iradon takes (num_det, num_angles)
        sino_T = sinogram.T
        theta_deg = np.degrees(angles)
        # filter_name=None implies unfiltered backprojection (adjoint of Radon)
        recon = iradon(sino_T, theta=theta_deg, output_size=image_shape[0], filter_name=None, circle=True)
        return recon
    else:
        # Manual backprojection
        num_rows, num_cols = image_shape
        backproj = np.zeros(image_shape, dtype=np.float32)
        for i, angle_rad in enumerate(angles):
            angle_deg = np.degrees(angle_rad)
            projection = sinogram[i, :]
            smeared = np.tile(projection, (num_rows, 1))
            rotated_back = scipy.ndimage.rotate(smeared, -angle_deg, reshape=False, order=1, mode='constant', cval=0.0)
            backproj += rotated_back
        return backproj

# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_and_preprocess_data(image_size, num_views, noise_level=0.01):
    """
    Generates synthetic Shepp-Logan phantom data and creates a noisy sinogram.
    
    Returns:
        gt_image (np.array): Ground truth image.
        sinogram (np.array): Observed noisy sinogram.
        angles (np.array): Projection angles in radians.
    """
    # 1. Define geometry
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    
    # 2. Generate Ground Truth
    gt_image = _gen_shepp_logan(image_size, image_size)
    
    # 3. Simulate "Clean" Sinogram using Forward Operator
    # Note: We call forward_operator locally here to simulate data creation.
    clean_sino = forward_operator(gt_image, angles)
    
    # 4. Add Noise
    noise = np.random.normal(0, noise_level * np.max(clean_sino), clean_sino.shape)
    sinogram = clean_sino + noise
    
    return gt_image, sinogram, angles

# ==============================================================================
# 2. Forward Operator
# ==============================================================================

def forward_operator(x, angles):
    """
    Computes the Radon Transform (Forward Projection).
    
    Args:
        x (np.array): Input image (N, N).
        angles (np.array): Projection angles in radians.
        
    Returns:
        y_pred (np.array): Sinogram (num_angles, num_detectors).
    """
    if HAS_SKIMAGE:
        theta_deg = np.degrees(angles)
        # skimage returns (num_det, num_angles), we want (num_angles, num_det)
        sino = radon(x, theta=theta_deg, circle=True)
        return sino.T
    else:
        num_angles = len(angles)
        num_rows, num_cols = x.shape
        num_det = num_cols 
        sinogram = np.zeros((num_angles, num_det), dtype=np.float32)
        
        for i, angle_rad in enumerate(angles):
            angle_deg = np.degrees(angle_rad)
            rotated_img = scipy.ndimage.rotate(x, angle_deg, reshape=False, order=1, mode='constant', cval=0.0)
            sinogram[i, :] = rotated_img.sum(axis=0)
        return sinogram

# ==============================================================================
# 3. Run Inversion
# ==============================================================================

def run_inversion(sinogram, angles, num_iters=100, step_size=0.001, 
                  p=1.2, q=2.0, T=1.0, sigma_x=None):
    """
    Performs MBIR reconstruction using Gradient Descent with Momentum and Q-GGMRF prior.
    
    Args:
        sinogram: Observed data.
        angles: Projection angles.
        num_iters: Number of iterations.
        step_size: Gradient descent step size (alpha).
        p, q, T: Prior parameters.
        sigma_x: Scaling parameter for prior.
    
    Returns:
        result (np.array): Reconstructed image.
    """
    num_angles, num_det = sinogram.shape
    img_shape = (num_det, num_det)
    
    # Heuristic for sigma_x if not provided
    if sigma_x is None:
        val = np.mean(np.abs(sinogram))
        sigma_x = 0.2 * val
    
    # Initialization: Normalized Backprojection
    bp = _back_project_internal(sinogram, angles, img_shape)
    x = bp / num_angles # Initial scaling guess
    
    # Momentum initialization
    velocity = np.zeros_like(x)
    mu = 0.9 # Momentum coefficient
    
    print(f"Starting Inversion: {num_iters} iters, sigma_x={sigma_x:.3f}")
    
    for k in range(num_iters):
        # 1. Forward Project
        Ax = forward_operator(x, angles)
        
        # 2. Residual (Ax - y)
        residual = Ax - sinogram 
        
        # 3. Data Gradient: A.T * (Ax - y)
        grad_data = _back_project_internal(residual, angles, img_shape)
        
        # 4. Prior Gradient
        grad_prior = _compute_prior_gradient(x, sigma_x, p, q, T)
        
        # 5. Total Gradient = Data Term + Prior Term
        total_grad = grad_data + grad_prior
        
        # 6. Update (Momentum + Gradient Descent)
        velocity = mu * velocity - step_size * total_grad
        x = x + velocity
        
        # 7. Positivity Constraint
        x[x < 0] = 0
        
        # Optional: Monitoring
        if k % 20 == 0:
            data_cost = 0.5 * np.sum(residual ** 2)
            print(f"  Iter {k}: Data Cost {data_cost:.2e}")
            
    return x

# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(gt, recon, save_path="reconstruction_result.png"):
    """
    Computes PSNR/SSIM and saves a comparison plot.
    
    Returns:
        metrics (dict): Dictionary containing PSNR and SSIM.
    """
    # Normalize for fair metric calculation
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn == 0: return arr
        return (arr - mn) / (mx - mn)

    gt_norm = normalize(gt)
    recon_norm = normalize(recon)
    
    # PSNR
    mse = np.mean((gt_norm - recon_norm) ** 2)
    if mse == 0:
        psnr_val = 100.0
    else:
        psnr_val = 20 * np.log10(1.0 / np.sqrt(mse))
        
    # SSIM
    ssim_val = 0.0
    if HAS_SKIMAGE:
        ssim_val = ssim_func(gt_norm, recon_norm, data_range=1.0)
    
    print(f"Evaluation -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    
    # Visualization
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gt, cmap='gray')
        ax[0].set_title("Ground Truth")
        ax[0].axis('off')
        
        ax[1].imshow(recon, cmap='gray')
        ax[1].set_title(f"Reconstruction\nPSNR: {psnr_val:.1f}")
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Figure saved to {save_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    return {"psnr": psnr_val, "ssim": ssim_val}

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == '__main__':
    # Configuration
    IMG_SIZE = 128
    NUM_VIEWS = 180
    ITERATIONS = 100
    STEP_SIZE = 0.001
    
    # 1. Load Data
    print("Step 1: Loading and preprocessing data...")
    gt, sino, angles = load_and_preprocess_data(IMG_SIZE, NUM_VIEWS)
    
    # 2. Check Forward Operator (Sanity Check)
    print("Step 2: verifying forward operator...")
    test_proj = forward_operator(gt, angles)
    if test_proj.shape != sino.shape:
        raise ValueError("Forward operator shape mismatch.")
    
    # 3. Run Inversion
    print("Step 3: Running inversion...")
    reconstruction = run_inversion(sino, angles, num_iters=ITERATIONS, step_size=STEP_SIZE)
    
    # 4. Evaluate
    print("Step 4: Evaluating results...")
    evaluate_results(gt, reconstruction)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")