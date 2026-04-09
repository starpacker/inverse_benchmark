import numpy as np

import scipy.ndimage

try:
    from skimage.transform import radon, iradon
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using slower fallback implementations.")

HAS_SKIMAGE = True

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
