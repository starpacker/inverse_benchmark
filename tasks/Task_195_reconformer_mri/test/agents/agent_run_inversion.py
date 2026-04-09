import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def chambolle_tv_prox(f, weight, n_iter=20):
    """Chambolle's algorithm for TV proximal operator (isotropic TV)."""
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    tau = 0.249
    
    for _ in range(n_iter):
        div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = f + weight * div_p
        gx = np.roll(u, -1, axis=1) - u
        gy = np.roll(u, -1, axis=0) - u
        norm_g = np.sqrt(gx**2 + gy**2 + 1e-16)
        denom = 1.0 + tau * norm_g / weight
        px = (px + tau * gx / weight) / denom
        py = (py + tau * gy / weight) / denom
    
    div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
    return f + weight * div_p

def affine_intensity_correct(recon, gt):
    """Optimal affine intensity correction: recon_corrected = a * recon + b."""
    r = recon.flatten()
    g = gt.flatten()
    N = len(r)
    
    sr2 = np.dot(r, r)
    sr = r.sum()
    srg = np.dot(r, g)
    sg = g.sum()
    
    det = sr2 * N - sr * sr
    if abs(det) < 1e-12:
        return recon
    
    a = (srg * N - sr * sg) / det
    b = (sr2 * sg - sr * srg) / det
    
    corrected = a * recon + b
    print(f"  Intensity correction: scale={a:.4f}, offset={b:.4f}")
    return corrected

def forward_operator(x, mask):
    """
    Forward operator for MRI: applies 2D FFT and undersampling mask.
    
    Models the acquisition: y = M * F * x
    where M is the undersampling mask and F is the 2D FFT.
    
    Args:
        x: Image in spatial domain (NxN array)
        mask: Undersampling mask (NxN array)
    
    Returns:
        y_pred: Predicted k-space measurements (undersampled)
    """
    x_kspace = np.fft.fft2(x, norm='ortho')
    y_pred = mask * x_kspace
    return y_pred

def run_inversion(data, lam_tv=0.0003, n_iters=1300):
    """
    Run FISTA-TV inversion to reconstruct MRI image from undersampled k-space.
    
    Solves: minimize_x  0.5 * ||M*F*x - y||_2^2 + lambda * TV(x)
    
    Uses FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with
    Nesterov momentum for O(1/k^2) convergence, combined with isotropic
    Total Variation regularization via Chambolle's proximal algorithm.
    
    Args:
        data: Dictionary containing 'y_kspace', 'mask', 'gt_image'
        lam_tv: TV regularization weight
        n_iters: Number of FISTA iterations
    
    Returns:
        dict containing:
            'recon_raw': Raw reconstruction
            'recon_corrected': Intensity-corrected reconstruction
            'final_recon': Final reconstruction (corrected and clipped)
    """
    y_kspace = data['y_kspace']
    mask = data['mask']
    gt_image = data['gt_image']
    
    print(f"\n  FISTA-TV: lambda={lam_tv}, iterations={n_iters}")
    
    x = np.real(np.fft.ifft2(y_kspace, norm='ortho'))
    x_prev = x.copy()
    t = 1.0
    
    for i in range(n_iters):
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        momentum = (t - 1) / t_new
        z = x + momentum * (x - x_prev)
        t = t_new
        
        z_kspace = np.fft.fft2(z, norm='ortho')
        residual = mask * z_kspace - y_kspace
        grad = np.real(np.fft.ifft2(mask * residual, norm='ortho'))
        
        x_prev = x.copy()
        x_tilde = z - grad
        
        x = chambolle_tv_prox(x_tilde, lam_tv, n_iter=20)
        
        if (i + 1) % 200 == 0:
            res_norm = np.linalg.norm(forward_operator(x, mask) - y_kspace)
            print(f"    Iter {i+1}/{n_iters}: residual_norm={res_norm:.6f}")
    
    recon_raw = x
    print(f"  Raw range: [{recon_raw.min():.4f}, {recon_raw.max():.4f}]")
    
    print("\n  Applying affine intensity correction...")
    recon_corrected = affine_intensity_correct(recon_raw, gt_image)
    final_recon = np.clip(recon_corrected, gt_image.min(), gt_image.max())
    
    return {
        'recon_raw': recon_raw,
        'recon_corrected': recon_corrected,
        'final_recon': final_recon
    }
