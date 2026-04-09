import numpy as np

import matplotlib

matplotlib.use('Agg')

def fft2c(img):
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    """Centered 2D IFFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

def gradient_2d(img):
    """Compute discrete gradient (finite differences)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy

def divergence_2d(gx, gy):
    """Compute divergence (adjoint of gradient)."""
    dx = np.zeros_like(gx)
    dy = np.zeros_like(gy)
    dx[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
    dx[:, 0] = gx[:, 0]
    dx[:, -1] = -gx[:, -2]
    dy[1:-1, :] = gy[1:-1, :] - gy[:-2, :]
    dy[0, :] = gy[0, :]
    dy[-1, :] = -gy[-2, :]
    return dx + dy

def tv_prox(img, lam, n_inner=80):
    """
    Proximal operator for isotropic TV using Chambolle's projection algorithm.
    Solves: argmin_x 0.5*||x - img||^2 + lam*TV(x)
    """
    if lam <= 0:
        return img.copy()

    px = np.zeros_like(img)
    py = np.zeros_like(img)
    tau = 1.0 / 8.0

    for _ in range(n_inner):
        div_p = divergence_2d(px, py)
        gx, gy = gradient_2d(div_p - img / lam)
        px_new = px + tau * gx
        py_new = py + tau * gy
        norm = np.sqrt(px_new**2 + py_new**2)
        norm = np.maximum(norm, 1.0)
        px = px_new / norm
        py = py_new / norm

    return img - lam * divergence_2d(px, py)

def forward_operator(x, mask):
    """
    Apply forward operator: FFT then mask.
    
    Parameters:
    -----------
    x : ndarray
        Input image in image domain
    mask : ndarray
        Undersampling mask (boolean or binary)
    
    Returns:
    --------
    y_pred : ndarray
        Undersampled k-space data
    """
    kspace = fft2c(x)
    y_pred = kspace * mask
    return y_pred

def run_inversion(data_dict, lam_tv_t1=0.0008, lam_tv_t2=0.001, n_iter=250, verbose=True):
    """
    Run FISTA with TV regularization for MRI reconstruction.
    
    Parameters:
    -----------
    data_dict : dict
        Contains ground truth images and masks from load_and_preprocess_data
    lam_tv_t1 : float
        TV regularization parameter for T1
    lam_tv_t2 : float
        TV regularization parameter for T2
    n_iter : int
        Number of FISTA iterations
    verbose : bool
        Print progress information
    
    Returns:
    --------
    result_dict : dict
        Contains reconstructions, zero-filled images, and k-space data
    """
    print("\n[3/4] Forward operation + reconstruction...")
    
    t1_gt = data_dict['t1_gt']
    t2_gt = data_dict['t2_gt']
    mask_t1 = data_dict['mask_t1']
    mask_t2 = data_dict['mask_t2']
    
    # Forward operation: generate undersampled k-space
    kspace_t1 = forward_operator(t1_gt, mask_t1)
    kspace_t2 = forward_operator(t2_gt, mask_t2)
    
    # Zero-filled reconstructions (baseline)
    zf_t1 = np.abs(ifft2c(kspace_t1))
    zf_t2 = np.abs(ifft2c(kspace_t2))
    
    # FISTA-TV reconstruction for T1
    print("\n  Reconstructing T1 (4x)...")
    recon_t1 = _fista_tv_recon(kspace_t1, mask_t1, lam_tv=lam_tv_t1, n_iter=n_iter, verbose=verbose)
    
    # FISTA-TV reconstruction for T2
    print("\n  Reconstructing T2 (8x)...")
    recon_t2 = _fista_tv_recon(kspace_t2, mask_t2, lam_tv=lam_tv_t2, n_iter=n_iter, verbose=verbose)
    
    result_dict = {
        'recon_t1': recon_t1,
        'recon_t2': recon_t2,
        'zf_t1': zf_t1,
        'zf_t2': zf_t2,
        'kspace_t1': kspace_t1,
        'kspace_t2': kspace_t2,
        'n_iter': n_iter,
    }
    
    return result_dict

def _fista_tv_recon(kspace_masked, mask, lam_tv=0.001, n_iter=300, verbose=True):
    """
    FISTA with TV regularization for MRI reconstruction.
    
    min_x 0.5 * ||F_mask(x) - y||^2 + lam_tv * TV(x)
    """
    # Initial: zero-filled reconstruction
    x = np.real(ifft2c(kspace_masked))
    z = x.copy()
    t = 1.0
    step = 1.0

    for k in range(n_iter):
        x_old = x.copy()

        # Gradient of data fidelity: Re{ F^H (mask * (F*z) - y) }
        fz = fft2c(z)
        residual = mask * fz - kspace_masked
        grad = np.real(ifft2c(residual))

        # Gradient step
        z_step = z - step * grad

        # Proximal step (TV denoising)
        x = tv_prox(z_step, lam_tv * step, n_inner=50)

        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = x + ((t - 1.0) / t_new) * (x - x_old)
        t = t_new

        if verbose and (k + 1) % 50 == 0:
            data_fit = np.linalg.norm(fft2c(x) * mask - kspace_masked) / np.linalg.norm(kspace_masked)
            print(f"  Iter {k+1}/{n_iter}, relative residual = {data_fit:.6f}")

    return x
