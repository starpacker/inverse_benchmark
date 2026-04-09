import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

import scipy.fft

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

def load_and_preprocess_data(grid_size, pixel_size, young_modulus, poisson_ratio, noise_std, seed=42):
    """
    Generate synthetic benchmark data:
    1. Create ground truth traction field (cell-like dipole pattern)
    2. Apply forward model to get displacement field (in pixels)
    3. Add measurement noise
    
    Parameters:
        grid_size: int, size of synthetic traction field (pixels)
        pixel_size: float, pixel size in meters
        young_modulus: float, Young's modulus in Pa
        poisson_ratio: float, Poisson's ratio
        noise_std: float, noise standard deviation in pixels
        seed: int, random seed for reproducibility
    
    Returns:
        measurements: tuple (u_noisy, v_noisy) displacement fields in pixels
        ground_truth: tuple (tx_gt, ty_gt) ground truth traction fields in Pa
        metadata: dict with physical parameters
    """
    print("[DATA] Generating synthetic traction force dipole...")
    
    N = grid_size
    y, x = np.mgrid[:N, :N]
    cx, cy = N // 2, N // 2

    # Two Gaussian blobs offset from center (contractile dipole)
    sigma_blob = N / 8.0
    offset = N / 5.0

    # Left blob (pulling right)
    g_left = np.exp(-((x - (cx - offset))**2 + (y - cy)**2) / (2 * sigma_blob**2))
    # Right blob (pulling left)
    g_right = np.exp(-((x - (cx + offset))**2 + (y - cy)**2) / (2 * sigma_blob**2))

    # Traction magnitudes (typical cell traction: 100-1000 Pa)
    max_traction = 500.0  # Pa

    # tx: left blob pulls right (+), right blob pulls left (-)
    tx_gt = max_traction * g_left - max_traction * g_right
    # ty: add some vertical component (realistic cell shape)
    ty_gt = 0.3 * max_traction * g_left * (y - cy) / (sigma_blob * 2)
    ty_gt -= 0.3 * max_traction * g_right * (y - cy) / (sigma_blob * 2)

    tx_gt = tx_gt.astype(np.float64)
    ty_gt = ty_gt.astype(np.float64)

    # Forward model: traction → displacement (returns pixels)
    print("[DATA] Applying forward operator (Boussinesq Green's function)...")
    u_clean_px, v_clean_px = forward_operator(tx_gt, ty_gt, pixel_size, young_modulus, poisson_ratio)

    print(f"[DATA] Max displacement: {np.max(np.sqrt(u_clean_px**2 + v_clean_px**2)):.4f} pixels")

    # Add Gaussian noise to displacement
    np.random.seed(seed)
    u_noisy_px = u_clean_px + noise_std * np.random.randn(*u_clean_px.shape)
    v_noisy_px = v_clean_px + noise_std * np.random.randn(*v_clean_px.shape)

    metadata = {
        "young": young_modulus,
        "sigma": poisson_ratio,
        "pixelsize1": pixel_size,
        "pixelsize2": pixel_size,
        "noise_std": noise_std,
    }

    measurements = (u_noisy_px, v_noisy_px)
    ground_truth = (tx_gt, ty_gt)

    return measurements, ground_truth, metadata

def forward_operator(tx, ty, pixelsize, young, sigma=0.49):
    """
    Boussinesq forward model (infinite half-space): compute surface displacements
    from traction forces using Fourier-space Green's function.

    The inverse solver computes:  T = K_inv * FFT(u * pixelsize1)
    So the forward is:  u = IFFT(K * T_ft) / pixelsize1

    where K_inv is the elastic stiffness kernel and K = K_inv^{-1} is the
    compliance (Green's function) kernel.

    Parameters:
        tx, ty: 2D arrays, traction fields in Pa
        pixelsize: float, pixel size in meters (same for pixelsize1 and pixelsize2)
        young: float, Young's modulus in Pa
        sigma: float, Poisson's ratio

    Returns:
        u, v: 2D arrays, displacement fields in PIXELS
    """
    ax1_length, ax2_length = tx.shape
    max_ind = int(max(ax1_length, ax2_length))
    if max_ind % 2 != 0:
        max_ind += 1

    # MUST match inverse solver's padding convention: top-left placement
    tx_expand = np.zeros((max_ind, max_ind))
    ty_expand = np.zeros((max_ind, max_ind))
    tx_expand[:ax1_length, :ax2_length] = tx
    ty_expand[:ax1_length, :ax2_length] = ty

    # Wave vectors — MUST match ffttc_traction exactly
    kx1 = np.array([list(range(0, int(max_ind / 2), 1)), ] * int(max_ind))
    kx2 = np.array([list(range(-int(max_ind / 2), 0, 1)), ] * int(max_ind))
    kx = np.append(kx1, kx2, axis=1) * 2 * np.pi
    ky = np.transpose(kx)
    k = np.sqrt(kx ** 2 + ky ** 2) / (pixelsize * max_ind)

    # Angle (same as inverse solver)
    alpha = np.arctan2(ky, kx)
    alpha[0, 0] = np.pi / 2

    # Build the inverse stiffness kernel components
    kix = ((k * young) / (2 * (1 - sigma ** 2))) * (1 - sigma + sigma * np.cos(alpha) ** 2)
    kiy = ((k * young) / (2 * (1 - sigma ** 2))) * (1 - sigma + sigma * np.sin(alpha) ** 2)
    kid = ((k * young) / (2 * (1 - sigma ** 2))) * (sigma * np.sin(alpha) * np.cos(alpha))

    # Zero out cross terms at Nyquist (same as inverse solver)
    kid[:, int(max_ind / 2)] = np.zeros(max_ind)
    kid[int(max_ind / 2), :] = np.zeros(max_ind)

    # Determinant of the K_inv 2x2 matrix
    det = kix * kiy - kid ** 2

    # Avoid division by zero at DC
    det[0, 0] = 1.0  # will be zeroed out anyway

    # Green's function (inverse of K_inv)
    g11 = kiy / det
    g12 = -kid / det
    g22 = kix / det

    # FFT of traction fields
    tx_ft = scipy.fft.fft2(tx_expand)
    ty_ft = scipy.fft.fft2(ty_expand)

    # Displacement in Fourier space (in meters, since inverse uses u*pixelsize1)
    u_ft = g11 * tx_ft + g12 * ty_ft
    v_ft = g12 * tx_ft + g22 * ty_ft

    # Zero DC component (mean displacement is unconstrained)
    u_ft[0, 0] = 0
    v_ft[0, 0] = 0

    # Back to real space
    u = scipy.fft.ifft2(u_ft).real
    v = scipy.fft.ifft2(v_ft).real

    # Cut to original size (MUST match inverse solver's placement)
    u_cut = u[:ax1_length, :ax2_length]
    v_cut = v[:ax1_length, :ax2_length]

    # The inverse solver multiplies u by pixelsize1 before FFT:
    # So the forward result in meters is: u_cut (from above)
    # And u_pixels = u_meters / pixelsize1
    return u_cut / pixelsize, v_cut / pixelsize
