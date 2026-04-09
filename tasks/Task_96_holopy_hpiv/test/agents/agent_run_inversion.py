import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.ndimage import gaussian_filter, label, center_of_mass

def _asm_kernel(nx, ny, dx, z, wl):
    """Angular Spectrum Method propagation kernel."""
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    kz2 = (1.0 / wl) ** 2 - FX ** 2 - FY ** 2
    prop = kz2 > 0
    return np.exp(1j * 2 * np.pi * np.sqrt(np.maximum(kz2, 0)) * z) * prop

def run_inversion(hologram, pixel_size, z_planes, wavelength, n_expected):
    """
    Detect 3D particle positions from hologram via backpropagation.
    
    Inverse model:
      1. Back-propagate hologram to z-planes via ASM
      2. Local focus metric (gradient-based)
      3. Peak detection in 3-D focus volume → (x, y, z)
    
    Parameters:
    -----------
    hologram : ndarray
        Inline hologram intensity (nx, ny)
    pixel_size : float
        Detector pixel pitch (m)
    z_planes : ndarray
        Array of z-positions to scan
    wavelength : float
        Illumination wavelength (m)
    n_expected : int
        Expected number of particles (not strictly used)
    
    Returns:
    --------
    detected_positions : ndarray
        Detected particle positions (M, 3) with (x, y, z)
    gradient_volume : ndarray
        3D focus metric volume (nz, nx-1, ny-1)
    """
    nx, ny = hologram.shape
    nz = len(z_planes)
    
    # FFT of hologram field (take sqrt to get amplitude)
    H_fft = np.fft.fft2(np.sqrt(hologram.astype(complex)))
    
    # Compute gradient-based focus metric at each z-plane
    grad_vol = np.zeros((nz, nx - 1, ny - 1), dtype=np.float32)
    
    for i, z in enumerate(z_planes):
        # Back-propagate to z (negative z for backpropagation)
        E = np.fft.ifft2(H_fft * _asm_kernel(nx, ny, pixel_size, -z, wavelength))
        I = np.abs(E) ** 2
        
        # Compute gradient magnitude
        gx = np.diff(I, axis=0)[:, :-1]
        gy = np.diff(I, axis=1)[:-1, :]
        grad_mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
        
        # Smooth for robustness
        grad_vol[i] = gaussian_filter(grad_mag, sigma=3.0)
    
    # Maximum intensity projection for 2D peak detection
    mip = np.max(grad_vol, axis=0)
    
    # Threshold and label connected components
    thr = np.percentile(mip, 93)
    lab, nf = label(mip > thr)
    centroids = center_of_mass(mip, lab, range(1, nf + 1))
    
    # For each 2D centroid, find z with maximum focus
    detected = []
    for cy, cx in centroids:
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < grad_vol.shape[1] and 0 <= ix < grad_vol.shape[2]:
            zidx = int(np.argmax(grad_vol[:, iy, ix]))
            detected.append([
                (ix + 0.5) * pixel_size,
                (iy + 0.5) * pixel_size,
                z_planes[zidx]
            ])
    
    detected_positions = np.array(detected) if detected else np.zeros((0, 3))
    
    print(f"  Detected {len(detected_positions)}")
    
    return detected_positions, grad_vol
