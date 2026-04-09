import numpy as np

import matplotlib

matplotlib.use("Agg")

def _asm_kernel(nx, ny, dx, z, wl):
    """Angular Spectrum Method propagation kernel."""
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    kz2 = (1.0 / wl) ** 2 - FX ** 2 - FY ** 2
    prop = kz2 > 0
    return np.exp(1j * 2 * np.pi * np.sqrt(np.maximum(kz2, 0)) * z) * prop

def _shadow(nx, ny, dx, x0, y0, r):
    """Generate shadow field for a particle at (x0, y0) with radius r."""
    xx = np.arange(nx) * dx
    yy = np.arange(ny) * dx
    XX, YY = np.meshgrid(xx, yy, indexing="ij")
    s = np.zeros((nx, ny), dtype=complex)
    s[(XX - x0) ** 2 + (YY - y0) ** 2 <= r ** 2] = -1.0
    return s

def forward_operator(gt_particles, nx, ny, pixel_size, wavelength):
    """
    Simulate inline hologram using Angular Spectrum Method.
    
    Forward model:
      H(x,y) = |E_ref + Σ_i E_scat_i|²
      E_scat_i = ASM_propagate(shadow_i, z_i)
    
    Parameters:
    -----------
    gt_particles : ndarray
        Particle array (N, 4) with (x, y, z, radius)
    nx, ny : int
        Image dimensions in pixels
    pixel_size : float
        Detector pixel pitch (m)
    wavelength : float
        Illumination wavelength (m)
    
    Returns:
    --------
    hologram : ndarray
        Simulated inline hologram intensity (nx, ny)
    """
    # Initialize reference field (plane wave)
    E = np.ones((nx, ny), dtype=complex)
    
    # Add scattered field from each particle
    for x0, y0, z0, rad in gt_particles:
        # Create shadow field for particle
        sh = _shadow(nx, ny, pixel_size, x0, y0, rad)
        # Propagate via ASM to detector plane
        E += np.fft.ifft2(np.fft.fft2(sh) * _asm_kernel(nx, ny, pixel_size, z0, wavelength))
    
    # Hologram is intensity (squared magnitude)
    hologram = np.abs(E) ** 2
    
    print(f"  {hologram.shape}  I∈[{hologram.min():.4f},{hologram.max():.4f}]")
    
    return hologram
