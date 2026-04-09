import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from scipy.ndimage import gaussian_filter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_complex_object(size=128):
    """Create a complex-valued object with amplitude and phase features."""
    yy, xx = np.mgrid[:size, :size].astype(np.float64)

    # Amplitude: piecewise constant features, then smoothed
    amp = np.ones((size, size), dtype=np.float64) * 0.8
    amp[20:40, 30:60] = 0.3
    amp[60:90, 50:80] = 0.5
    amp[40:70, 20:45] = 0.6
    amp[100:115, 80:110] = 0.25
    amp[15:25, 70:85] = 0.55
    r1 = np.sqrt((yy - 80)**2 + (xx - 30)**2)
    amp[r1 < 15] = 0.35
    r2 = np.sqrt((yy - 30)**2 + (xx - 90)**2)
    amp[r2 < 12] = 0.45
    amp = gaussian_filter(amp, sigma=1.5)

    # Phase: smooth low-frequency variation
    phase = (0.5 * np.sin(2 * np.pi * xx / size) *
             np.cos(2 * np.pi * yy / size))
    phase += 0.8 * np.exp(-((yy - 60)**2 + (xx - 70)**2) / (2 * 25**2))
    phase -= 0.6 * np.exp(-((yy - 40)**2 + (xx - 40)**2) / (2 * 20**2))

    return amp * np.exp(1j * phase)

def create_probe(probe_size):
    """Circular aperture x Gaussian envelope with mild phase curvature."""
    yy, xx = np.mgrid[:probe_size, :probe_size].astype(np.float64)
    cy = cx = probe_size / 2.0
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    R = probe_size * 0.38
    sigma = probe_size * 0.22
    aperture = (r <= R).astype(np.float64)
    gaussian = np.exp(-r**2 / (2 * sigma**2))
    probe = aperture * gaussian * np.exp(-0.3j * (r / R)**2)
    probe /= np.sqrt(np.sum(np.abs(probe)**2))
    return probe

def generate_scan_positions(obj_size, probe_size, overlap):
    """Regular grid scan with given overlap fraction."""
    step = max(int(probe_size * (1 - overlap)), 1)
    pos_1d = np.arange(0, obj_size - probe_size + 1, step)
    if len(pos_1d) == 0:
        pos_1d = np.array([0])
    if pos_1d[-1] < obj_size - probe_size:
        pos_1d = np.append(pos_1d, obj_size - probe_size)
    return [(int(py), int(px)) for py in pos_1d for px in pos_1d]

def add_poisson_noise(patterns, photon_count):
    """Scale each pattern to *photon_count* total photons, Poisson-sample."""
    noisy = []
    for pat in patterns:
        s = photon_count / (pat.sum() + 1e-30)
        n = np.random.poisson(np.maximum(pat * s, 0)).astype(np.float64) / s
        noisy.append(n)
    return noisy

def load_and_preprocess_data(obj_size, probe_size, overlap, photon_count, seed=42):
    """
    Create/load ground truth object and probe, generate scan positions,
    simulate diffraction patterns with Poisson noise.

    Parameters
    ----------
    obj_size : int
        Size of the square complex object.
    probe_size : int
        Size of the square probe.
    overlap : float
        Overlap fraction between adjacent scan positions (0 to 1).
    photon_count : float
        Total photon count per diffraction pattern for noise simulation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'gt_object': complex ndarray, ground truth object
        - 'gt_probe': complex ndarray, ground truth probe
        - 'positions': list of (py, px) tuples, scan positions
        - 'patterns_noisy': list of ndarrays, noisy diffraction patterns
        - 'patterns_clean': list of ndarrays, clean diffraction patterns
        - 'obj_shape': tuple, shape of the object
        - 'probe_shape': tuple, shape of the probe
    """
    np.random.seed(seed)

    # Create ground truth object and probe
    gt_object = create_complex_object(obj_size)
    gt_probe = create_probe(probe_size)

    print(f"  Object  {gt_object.shape}  amp in [{np.abs(gt_object).min():.3f}, "
          f"{np.abs(gt_object).max():.3f}]  phase in [{np.angle(gt_object).min():.3f}, "
          f"{np.angle(gt_object).max():.3f}] rad")
    print(f"  Probe   {gt_probe.shape}")

    # Generate scan positions
    positions = generate_scan_positions(obj_size, probe_size, overlap)
    print(f"  {len(positions)} positions, overlap {overlap*100:.0f}%")

    # Compute clean diffraction patterns using forward model
    ph, pw = gt_probe.shape
    patterns_clean = []
    for py, px in positions:
        exit_wave = gt_probe * gt_object[py:py+ph, px:px+pw]
        intensity = np.abs(np.fft.fft2(exit_wave))**2
        patterns_clean.append(intensity)

    # Add Poisson noise
    patterns_noisy = add_poisson_noise(patterns_clean, photon_count)
    print(f"  {len(patterns_noisy)} patterns of shape {patterns_noisy[0].shape}, "
          f"{photon_count:.0e} photons")

    return {
        'gt_object': gt_object,
        'gt_probe': gt_probe,
        'positions': positions,
        'patterns_noisy': patterns_noisy,
        'patterns_clean': patterns_clean,
        'obj_shape': gt_object.shape,
        'probe_shape': gt_probe.shape
    }
