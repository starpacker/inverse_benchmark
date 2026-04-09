import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_ctf_2d(N, defocus, pixel_size=2.5, voltage=300.0, cs=2.7, w=0.07):
    """Compute the 2D Contrast Transfer Function."""
    freq = np.fft.fftfreq(N, d=pixel_size)
    freq_s = np.fft.fftshift(freq)
    fy, fx = np.meshgrid(freq_s, freq_s, indexing='ij')
    s2 = fx**2 + fy**2
    voltage_V = voltage * 1e3
    lam = 12.2639 / np.sqrt(voltage_V + 0.97845e-6 * voltage_V**2)
    cs_A = cs * 1e7
    gamma = 2 * np.pi * (-0.5 * defocus * lam * s2 + 0.25 * cs_A * lam**3 * s2**2)
    ctf = np.sqrt(1 - w**2) * np.sin(gamma) - w * np.cos(gamma)
    return ctf.astype(np.float64)
