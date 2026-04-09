import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter

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

def run_inversion(projections, rot_mats, ctf_defocus, pixel_size, wiener_eps, apply_ctf=True):
    """
    Inverse solver: CTF-Weighted Direct Fourier Inversion.
    
    For each projection, compute its 2D FFT, apply CTF^2-weighted insertion
    into 3D Fourier space (normal equations for least-squares),
    then divide by accumulated CTF^2 weights and iFFT3 to get the volume.
    
    Parameters:
    -----------
    projections : np.ndarray
        Array of 2D projection images.
    rot_mats : np.ndarray
        Array of rotation matrices corresponding to each projection.
    ctf_defocus : float
        CTF defocus parameter.
    pixel_size : float
        Pixel size in Angstroms.
    wiener_eps : float
        Wiener filter regularization parameter.
    apply_ctf : bool
        Whether CTF correction should be applied.
    
    Returns:
    --------
    reconstruction : np.ndarray
        The reconstructed 3D volume.
    """
    N = projections.shape[1]
    n_proj = len(projections)
    vol_num = np.zeros((N, N, N), dtype=np.complex128)
    vol_den = np.zeros((N, N, N), dtype=np.float64)
    freq_1d = np.fft.fftshift(np.fft.fftfreq(N))
    gy, gx = np.meshgrid(freq_1d, freq_1d, indexing='ij')
    ctf = compute_ctf_2d(N, ctf_defocus, pixel_size) if apply_ctf else np.ones((N, N))
    coords_2d = np.stack([gx.ravel(), gy.ravel(), np.zeros(N * N)], axis=-1)
    ctf_flat = ctf.ravel()
    ctf2_flat = ctf_flat ** 2

    print(f"Reconstructing from {n_proj} projections...")
    for i in range(n_proj):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_proj}")
        R = rot_mats[i]
        coords_3d = coords_2d @ R.T
        cv = coords_3d * N + N / 2.0
        cv = np.clip(cv, 0, N - 1.001)

        proj_fft = np.fft.fftshift(np.fft.fft2(projections[i]))
        wvals = ctf_flat * proj_fft.ravel()

        ix = np.round(cv[:, 0]).astype(int)
        iy = np.round(cv[:, 1]).astype(int)
        iz = np.round(cv[:, 2]).astype(int)
        mask = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N)
        ix, iy, iz = ix[mask], iy[mask], iz[mask]

        np.add.at(vol_num, (iz, iy, ix), wvals[mask])
        np.add.at(vol_den, (iz, iy, ix), ctf2_flat[mask])

    vol_fft_r = vol_num / (vol_den + wiener_eps)
    vol_r = np.real(np.fft.ifftn(np.fft.ifftshift(vol_fft_r)))
    vol_r = gaussian_filter(vol_r, sigma=0.4)
    vol_r = vol_r.astype(np.float32)
    vol_r = (vol_r - vol_r.min()) / (vol_r.max() - vol_r.min() + 1e-10)
    
    return vol_r
