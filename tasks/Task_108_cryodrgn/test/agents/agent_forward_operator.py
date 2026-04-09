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

def trilinear_interp(vol3d, coords, N):
    """Trilinear interpolation for extracting values from 3D volume."""
    coords = np.clip(coords, 0, N - 1.001)
    x0 = np.floor(coords[:, 0]).astype(int)
    y0 = np.floor(coords[:, 1]).astype(int)
    z0 = np.floor(coords[:, 2]).astype(int)
    x1 = np.minimum(x0 + 1, N - 1)
    y1 = np.minimum(y0 + 1, N - 1)
    z1 = np.minimum(z0 + 1, N - 1)
    xd = coords[:, 0] - x0
    yd = coords[:, 1] - y0
    zd = coords[:, 2] - z0

    c000 = vol3d[z0, y0, x0]
    c001 = vol3d[z0, y0, x1]
    c010 = vol3d[z0, y1, x0]
    c011 = vol3d[z0, y1, x1]
    c100 = vol3d[z1, y0, x0]
    c101 = vol3d[z1, y0, x1]
    c110 = vol3d[z1, y1, x0]
    c111 = vol3d[z1, y1, x1]

    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd

def forward_operator(volume, rot_mats, noise_std, ctf_defocus, pixel_size, apply_ctf=True):
    """
    Forward operator: Generate 2D cryo-EM projections from 3D volume.
    
    Forward Model:
    3D Volume -> 3D FFT -> Extract 2D central Fourier slice at given rotation ->
    Apply CTF -> iFFT2 -> Add Gaussian noise -> 2D projection
    
    Parameters:
    -----------
    volume : np.ndarray
        The 3D volume to project.
    rot_mats : np.ndarray
        Array of rotation matrices.
    noise_std : float
        Standard deviation of Gaussian noise to add.
    ctf_defocus : float
        CTF defocus parameter.
    pixel_size : float
        Pixel size in Angstroms.
    apply_ctf : bool
        Whether to apply CTF modulation.
    
    Returns:
    --------
    projections : np.ndarray
        Array of 2D projection images.
    """
    N = volume.shape[0]
    vol_fft = np.fft.fftshift(np.fft.fftn(volume))
    freq_1d = np.fft.fftshift(np.fft.fftfreq(N))
    gy, gx = np.meshgrid(freq_1d, freq_1d, indexing='ij')
    ctf = compute_ctf_2d(N, ctf_defocus, pixel_size) if apply_ctf else None
    coords_2d = np.stack([gx.ravel(), gy.ravel(), np.zeros(N * N)], axis=-1)

    projections = []
    n_proj = len(rot_mats)
    print(f"Forward projecting {n_proj} images...")
    
    for i in range(n_proj):
        R = rot_mats[i]
        coords_3d = coords_2d @ R.T
        coords_vox = coords_3d * N + N / 2.0
        sl = trilinear_interp(vol_fft, coords_vox, N).reshape(N, N)
        if apply_ctf:
            sl = sl * ctf
        proj = np.real(np.fft.ifft2(np.fft.ifftshift(sl)))
        sig = np.std(proj) + 1e-10
        proj += np.random.RandomState(i).normal(0, noise_std * sig, proj.shape)
        projections.append(proj.astype(np.float32))
    
    return np.array(projections)
