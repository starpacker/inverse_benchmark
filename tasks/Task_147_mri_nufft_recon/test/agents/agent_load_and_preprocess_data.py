import matplotlib

matplotlib.use('Agg')

import sys

import os

import warnings

import numpy as np

from skimage.data import shepp_logan_phantom as skimage_shepp_logan

from skimage.transform import resize

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo')

sys.path.insert(0, REPO_DIR)

warnings.filterwarnings('ignore', message='Samples will be rescaled')

def load_and_preprocess_data(N=128, n_spokes=220, noise_level=0.002):
    """
    Load and preprocess data for non-Cartesian MRI reconstruction.
    
    Generates the Shepp-Logan phantom, creates radial trajectory and NUFFT operators,
    and simulates non-Cartesian k-space acquisition with noise.
    
    Parameters
    ----------
    N : int
        Image size (NxN)
    n_spokes : int
        Number of radial spokes
    noise_level : float
        Relative noise level for k-space
        
    Returns
    -------
    dict containing:
        - phantom: ground truth image (N, N)
        - kdata: noisy k-space measurements
        - kdata_clean: clean k-space measurements
        - op_dc: NUFFT operator with density compensation
        - op_plain: NUFFT operator without density compensation
        - trajectory: radial trajectory
        - params: dictionary of parameters
    """
    from mrinufft import get_operator
    from mrinufft.trajectories import initialize_2D_radial
    
    # Generate Shepp-Logan phantom
    phantom_400 = skimage_shepp_logan()
    phantom = resize(phantom_400, (N, N), anti_aliasing=True).astype(np.float64)
    
    # Generate radial trajectory: shape (Nc, Ns, 2) in [-0.5, 0.5)
    trajectory = initialize_2D_radial(Nc=n_spokes, Ns=N, tilt='uniform')
    
    # Flatten for operator: (N_samples, 2)
    traj_flat = trajectory.reshape(-1, 2)
    
    # Operator with Voronoi density compensation (for adjoint reconstruction)
    op_dc = get_operator('finufft')(
        samples=traj_flat,
        shape=(N, N),
        density=True,
        n_coils=1,
    )
    
    # Plain operator without DC (for iterative reconstruction)
    op_plain = get_operator('finufft')(
        samples=traj_flat,
        shape=(N, N),
        density=False,
        n_coils=1,
    )
    
    # Forward model: apply NUFFT and add noise
    image_complex = phantom.astype(np.complex64)
    kdata_clean = op_plain.op(image_complex)
    
    # Add complex Gaussian noise
    rng = np.random.default_rng(42)
    noise_std = noise_level * np.abs(kdata_clean).max()
    noise = noise_std * (rng.standard_normal(kdata_clean.shape) +
                         1j * rng.standard_normal(kdata_clean.shape)).astype(np.complex64) / np.sqrt(2)
    kdata = kdata_clean + noise
    
    # Compute parameters
    n_samples = trajectory.reshape(-1, 2).shape[0]
    nyquist_spokes = int(np.ceil(np.pi * N / 2))
    acceleration = nyquist_spokes / n_spokes
    snr_kspace = 20 * np.log10(np.linalg.norm(kdata_clean) /
                                np.linalg.norm(kdata - kdata_clean))
    
    params = {
        'N': N,
        'n_spokes': n_spokes,
        'noise_level': noise_level,
        'n_samples': n_samples,
        'nyquist_spokes': nyquist_spokes,
        'acceleration': acceleration,
        'snr_kspace': snr_kspace,
    }
    
    return {
        'phantom': phantom,
        'kdata': kdata,
        'kdata_clean': kdata_clean,
        'op_dc': op_dc,
        'op_plain': op_plain,
        'trajectory': trajectory,
        'params': params,
    }
