import warnings

import matplotlib

matplotlib.use('Agg')

import numpy as np

import odl

warnings.filterwarnings('ignore')

def load_and_preprocess_data(image_size, num_angles, noise_level):
    """
    Load and preprocess data for CT reconstruction.
    
    Creates the reconstruction space, generates the Shepp-Logan phantom,
    sets up the parallel-beam geometry, and generates noisy sinogram data.
    
    Parameters
    ----------
    image_size : int
        Size of the reconstruction grid (N x N).
    num_angles : int
        Number of projection angles.
    noise_level : float
        Relative noise level to add to the sinogram.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'reco_space': ODL reconstruction space
        - 'phantom': ODL element of the ground truth phantom
        - 'ground_truth': numpy array of ground truth
        - 'geometry': ODL geometry object
        - 'ray_transform': ODL RayTransform operator
        - 'sinogram': ODL element of noisy sinogram
        - 'sinogram_clean': ODL element of clean sinogram
        - 'parameters': dict of input parameters
    """
    # Create reconstruction space
    reco_space = odl.uniform_discr(
        min_pt=[-1, -1], max_pt=[1, 1], shape=[image_size, image_size], dtype='float32'
    )
    
    # Generate Shepp-Logan phantom
    phantom = odl.phantom.shepp_logan(reco_space, modified=True)
    gt = phantom.asarray()
    
    # Set up parallel-beam geometry
    geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=num_angles)
    ray_transform = odl.tomo.RayTransform(reco_space, geometry)
    
    # Generate clean sinogram
    sinogram_clean = ray_transform(phantom)
    
    # Add noise to sinogram
    noise = odl.phantom.white_noise(ray_transform.range)
    sinogram = sinogram_clean + noise * np.mean(np.abs(sinogram_clean.asarray())) * noise_level
    
    return {
        'reco_space': reco_space,
        'phantom': phantom,
        'ground_truth': gt,
        'geometry': geometry,
        'ray_transform': ray_transform,
        'sinogram': sinogram,
        'sinogram_clean': sinogram_clean,
        'parameters': {
            'image_size': image_size,
            'num_angles': num_angles,
            'noise_level': noise_level
        }
    }
