import matplotlib

matplotlib.use('Agg')

import os

import sys

import logging

import numpy as np

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

sys.path.insert(0, REPO_DIR)

import muDIC as dic

import muDIC.vlab as vlab

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(image_shape=(256, 256), downsample_factor=4,
                              dot_size=4, density=0.5, smoothness=2.0,
                              noise_sigma=0.02):
    """
    Generate synthetic speckle image data with known deformation using muDIC's virtual lab.
    
    Parameters
    ----------
    image_shape : tuple
        (H, W) of the output (down-sampled) images.
    downsample_factor : int
        Factor by which super-resolution images are downsampled.
    dot_size : int
        Speckle dot size parameter.
    density : float
        Speckle density parameter.
    smoothness : float
        Speckle smoothness parameter.
    noise_sigma : float
        Standard deviation of Gaussian noise to add.
    
    Returns
    -------
    data : dict
        Dictionary containing:
        - 'image_stack': dic.ImageStack with [reference, deformed] images
        - 'displacement_function': callable u(xs, ys) -> (u_x, u_y)
        - 'omega_out': angular frequency in output-image pixel units
        - 'amp_out': amplitude in output-image pixels
        - 'omega_super': angular frequency in super-image pixel units
        - 'amp_super': amplitude in super-image pixels
        - 'image_shape': (H, W) of output images
        - 'downsample_factor': downsampling factor used
    """
    n_imgs = 2  # reference + 1 deformed
    super_image_shape = tuple(d * downsample_factor for d in image_shape)

    # Speckle pattern
    speckle_image = vlab.rosta_speckle(super_image_shape, dot_size=dot_size,
                                       density=density, smoothness=smoothness)

    # Deformation: harmonic bilateral (sinusoidal in both x and y)
    displacement_function = vlab.deformation_fields.harmonic_bilat
    omega_super = 2.0 * np.pi / (image_shape[0] * downsample_factor)  # in super-image coords
    amp_super = 2.0 * downsample_factor  # amplitude in super-image pixels

    image_deformer = vlab.imageDeformer_from_uFunc(
        displacement_function, omega=omega_super, amp=amp_super
    )

    # Down-sampler with realistic sensor model
    downsampler = vlab.Downsampler(
        image_shape=super_image_shape, factor=downsample_factor,
        fill=0.95, pixel_offset_stddev=0.05
    )

    # Small additive Gaussian noise
    noise_inj = vlab.noise_injector("gaussian", sigma=noise_sigma)

    # Build the synthetic image pipeline
    image_generator = vlab.SyntheticImageGenerator(
        speckle_image=speckle_image,
        image_deformer=image_deformer,
        downsampler=downsampler,
        noise_injector=noise_inj,
        n=n_imgs,
    )
    image_stack = dic.ImageStack(image_generator)

    # omega / amp in *output-image* coordinates (after downsampling)
    omega_out = omega_super * downsample_factor   # angular freq per output pixel
    amp_out = amp_super / downsample_factor       # amplitude in output pixels

    data = {
        'image_stack': image_stack,
        'displacement_function': displacement_function,
        'omega_out': omega_out,
        'amp_out': amp_out,
        'omega_super': omega_super,
        'amp_super': amp_super,
        'image_shape': image_shape,
        'downsample_factor': downsample_factor,
    }
    
    return data
