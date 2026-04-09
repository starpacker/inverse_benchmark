import numpy as np
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

def load_and_preprocess_data(
    numPix: int,
    deltaPix: float,
    background_rms: float,
    exp_time: float,
    fwhm: float,
    kwargs_lens: list,
    kwargs_source_true: list,
    lens_model_list: list,
    source_model_list_true: list,
    random_seed: int = 42
) -> dict:
    """
    Load and preprocess data by simulating a lensed complex source with noise.
    
    Returns a dictionary containing all necessary data and class instances.
    """
    np.random.seed(random_seed)
    
    # Transformation matrix: pixel coordinates -> angular coordinates
    transform_pix2angle = np.array([[-deltaPix, 0], [0, deltaPix]])
    
    # Calculate the RA/Dec of the pixel (0,0) such that the image is centered at (0,0)
    cx = (numPix - 1) / 2.0
    cy = (numPix - 1) / 2.0
    ra_at_xy_0 = -(transform_pix2angle[0, 0] * cx + transform_pix2angle[0, 1] * cy)
    dec_at_xy_0 = -(transform_pix2angle[1, 0] * cx + transform_pix2angle[1, 1] * cy)
    
    # Initialize data class with zeros
    kwargs_data = {
        'background_rms': background_rms,
        'exposure_time': exp_time,
        'ra_at_xy_0': ra_at_xy_0,
        'dec_at_xy_0': dec_at_xy_0,
        'transform_pix2angle': transform_pix2angle,
        'image_data': np.zeros((numPix, numPix))
    }
    data_class = ImageData(**kwargs_data)
    
    # Configure PSF
    kwargs_psf = {
        'psf_type': 'GAUSSIAN',
        'fwhm': fwhm,
        'pixel_size': deltaPix,
        'truncation': 3
    }
    psf_class = PSF(**kwargs_psf)
    
    # Define Lens Model
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    
    # Define True Source Model
    source_model_class_true = LightModel(light_model_list=source_model_list_true)
    
    # No Lens Light
    lens_light_model_list = []
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    
    # Numerics settings
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    
    # Create ImageModel for simulation
    imageModel = ImageModel(
        data_class, psf_class, lens_model_class, source_model_class_true,
        lens_light_model_class, kwargs_numerics=kwargs_numerics
    )
    
    # Simulate clean image
    image_sim = imageModel.image(kwargs_lens, kwargs_source_true, kwargs_lens_light=None, kwargs_ps=None)
    
    # Add Poisson Noise
    image_sim_counts = image_sim * exp_time
    image_sim_counts[image_sim_counts < 0] = 0
    poisson_counts = np.random.poisson(image_sim_counts)
    poisson = poisson_counts / exp_time
    poisson_noise = poisson - image_sim
    
    # Add Gaussian Background Noise
    bkg_noise = np.random.randn(*image_sim.shape) * background_rms
    
    # Combine to get noisy image
    image_noisy = image_sim + bkg_noise + poisson_noise
    
    # Update Data with Simulated Image
    kwargs_data['image_data'] = image_noisy
    data_class.update_data(image_noisy)
    
    return {
        'data_class': data_class,
        'psf_class': psf_class,
        'lens_model_class': lens_model_class,
        'kwargs_numerics': kwargs_numerics,
        'kwargs_lens': kwargs_lens,
        'image_data': image_noisy,
        'image_clean': image_sim,
        'numPix': numPix,
        'deltaPix': deltaPix,
        'background_rms': background_rms,
        'exp_time': exp_time
    }