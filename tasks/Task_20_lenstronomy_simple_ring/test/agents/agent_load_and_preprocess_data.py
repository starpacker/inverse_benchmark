import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel

def load_and_preprocess_data(
    numPix: int,
    pixel_scale: float,
    background_rms: float,
    exp_time: float,
    fwhm: float,
    psf_type: str,
    lens_model_list: list,
    kwargs_lens: list,
    source_model_list: list,
    kwargs_source: list,
    lens_light_model_list: list,
    kwargs_lens_light: list,
    random_seed: int = None
) -> dict:
    """
    Load and preprocess data: create simulated lensed image with noise.
    
    Returns a dictionary containing all data and model objects needed for fitting.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Define coordinate system
    transform_pix2angle = np.array([[-pixel_scale, 0], [0, pixel_scale]])
    
    # Calculate the RA/Dec of the pixel (0,0) such that the image is centered at (0,0)
    cx = (numPix - 1) / 2.0
    cy = (numPix - 1) / 2.0
    ra_at_xy_0 = -(transform_pix2angle[0, 0] * cx + transform_pix2angle[0, 1] * cy)
    dec_at_xy_0 = -(transform_pix2angle[1, 0] * cx + transform_pix2angle[1, 1] * cy)
    
    # Create kwargs_data
    kwargs_data = {
        'background_rms': background_rms,
        'exposure_time': exp_time,
        'ra_at_xy_0': ra_at_xy_0,
        'dec_at_xy_0': dec_at_xy_0,
        'transform_pix2angle': transform_pix2angle,
        'image_data': np.zeros((numPix, numPix))
    }
    
    # Create data class
    data_class = ImageData(**kwargs_data)
    
    # Create PSF
    kwargs_psf = {
        'psf_type': psf_type,
        'fwhm': fwhm,
        'pixel_size': pixel_scale,
        'truncation': 3
    }
    psf_class = PSF(**kwargs_psf)
    
    # Numerics settings
    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }
    
    # Create model classes
    lens_model_class = LensModel(lens_model_list)
    source_model_class = LightModel(source_model_list)
    lens_light_model_class = LightModel(lens_light_model_list)
    
    # Create image model for simulation
    imageModel = ImageModel(
        data_class, psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        kwargs_numerics=kwargs_numerics
    )
    
    # Generate noise-free model image
    image_model = imageModel.image(
        kwargs_lens, kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_ps=None
    )
    
    # Add Poisson Noise (Photon Shot Noise)
    image_model_counts = image_model * exp_time
    image_model_counts[image_model_counts < 0] = 0  # Sanity check for negative flux
    poisson_counts = np.random.poisson(image_model_counts)
    image_with_poisson = poisson_counts / exp_time
    
    # Add Gaussian Background Noise
    bkg_noise = np.random.randn(*image_model.shape) * background_rms
    
    # Final simulated image
    image_real = image_with_poisson + bkg_noise
    
    # Update data class with simulated image
    data_class.update_data(image_real)
    kwargs_data['image_data'] = image_real
    
    print("Simulated Image Generated.")
    
    return {
        'image_data': image_real,
        'kwargs_data': kwargs_data,
        'kwargs_psf': kwargs_psf,
        'kwargs_numerics': kwargs_numerics,
        'lens_model_list': lens_model_list,
        'source_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list,
        'data_class': data_class,
        'psf_class': psf_class,
        'lens_model_class': lens_model_class,
        'source_model_class': source_model_class,
        'lens_light_model_class': lens_light_model_class,
        'true_kwargs_lens': kwargs_lens,
        'true_kwargs_source': kwargs_source,
        'true_kwargs_lens_light': kwargs_lens_light,
        'numPix': numPix,
        'pixel_scale': pixel_scale,
        'background_rms': background_rms,
        'exp_time': exp_time
    }