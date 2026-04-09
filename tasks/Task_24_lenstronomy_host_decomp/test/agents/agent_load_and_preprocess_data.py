import numpy as np
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LightModel.light_model import LightModel

def load_and_preprocess_data(numPix, deltaPix, background_rms, exp_time, fwhm, psf_type,
                              center_x, center_y, point_amp, kwargs_disk, kwargs_bulge):
    """
    Load and preprocess data: set up coordinate system, PSF, light models,
    simulate image with noise, and return all necessary components.
    
    Returns:
        dict containing:
            - data_class: ImageData object
            - psf_class: PSF object
            - lightModel: LightModel object
            - pointSource: PointSource object
            - image_sim: simulated image with noise
            - kwargs_data: data configuration
            - kwargs_psf: PSF configuration
            - kwargs_numerics: numerics configuration
            - kwargs_host: host galaxy parameters (ground truth)
            - kwargs_ps: point source parameters (ground truth)
            - light_model_list: list of light model names
            - point_source_list: list of point source model names
    """
    # Transformation matrix: pixel coordinates -> angular coordinates
    transform_pix2angle = np.array([[-deltaPix, 0], [0, deltaPix]])
    
    # Calculate the RA/Dec of the pixel (0,0) such that the image is centered at (0,0)
    cx = (numPix - 1) / 2.
    cy = (numPix - 1) / 2.
    ra_at_xy_0 = - (transform_pix2angle[0, 0] * cx + transform_pix2angle[0, 1] * cy)
    dec_at_xy_0 = - (transform_pix2angle[1, 0] * cx + transform_pix2angle[1, 1] * cy)
    
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
    kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
    psf_class = PSF(**kwargs_psf)
    
    # Define Point Source Model
    point_source_list = ['UNLENSED']
    pointSource = PointSource(point_source_type_list=point_source_list)
    kwargs_ps = [{'ra_image': [center_x], 'dec_image': [center_y], 'point_amp': [point_amp]}]
    
    # Define Host Galaxy Model (Disk + Bulge)
    light_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
    lightModel = LightModel(light_model_list=light_model_list)
    kwargs_host = [kwargs_disk, kwargs_bulge]
    
    # Numerics configuration
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    
    # Create image model for simulation
    imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                            point_source_class=pointSource, kwargs_numerics=kwargs_numerics)
    
    # Simulate clean image
    image_sim = imageModel.image(kwargs_lens_light=kwargs_host, kwargs_ps=kwargs_ps)
    
    # Add Poisson Noise
    image_sim_counts = image_sim * exp_time
    image_sim_counts[image_sim_counts < 0] = 0  # Ensure non-negative counts
    poisson_counts = np.random.poisson(image_sim_counts)
    poisson = poisson_counts / exp_time
    poisson_noise = poisson - image_sim
    
    # Add Gaussian Background Noise
    bkg_noise = np.random.randn(*image_sim.shape) * background_rms
    
    # Combine noise
    image_sim = image_sim + bkg_noise + poisson_noise
    
    # Update Data with the noisy image
    kwargs_data['image_data'] = image_sim
    data_class.update_data(image_sim)
    
    return {
        'data_class': data_class,
        'psf_class': psf_class,
        'lightModel': lightModel,
        'pointSource': pointSource,
        'image_sim': image_sim,
        'kwargs_data': kwargs_data,
        'kwargs_psf': kwargs_psf,
        'kwargs_numerics': kwargs_numerics,
        'kwargs_host': kwargs_host,
        'kwargs_ps': kwargs_ps,
        'light_model_list': light_model_list,
        'point_source_list': point_source_list
    }