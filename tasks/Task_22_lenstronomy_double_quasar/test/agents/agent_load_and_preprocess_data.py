import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.ImSim.image_model import ImageModel

def load_and_preprocess_data(
    background_rms=0.5,
    exp_time=100,
    numPix=100,
    deltaPix=0.05,
    fwhm=0.2,
    seed=None
):
    """
    Generate mock lensing data with noise.
    
    Returns:
        dict: Contains all data and configuration needed for fitting.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Transformation matrix: pixel coordinates -> angular coordinates
    transform_pix2angle = np.array([[-deltaPix, 0], [0, deltaPix]])
    
    # Calculate the RA/Dec of the pixel (0,0) such that the image is centered at (0,0)
    cx = (numPix - 1) / 2.
    cy = (numPix - 1) / 2.
    ra_at_xy_0 = -(transform_pix2angle[0, 0] * cx + transform_pix2angle[0, 1] * cy)
    dec_at_xy_0 = -(transform_pix2angle[1, 0] * cx + transform_pix2angle[1, 1] * cy)
    
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
    
    # Define Lens Model (EPL + Shear)
    lens_model_list = ['EPL', 'SHEAR']
    kwargs_spemd = {
        'theta_E': 1., 'gamma': 1.96, 'center_x': 0, 'center_y': 0,
        'e1': 0.07, 'e2': -0.03
    }
    kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}
    kwargs_lens_true = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    
    # Define Lens Light Model (Spherical Sersic)
    lens_light_model_list = ['SERSIC']
    kwargs_sersic = {
        'amp': 400., 'R_sersic': 1., 'n_sersic': 2,
        'center_x': 0, 'center_y': 0
    }
    kwargs_lens_light_true = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    
    # Define Source Light Model (Elliptical Sersic)
    source_model_list = ['SERSIC_ELLIPSE']
    ra_source, dec_source = 0.1, 0.3
    kwargs_sersic_ellipse = {
        'amp': 160, 'R_sersic': .5, 'n_sersic': 7,
        'center_x': ra_source, 'center_y': dec_source,
        'e1': 0., 'e2': 0.1
    }
    kwargs_source_true = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)
    
    # Solve for Image Positions
    lensEquationSolver = LensEquationSolver(lens_model_class)
    x_image, y_image = lensEquationSolver.findBrightImage(
        ra_source, dec_source, kwargs_lens_true, numImages=4,
        min_distance=deltaPix, search_window=numPix * deltaPix
    )
    print("Number of images found:", len(x_image))
    
    mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens_true)
    kwargs_ps_true = [{
        'ra_image': x_image,
        'dec_image': y_image,
        'point_amp': np.abs(mag) * 100
    }]
    point_source_list = ['LENSED_POSITION']
    point_source_class = PointSource(
        point_source_type_list=point_source_list,
        fixed_magnification_list=[False]
    )
    
    # Numerics
    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }
    
    # Create ImageModel for simulation
    imageModel = ImageModel(
        data_class, psf_class, lens_model_class, source_model_class,
        lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics
    )
    
    # Simulate clean image
    image_sim = imageModel.image(
        kwargs_lens_true, kwargs_source_true, kwargs_lens_light_true, kwargs_ps_true
    )
    
    # Add Poisson Noise
    image_sim_counts = image_sim * exp_time
    image_sim_counts[image_sim_counts < 0] = 0
    poisson_counts = np.random.poisson(image_sim_counts.astype(int))
    poisson = poisson_counts / exp_time
    poisson_noise = poisson - image_sim
    
    # Add Gaussian Background Noise
    bkg_noise = np.random.randn(*image_sim.shape) * background_rms
    
    # Combine
    image_noisy = image_sim + bkg_noise + poisson_noise
    
    # Update Data with Simulated Image
    kwargs_data['image_data'] = image_noisy
    data_class.update_data(image_noisy)
    
    print("Data generation complete.")
    
    return {
        'kwargs_data': kwargs_data,
        'kwargs_psf': kwargs_psf,
        'kwargs_numerics': kwargs_numerics,
        'data_class': data_class,
        'psf_class': psf_class,
        'lens_model_list': lens_model_list,
        'lens_model_class': lens_model_class,
        'source_model_list': source_model_list,
        'source_model_class': source_model_class,
        'lens_light_model_list': lens_light_model_list,
        'lens_light_model_class': lens_light_model_class,
        'point_source_list': point_source_list,
        'point_source_class': point_source_class,
        'x_image': x_image,
        'y_image': y_image,
        'kwargs_lens_true': kwargs_lens_true,
        'kwargs_source_true': kwargs_source_true,
        'kwargs_lens_light_true': kwargs_lens_light_true,
        'kwargs_ps_true': kwargs_ps_true,
        'image_noisy': image_noisy,
        'background_rms': background_rms,
        'exp_time': exp_time,
        'numPix': numPix,
        'deltaPix': deltaPix
    }