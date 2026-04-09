import numpy as np
from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Workflow.fitting_sequence import FittingSequence

# --- Function 1: Simulation (From Section 3) ---
def load_and_preprocess_data(background_rms, exp_time, numPix, deltaPix, fwhm):
    """
    Load and preprocess data: create simulation parameters, generate lensed image with noise.
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
    kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 5}
    psf_class = PSF(**kwargs_psf)
    
    # Lens model
    lens_model_list = ['EPL', 'SHEAR']
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi=0.1, gamma=0.02)
    kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}
    kwargs_pemd = {'theta_E': 1., 'gamma': 1.96, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.2}
    kwargs_lens = [kwargs_pemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    
    # Lens light model
    lens_light_model_list = ['SERSIC']
    kwargs_sersic = {'amp': 400, 'R_sersic': 1., 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    
    # Source model
    source_model_list = ['SERSIC_ELLIPSE']
    ra_source, dec_source = 0., 0.1
    kwargs_sersic_ellipse = {'amp': 4000., 'R_sersic': .1, 'n_sersic': 3, 'center_x': ra_source,
                             'center_y': dec_source, 'e1': -0.1, 'e2': 0.01}
    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)
    
    # Point source
    lensEquationSolver = LensEquationSolver(lens_model_class)
    x_image, y_image = lensEquationSolver.findBrightImage(ra_source, dec_source, kwargs_lens, numImages=4,
                                                          min_distance=deltaPix, search_window=numPix * deltaPix)
    mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
    mag = np.abs(mag)
    mag_pert = np.random.normal(mag, 0.5, len(mag))
    point_amp = mag_pert * 100
    kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image, 'point_amp': point_amp}]
    
    point_source_list = ['LENSED_POSITION']
    point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    
    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                            lens_light_model_class,
                            point_source_class, kwargs_numerics=kwargs_numerics)
    
    # Generate simulated image
    image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    
    # Add Poisson Noise
    image_sim_counts = image_sim * exp_time
    image_sim_counts[image_sim_counts < 0] = 0
    poisson_counts = np.random.poisson(image_sim_counts)
    poisson = poisson_counts / exp_time
    poisson_noise = poisson - image_sim
    
    # Add Gaussian Background Noise
    bkg_noise = np.random.randn(*image_sim.shape) * background_rms
    
    # Combine
    image_sim = image_sim + bkg_noise + poisson_noise
    
    kwargs_data['image_data'] = image_sim
    data_class.update_data(image_sim)
    
    return {
        'data_class': data_class,
        'psf_class': psf_class,
        'lens_model_class': lens_model_class,
        'source_model_class': source_model_class,
        'lens_light_model_class': lens_light_model_class,
        'point_source_class': point_source_class,
        'image_model': imageModel,
        'kwargs_data': kwargs_data,
        'kwargs_psf': kwargs_psf,
        'kwargs_numerics': kwargs_numerics,
        'kwargs_lens_true': kwargs_lens,
        'kwargs_source_true': kwargs_source,
        'kwargs_lens_light_true': kwargs_lens_light,
        'kwargs_ps_true': kwargs_ps,
        'x_image': x_image,
        'y_image': y_image,
        'lens_model_list': lens_model_list,
        'source_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list,
        'point_source_list': point_source_list,
        'image_sim': image_sim,
        'numPix': numPix,
        'deltaPix': deltaPix
    }

# --- Function 2: Fitting Setup (From Section 4) ---
def setup_fitting_sequence(sim_data):
    """
    Sets up the FittingSequence for the lens modeling.

    Args:
        sim_data (dict): Output from load_and_preprocess_data containing data and model classes.

    Returns:
        FittingSequence: The initialized fitting sequence object ready for fitting.
    """
    # Unpack necessary data from simulation dictionary
    kwargs_data = sim_data['kwargs_data']
    kwargs_psf = sim_data['kwargs_psf']
    kwargs_numerics = sim_data['kwargs_numerics']
    lens_model_list = sim_data['lens_model_list']
    source_model_list = sim_data['source_model_list']
    lens_light_model_list = sim_data['lens_light_model_list']
    point_source_list = sim_data['point_source_list']
    
    # 1. Define Parameter Space
    fixed_lens = [{'gamma': 1.96}, {'ra_0': 0, 'dec_0': 0}] 
    sigma_lens = [{'theta_E': 0.1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
                  {'gamma1': 0.1, 'gamma2': 0.1}]
    lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10},
                  {'gamma1': -0.5, 'gamma2': -0.5}]
    upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
                  {'gamma1': 0.5, 'gamma2': 0.5}]
    
    fixed_source = []
    sigma_source = [{'R_sersic': 0.05, 'n_sersic': 0.5, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.1, 'e2': 0.1}]
    lower_source = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10, 'e1': -0.5, 'e2': -0.5}]
    upper_source = [{'R_sersic': 10, 'n_sersic': 5, 'center_x': 10, 'center_y': 10, 'e1': 0.5, 'e2': 0.5}]
    
    fixed_lens_light = []
    sigma_lens_light = [{'R_sersic': 0.05, 'n_sersic': 0.5, 'center_x': 0.1, 'center_y': 0.1}]
    lower_lens_light = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10}]
    upper_lens_light = [{'R_sersic': 10, 'n_sersic': 5, 'center_x': 10, 'center_y': 10}]
    
    x_image = sim_data['x_image']
    y_image = sim_data['y_image']
    fixed_ps = []
    sigma_ps = [{'ra_image': [0.01]*4, 'dec_image': [0.01]*4}]
    lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]
    upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]

    lens_params = [sim_data['kwargs_lens_true'], sigma_lens, fixed_lens, lower_lens, upper_lens]
    source_params = [sim_data['kwargs_source_true'], sigma_source, fixed_source, lower_source, upper_source]
    lens_light_params = [sim_data['kwargs_lens_light_true'], sigma_lens_light, fixed_lens_light, lower_lens_light, upper_lens_light]
    ps_params = [sim_data['kwargs_ps_true'], sigma_ps, fixed_ps, lower_ps, upper_ps]

    kwargs_params = {
        'lens_model': lens_params,
        'source_model': source_params,
        'lens_light_model': lens_light_params,
        'point_source_model': ps_params
    }

    # 2. Define Model Settings
    kwargs_model = {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list,
        'point_source_model_list': point_source_list
    }

    # 3. Define Constraints
    kwargs_constraints = {
        'joint_source_with_point_source': [[0, 0]],
        'num_point_source_list': [4]
    }

    # 4. Define Likelihood Settings
    kwargs_likelihood = {
        'check_bounds': True,
        'source_marg': False,
        'check_matched_source_position': True,
        'source_position_tolerance': 0.001
    }

    # 5. Prepare Data for FittingSequence
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    kwargs_data_joint = {
        'multi_band_list': multi_band_list,
        'multi_band_type': 'multi-linear'
    }

    # 6. Initialize FittingSequence
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, 
                                  kwargs_likelihood, kwargs_params)
    
    return fitting_seq