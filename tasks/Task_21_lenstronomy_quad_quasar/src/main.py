import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.Util import util
from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Workflow.fitting_sequence import FittingSequence


def load_and_preprocess_data(background_rms, exp_time, numPix, deltaPix, fwhm):
    """
    Load and preprocess data: create simulation parameters, generate lensed image with noise.
    
    Returns:
        dict containing all necessary data and model classes for fitting
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


def forward_operator(params, image_model, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
    """
    Forward operator: compute the predicted image given model parameters.
    
    Args:
        params: dictionary of parameters to update (optional, can be None to use defaults)
        image_model: ImageModel instance
        kwargs_lens: lens model parameters
        kwargs_source: source model parameters
        kwargs_lens_light: lens light model parameters
        kwargs_ps: point source parameters
    
    Returns:
        y_pred: predicted image as numpy array
    """
    # If params provided, update the kwargs
    if params is not None:
        if 'lens' in params:
            for i, p in enumerate(params['lens']):
                kwargs_lens[i].update(p)
        if 'source' in params:
            for i, p in enumerate(params['source']):
                kwargs_source[i].update(p)
        if 'lens_light' in params:
            for i, p in enumerate(params['lens_light']):
                kwargs_lens_light[i].update(p)
        if 'ps' in params:
            for i, p in enumerate(params['ps']):
                kwargs_ps[i].update(p)
    
    # Compute the forward model
    y_pred = image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    
    return y_pred


def run_inversion(data_dict):
    """
    Run the inversion/fitting procedure using PSO and MCMC.
    
    Args:
        data_dict: dictionary containing all data and model classes
    
    Returns:
        dict containing fitting results and chain list
    """
    x_image = data_dict['x_image']
    y_image = data_dict['y_image']
    kwargs_data = data_dict['kwargs_data']
    kwargs_psf = data_dict['kwargs_psf']
    kwargs_numerics = data_dict['kwargs_numerics']
    lens_model_list = data_dict['lens_model_list']
    source_model_list = data_dict['source_model_list']
    lens_light_model_list = data_dict['lens_light_model_list']
    point_source_list = data_dict['point_source_list']
    
    # Model setup
    kwargs_model = {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list,
        'point_source_model_list': point_source_list,
        'additional_images_list': [False],
        'fixed_magnification_list': [False]
    }
    
    kwargs_constraints = {
        'joint_source_with_point_source': [[0, 0]],
        'num_point_source_list': [4],
        'solver_type': 'PROFILE_SHEAR'
    }
    
    prior_lens = [[0, 'e1', 0, 0.2], [0, 'e2', 0, 0.2]]
    kwargs_likelihood = {
        'check_bounds': True,
        'force_no_add_image': False,
        'source_marg': False,
        'image_position_uncertainty': 0.004,
        'source_position_tolerance': 0.001,
        'source_position_sigma': 0.001,
        'prior_lens': prior_lens
    }
    
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    
    # Initial parameters
    kwargs_lens_init = [
        {'theta_E': 1.2, 'e1': 0, 'e2': 0, 'gamma': 2., 'center_x': 0., 'center_y': 0},
        {'gamma1': 0, 'gamma2': 0}
    ]
    kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
    kwargs_lens_light_init = [{'R_sersic': 0.1, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
    kwargs_ps_init = [{'ra_image': x_image + 0.01, 'dec_image': y_image - 0.01}]
    
    # Sigma parameters
    kwargs_lens_sigma = [
        {'theta_E': 0.3, 'e1': 0.2, 'e2': 0.2, 'gamma': .2, 'center_x': 0.1, 'center_y': 0.1},
        {'gamma1': 0.1, 'gamma2': 0.1}
    ]
    kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2}]
    kwargs_lens_light_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': .1, 'center_y': 0.1}]
    kwargs_ps_sigma = [{'ra_image': [0.02] * 4, 'dec_image': [0.02] * 4}]
    
    # Lower bounds
    kwargs_lower_lens = [
        {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10},
        {'gamma1': -0.5, 'gamma2': -0.5}
    ]
    kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
    kwargs_lower_lens_light = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
    kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]
    
    # Upper bounds
    kwargs_upper_lens = [
        {'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10},
        {'gamma1': 0.5, 'gamma2': 0.5}
    ]
    kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
    kwargs_upper_lens_light = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
    kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]
    
    # Fixed parameters
    kwargs_lens_fixed = [{}, {'ra_0': 0, 'dec_0': 0}]
    kwargs_source_fixed = [{}]
    kwargs_lens_light_fixed = [{}]
    kwargs_ps_fixed = [{}]
    
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source]
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light, kwargs_upper_lens_light]
    ps_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_lower_ps, kwargs_upper_ps]
    
    kwargs_params = {
        'lens_model': lens_params,
        'source_model': source_params,
        'lens_light_model': lens_light_params,
        'point_source_model': ps_params
    }
    
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
    
    fitting_kwargs_list = [
        ['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 10}],
        ['MCMC', {'n_burn': 10, 'n_run': 10, 'n_walkers': 50, 'sigma_scale': .1}]
    ]
    
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    
    return {
        'kwargs_result': kwargs_result,
        'chain_list': chain_list,
        'fitting_seq': fitting_seq
    }


def evaluate_results(data_dict, inversion_result):
    """
    Evaluate the fitting results by comparing with true parameters and computing residuals.
    
    Args:
        data_dict: dictionary containing all data and true parameters
        inversion_result: dictionary containing fitting results
    
    Returns:
        dict containing evaluation metrics
    """
    kwargs_result = inversion_result['kwargs_result']
    image_model = data_dict['image_model']
    image_sim = data_dict['image_sim']
    
    kwargs_lens_true = data_dict['kwargs_lens_true']
    kwargs_source_true = data_dict['kwargs_source_true']
    kwargs_lens_light_true = data_dict['kwargs_lens_light_true']
    
    # Extract fitted parameters
    kwargs_lens_fit = kwargs_result['kwargs_lens']
    kwargs_source_fit = kwargs_result['kwargs_source']
    kwargs_lens_light_fit = kwargs_result['kwargs_lens_light']
    kwargs_ps_fit = kwargs_result['kwargs_ps']
    
    # Compute model prediction with fitted parameters
    image_reconstructed = forward_operator(
        None,
        image_model,
        kwargs_lens_fit,
        kwargs_source_fit,
        kwargs_lens_light_fit,
        kwargs_ps_fit
    )
    
    # Compute residuals
    residuals = image_sim - image_reconstructed
    residual_rms = np.sqrt(np.mean(residuals**2))
    
    # Compare key parameters
    theta_E_true = kwargs_lens_true[0]['theta_E']
    theta_E_fit = kwargs_lens_fit[0]['theta_E']
    theta_E_error = np.abs(theta_E_fit - theta_E_true)
    
    gamma_true = kwargs_lens_true[0]['gamma']
    gamma_fit = kwargs_lens_fit[0]['gamma']
    gamma_error = np.abs(gamma_fit - gamma_true)
    
    e1_true = kwargs_lens_true[0]['e1']
    e1_fit = kwargs_lens_fit[0]['e1']
    e1_error = np.abs(e1_fit - e1_true)
    
    e2_true = kwargs_lens_true[0]['e2']
    e2_fit = kwargs_lens_fit[0]['e2']
    e2_error = np.abs(e2_fit - e2_true)
    
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    print(f"Residual RMS: {residual_rms:.6f}")
    print(f"\nLens Parameters Comparison:")
    print(f"  theta_E: True={theta_E_true:.4f}, Fit={theta_E_fit:.4f}, Error={theta_E_error:.4f}")
    print(f"  gamma: True={gamma_true:.4f}, Fit={gamma_fit:.4f}, Error={gamma_error:.4f}")
    print(f"  e1: True={e1_true:.4f}, Fit={e1_fit:.4f}, Error={e1_error:.4f}")
    print(f"  e2: True={e2_true:.4f}, Fit={e2_fit:.4f}, Error={e2_error:.4f}")
    
    return {
        'residual_rms': residual_rms,
        'theta_E_error': theta_E_error,
        'gamma_error': gamma_error,
        'e1_error': e1_error,
        'e2_error': e2_error,
        'image_reconstructed': image_reconstructed,
        'residuals': residuals,
        'kwargs_lens_fit': kwargs_lens_fit,
        'kwargs_source_fit': kwargs_source_fit,
        'kwargs_lens_light_fit': kwargs_lens_light_fit,
        'kwargs_ps_fit': kwargs_ps_fit
    }


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulation parameters
    background_rms = 0.5
    exp_time = 100
    numPix = 100
    deltaPix = 0.05
    fwhm = 0.1
    
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(background_rms, exp_time, numPix, deltaPix, fwhm)
    print("Simulation done.")
    
    # Step 2: Test forward operator
    print("\nTesting forward operator...")
    y_pred = forward_operator(
        None,
        data_dict['image_model'],
        data_dict['kwargs_lens_true'].copy(),
        data_dict['kwargs_source_true'].copy(),
        data_dict['kwargs_lens_light_true'].copy(),
        data_dict['kwargs_ps_true'].copy()
    )
    print(f"Forward model output shape: {y_pred.shape}")
    
    # Step 3: Run inversion
    print("\nStarting Fitting...")
    inversion_result = run_inversion(data_dict)
    print("Fitting done.")
    print(inversion_result['kwargs_result'])
    
    # Step 4: Evaluate results
    evaluation = evaluate_results(data_dict, inversion_result)
    
    print("\nOPTIMIZATION_FINISHED_SUCCESSFULLY")