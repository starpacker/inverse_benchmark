import numpy as np
from lenstronomy.Workflow.fitting_sequence import FittingSequence

def run_inversion(data_dict):
    """
    Run the inversion/fitting procedure using PSO and MCMC.
    
    Args:
        data_dict: dictionary containing all data and model classes
    
    Returns:
        dict containing fitting results and chain list
    """
    # 1. Unpack Data and Model Components
    x_image = data_dict['x_image']
    y_image = data_dict['y_image']
    kwargs_data = data_dict['kwargs_data']
    kwargs_psf = data_dict['kwargs_psf']
    kwargs_numerics = data_dict['kwargs_numerics']
    lens_model_list = data_dict['lens_model_list']
    source_model_list = data_dict['source_model_list']
    lens_light_model_list = data_dict['lens_light_model_list']
    point_source_list = data_dict['point_source_list']
    
    # 2. Define Model Architecture
    kwargs_model = {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list,
        'point_source_model_list': point_source_list,
        'additional_images_list': [False],
        'fixed_magnification_list': [False]
    }
    
    # 3. Define Constraints
    kwargs_constraints = {
        'joint_source_with_point_source': [[0, 0]],
        'num_point_source_list': [4],
        'solver_type': 'PROFILE_SHEAR'
    }
    
    # 4. Define Likelihood Settings
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
    
    # 5. Prepare Data for FittingSequence
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    
    # 6. Define Parameter Space (Init, Sigma, Lower, Upper, Fixed)
    
    # Initial guesses
    kwargs_lens_init = [
        {'theta_E': 1.2, 'e1': 0, 'e2': 0, 'gamma': 2., 'center_x': 0., 'center_y': 0},
        {'gamma1': 0, 'gamma2': 0}
    ]
    kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
    kwargs_lens_light_init = [{'R_sersic': 0.1, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
    kwargs_ps_init = [{'ra_image': x_image + 0.01, 'dec_image': y_image - 0.01}]
    
    # Initial spread (sigma) for PSO particles
    kwargs_lens_sigma = [
        {'theta_E': 0.3, 'e1': 0.2, 'e2': 0.2, 'gamma': .2, 'center_x': 0.1, 'center_y': 0.1},
        {'gamma1': 0.1, 'gamma2': 0.1}
    ]
    kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2}]
    kwargs_lens_light_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': .1, 'center_y': 0.1}]
    kwargs_ps_sigma = [{'ra_image': [0.02] * 4, 'dec_image': [0.02] * 4}]
    
    # Hard bounds (Lower and Upper)
    kwargs_lower_lens = [
        {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10},
        {'gamma1': -0.5, 'gamma2': -0.5}
    ]
    kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
    kwargs_lower_lens_light = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
    kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]
    
    kwargs_upper_lens = [
        {'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10},
        {'gamma1': 0.5, 'gamma2': 0.5}
    ]
    kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
    kwargs_upper_lens_light = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
    kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]
    
    # Fixed parameters (not optimized)
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
    
    # 7. Initialize Fitting Sequence
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
    
    # 8. Define Sampling Strategy (PSO followed by MCMC)
    fitting_kwargs_list = [
        ['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 10}],
        ['MCMC', {'n_burn': 10, 'n_run': 10, 'n_walkers': 50, 'sigma_scale': .1}]
    ]
    
    # 9. Execute Fit
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    
    return {
        'kwargs_result': kwargs_result,
        'chain_list': chain_list,
        'fitting_seq': fitting_seq
    }