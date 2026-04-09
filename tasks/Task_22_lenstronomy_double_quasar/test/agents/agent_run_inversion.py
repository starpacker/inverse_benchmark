import numpy as np
import time
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

def run_inversion(data_dict, n_particles=50, n_iterations=10, n_burn=10, n_run=10):
    """
    Run the lens modeling inversion using PSO and MCMC.
    
    Args:
        data_dict: Dictionary containing all necessary data and configuration.
        n_particles: Number of PSO particles.
        n_iterations: Number of PSO iterations.
        n_burn: MCMC burn-in steps.
        n_run: MCMC sampling steps.
    
    Returns:
        dict: Contains best fit parameters and chain list.
    """
    x_image = data_dict['x_image']
    y_image = data_dict['y_image']
    
    # --- 1. Define Models ---
    kwargs_model = {
        'lens_model_list': data_dict['lens_model_list'],
        'source_light_model_list': data_dict['source_model_list'],
        'lens_light_model_list': data_dict['lens_light_model_list'],
        'point_source_model_list': data_dict['point_source_list'],
        'fixed_magnification_list': [False],
    }
    
    # --- 2. Define Constraints ---
    kwargs_constraints = {
        'joint_source_with_point_source': [[0, 0]],
        'num_point_source_list': [len(x_image)],
        'solver_type': 'THETA_E_PHI',
    }
    
    # --- 3. Define Likelihood Settings ---
    kwargs_likelihood = {
        'check_bounds': True,
        'force_no_add_image': False,
        'source_marg': False,
        'image_position_uncertainty': 0.004,
        'source_position_tolerance': 0.001
    }
    
    # --- 4. Prepare Data ---
    image_band = [
        data_dict['kwargs_data'],
        data_dict['kwargs_psf'],
        data_dict['kwargs_numerics']
    ]
    multi_band_list = [image_band]
    kwargs_data_joint = {
        'multi_band_list': multi_band_list,
        'multi_band_type': 'multi-linear'
    }
    
    # --- 5. Parameter Initialization & Priors ---
    kwargs_lens_init = [
        {'theta_E': 1.1, 'e1': 0, 'e2': 0, 'gamma': 2., 'center_x': 0., 'center_y': 0},
        {'gamma1': 0., 'gamma2': 0.}
    ]
    kwargs_lens_sigma = [
        {'theta_E': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': .1, 'center_x': 0.1, 'center_y': 0.1},
        {'gamma1': 0.1, 'gamma2': 0.1}
    ]
    kwargs_lower_lens = [
        {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10},
        {'gamma1': -.2, 'gamma2': -0.2}
    ]
    kwargs_upper_lens = [
        {'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10},
        {'gamma1': 0.2, 'gamma2': 0.2}
    ]
    
    kwargs_source_init = [
        {'R_sersic': 0.03, 'n_sersic': 1., 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}
    ]
    kwargs_source_sigma = [
        {'R_sersic': 0.2, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2}
    ]
    kwargs_lower_source = [
        {'R_sersic': 0.001, 'n_sersic': .5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}
    ]
    kwargs_upper_source = [
        {'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}
    ]
    
    kwargs_lens_light_init = [
        {'R_sersic': 0.1, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}
    ]
    kwargs_lens_light_sigma = [
        {'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0, 'e2': 0, 'center_x': .1, 'center_y': 0.1}
    ]
    kwargs_lower_lens_light = [
        {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}
    ]
    kwargs_upper_lens_light = [
        {'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}
    ]
    
    kwargs_ps_init = [{'ra_image': x_image + 0.01, 'dec_image': y_image - 0.01}]
    kwargs_ps_sigma = [
        {'ra_image': [0.02] * len(x_image), 'dec_image': [0.02] * len(x_image)}
    ]
    kwargs_lower_ps = [
        {'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}
    ]
    kwargs_upper_ps = [
        {'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}
    ]
    
    lens_params = [
        kwargs_lens_init, kwargs_lens_sigma,
        [{}, {'ra_0': 0, 'dec_0': 0}],
        kwargs_lower_lens, kwargs_upper_lens
    ]
    source_params = [
        kwargs_source_init, kwargs_source_sigma,
        [{}], kwargs_lower_source, kwargs_upper_source
    ]
    lens_light_params = [
        kwargs_lens_light_init, kwargs_lens_light_sigma,
        [{}], kwargs_lower_lens_light, kwargs_upper_lens_light
    ]
    ps_params = [
        kwargs_ps_init, kwargs_ps_sigma,
        [{}], kwargs_lower_ps, kwargs_upper_ps
    ]
    
    kwargs_params = {
        'lens_model': lens_params,
        'source_model': source_params,
        'lens_light_model': lens_light_params,
        'point_source_model': ps_params
    }
    
    # --- 6. Initialize Fitting Sequence ---
    fitting_seq = FittingSequence(
        kwargs_data_joint, kwargs_model, kwargs_constraints,
        kwargs_likelihood, kwargs_params
    )
    
    # --- 7. Define Workflow ---
    fitting_kwargs_list = [
        ['PSO', {'sigma_scale': 1., 'n_particles': n_particles, 'n_iterations': n_iterations}],
        ['MCMC', {'n_burn': n_burn, 'n_run': n_run, 'walkerRatio': 4, 'sigma_scale': .1}]
    ]
    
    print("Starting fitting sequence...")
    start_time = time.time()
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    end_time = time.time()
    print(f"Fitting completed in {end_time - start_time:.2f} seconds.")
    
    return {
        'kwargs_result': kwargs_result,
        'chain_list': chain_list,
        'fitting_time': end_time - start_time
    }

def create_mock_data():
    num_pix = 50
    delta_pix = 0.05
    kwargs_data = {
        'image_data': np.zeros((num_pix, num_pix)),
        'background_rms': 0.01,
        'exposure_time': 1000,
        'ra_at_xy_0': -1.25,
        'dec_at_xy_0': -1.25,
        'transform_pix2angle': np.array([[delta_pix, 0], [0, delta_pix]])
    }
    data_class = ImageData(**kwargs_data)
    kwargs_data['noise_map'] = np.ones((num_pix, num_pix)) * 0.01

    kwargs_psf = {
        'psf_type': 'GAUSSIAN',
        'fwhm': 0.1,
        'pixel_size': delta_pix
    }
    psf_class = PSF(**kwargs_psf)

    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }

    x_image = np.array([0.5, 0, -0.5, 0])
    y_image = np.array([0, 0.5, 0, -0.5])

    data_dict = {
        'x_image': x_image,
        'y_image': y_image,
        'lens_model_list': ['EPL', 'SHEAR'],
        'source_model_list': ['SERSIC'],
        'lens_light_model_list': ['SERSIC'],
        'point_source_list': ['LENSED_POSITION'],
        'kwargs_data': kwargs_data,
        'kwargs_psf': kwargs_psf,
        'kwargs_numerics': kwargs_numerics
    }
    
    return data_dict

def test_run_inversion():
    data_dict = create_mock_data()
    print("Running reproduction test...")
    result = run_inversion(
        data_dict, 
        n_particles=10, 
        n_iterations=2, 
        n_burn=2, 
        n_run=2
    )

    print("\n--- Test Results ---")
    print(f"Best fit Theta_E: {result['kwargs_result']['kwargs_lens'][0]['theta_E']:.4f}")
    print(f"Chain list length: {len(result['chain_list'])}")
    assert 'kwargs_result' in result
    assert 'chain_list' in result
    print("Test Passed: Code reproduced successfully.")