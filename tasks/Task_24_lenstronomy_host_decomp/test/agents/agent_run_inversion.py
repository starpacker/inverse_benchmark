import time
import numpy as np
from lenstronomy.Workflow.fitting_sequence import FittingSequence

def run_inversion(data_class, psf_class, lightModel, pointSource, kwargs_data, kwargs_psf,
                  kwargs_numerics, light_model_list, point_source_list,
                  n_particles=50, n_iterations=50):
    """
    Run the inversion/fitting process using PSO optimization.
    
    Args:
        data_class: ImageData object
        psf_class: PSF object
        lightModel: LightModel object
        pointSource: PointSource object
        kwargs_data: data configuration
        kwargs_psf: PSF configuration
        kwargs_numerics: numerics configuration
        light_model_list: list of light model names
        point_source_list: list of point source model names
        n_particles: number of PSO particles
        n_iterations: number of PSO iterations
    
    Returns:
        dict containing:
            - kwargs_result: best fit parameters
            - chain_list: fitting chain
            - fitting_time: time taken for fitting
    """
    # Define models for fitting
    kwargs_model = {
        'lens_light_model_list': light_model_list,
        'point_source_model_list': point_source_list
    }
    
    # Constraints: Joint center for all components
    kwargs_constraints = {
        'joint_lens_light_with_lens_light': [[0, 1, ['center_x', 'center_y']]],
        'joint_lens_light_with_point_source': [[0, 0], [0, 1]],
        'num_point_source_list': [1]
    }
    
    kwargs_likelihood = {'check_bounds': True, 'source_marg': False}
    
    # Package data for FittingSequence
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    
    # --- Parameter Configuration ---
    
    # Initial Parameters for Host Galaxy
    fixed_source = []
    kwargs_source_init = []
    kwargs_source_sigma = []
    kwargs_lower_source = []
    kwargs_upper_source = []
    
    # 1. Disk Component (n=1 fixed)
    fixed_source.append({'n_sersic': 1})
    kwargs_source_init.append({'R_sersic': 1., 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0})
    kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
    kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10})
    kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10})
    
    # 2. Bulge Component (n=4 fixed)
    fixed_source.append({'n_sersic': 4})
    kwargs_source_init.append({'R_sersic': .5, 'n_sersic': 4, 'center_x': 0, 'center_y': 0})
    kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.3, 'center_x': 0.1, 'center_y': 0.1})
    kwargs_lower_source.append({'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10})
    kwargs_upper_source.append({'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10})
    
    source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    
    # Point Source Parameters
    fixed_ps = [{}]
    kwargs_ps_init = [{'ra_image': [0.0], 'dec_image': [0.0]}]
    kwargs_ps_sigma = [{'ra_image': [0.01], 'dec_image': [0.01]}]
    kwargs_lower_ps = [{'ra_image': [-10], 'dec_image': [-10]}]
    kwargs_upper_ps = [{'ra_image': [10], 'dec_image': [10]}]
    
    ps_param = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]
    
    # Combine parameters into the format required by FittingSequence
    kwargs_params = {
        'lens_light_model': source_params,
        'point_source_model': ps_param
    }
    
    # Initialize FittingSequence
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints,
                                   kwargs_likelihood, kwargs_params)
    
    # Define the optimization routine (Particle Swarm Optimization)
    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': n_particles, 'n_iterations': n_iterations}]]
    
    start_time = time.time()
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    end_time = time.time()
    fitting_time = end_time - start_time
    
    return {
        'kwargs_result': kwargs_result,
        'chain_list': chain_list,
        'fitting_time': fitting_time
    }