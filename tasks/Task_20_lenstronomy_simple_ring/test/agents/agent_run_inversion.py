import time
import numpy as np
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

def run_inversion(preprocessed_data: dict, fitting_kwargs_list: list) -> dict:
    """
    Run the inversion/fitting sequence using lenstronomy's FittingSequence.
    
    Args:
        preprocessed_data (dict): Dictionary containing data, PSF, numerics, 
                                  and model list definitions.
        fitting_kwargs_list (list): List of lists defining the optimization 
                                    routine (e.g., PSO settings, MCMC settings).

    Returns:
        dict: Contains best fit parameters ('kwargs_result'), the chain history 
              ('chain_list'), and the fitting sequence object.
    """
    # --- 1. Extract Dependencies ---
    kwargs_data = preprocessed_data['kwargs_data']
    kwargs_psf = preprocessed_data['kwargs_psf']
    kwargs_numerics = preprocessed_data['kwargs_numerics']
    
    # Model definitions (strings defining the profiles)
    lens_model_list = preprocessed_data['lens_model_list']
    source_model_list = preprocessed_data['source_model_list']
    lens_light_model_list = preprocessed_data['lens_light_model_list']
    
    # --- 2. Define Parameter Space ---
    # We must define 5 lists for each model component:
    # 1. init: Initial guess
    # 2. sigma: Spread of the initial particle cloud (for PSO)
    # 3. fixed: Parameters to hold constant
    # 4. lower: Hard lower bound
    # 5. upper: Hard upper bound
    
    # A. Lens Model Parameters (SIE + Shear)
    fixed_lens = []
    kwargs_lens_init = []
    kwargs_lens_sigma = []
    kwargs_lower_lens = []
    kwargs_upper_lens = []
    
    # Component 1: SIE (Singular Isothermal Ellipsoid)
    fixed_lens.append({}) # No fixed parameters for SIE
    kwargs_lens_init.append({
        'theta_E': 0.7, 'e1': 0., 'e2': 0.,
        'center_x': 0., 'center_y': 0.
    })
    kwargs_lens_sigma.append({
        'theta_E': 0.2, 'e1': 0.05, 'e2': 0.05,
        'center_x': 0.05, 'center_y': 0.05
    })
    kwargs_lower_lens.append({
        'theta_E': 0.01, 'e1': -0.5, 'e2': -0.5,
        'center_x': -10, 'center_y': -10
    })
    kwargs_upper_lens.append({
        'theta_E': 10., 'e1': 0.5, 'e2': 0.5,
        'center_x': 10, 'center_y': 10
    })
    
    # Component 2: External Shear
    fixed_lens.append({'ra_0': 0, 'dec_0': 0}) # Fix shear origin
    kwargs_lens_init.append({'gamma1': 0., 'gamma2': 0.0})
    kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1})
    kwargs_lower_lens.append({'gamma1': -0.2, 'gamma2': -0.2})
    kwargs_upper_lens.append({'gamma1': 0.2, 'gamma2': 0.2})
    
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, fixed_lens, kwargs_lower_lens, kwargs_upper_lens]
    
    # B. Source Light Parameters (Sersic)
    fixed_source = []
    kwargs_source_init = []
    kwargs_source_sigma = []
    kwargs_lower_source = []
    kwargs_upper_source = []
    
    fixed_source.append({})
    kwargs_source_init.append({
        'R_sersic': 0.2, 'n_sersic': 1, 'e1': 0, 'e2': 0,
        'center_x': 0., 'center_y': 0, 'amp': 16
    })
    kwargs_source_sigma.append({
        'n_sersic': 0.5, 'R_sersic': 0.1, 'e1': 0.05, 'e2': 0.05,
        'center_x': 0.2, 'center_y': 0.2, 'amp': 10
    })
    kwargs_lower_source.append({
        'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.001, 'n_sersic': 0.5,
        'center_x': -10, 'center_y': -10, 'amp': 0
    })
    kwargs_upper_source.append({
        'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 5.,
        'center_x': 10, 'center_y': 10, 'amp': 100
    })
    
    source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    
    # C. Lens Light Parameters (Sersic)
    fixed_lens_light = []
    kwargs_lens_light_init = []
    kwargs_lens_light_sigma = []
    kwargs_lower_lens_light = []
    kwargs_upper_lens_light = []
    
    fixed_lens_light.append({})
    kwargs_lens_light_init.append({
        'R_sersic': 0.5, 'n_sersic': 2, 'e1': 0, 'e2': 0,
        'center_x': 0., 'center_y': 0, 'amp': 16
    })
    kwargs_lens_light_sigma.append({
        'n_sersic': 1, 'R_sersic': 0.3, 'e1': 0.05, 'e2': 0.05,
        'center_x': 0.1, 'center_y': 0.1, 'amp': 10
    })
    kwargs_lower_lens_light.append({
        'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.001, 'n_sersic': 0.5,
        'center_x': -10, 'center_y': -10, 'amp': 0
    })
    kwargs_upper_lens_light.append({
        'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 5.,
        'center_x': 10, 'center_y': 10, 'amp': 100
    })
    
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
    
    # --- 3. Configure Fitting Sequence ---
    kwargs_params = {
        'lens_model': lens_params,
        'source_model': source_params,
        'lens_light_model': lens_light_params
    }
    
    kwargs_likelihood = {'source_marg': False}
    
    kwargs_model = {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list
    }
    
    # Pack data into the format required by FittingSequence
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    kwargs_data_joint = {
        'multi_band_list': multi_band_list,
        'multi_band_type': 'single-band'
    }
    
    # Linear solver allows us to solve for 'amp' parameters analytically
    kwargs_constraints = {'linear_solver': True}
    
    fitting_seq = FittingSequence(
        kwargs_data_joint, kwargs_model, kwargs_constraints,
        kwargs_likelihood, kwargs_params, verbose=True
    )
    
    # --- 4. Execute Fit ---
    print("Starting Fitting Sequence...")
    start_time = time.time()
    
    # This is the heavy lifting: running PSO or MCMC
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    
    kwargs_result = fitting_seq.best_fit()
    end_time = time.time()
    
    print(f"Fitting completed in {end_time - start_time:.2f} seconds.")
    
    return {
        'kwargs_result': kwargs_result,
        'chain_list': chain_list,
        'fitting_time': end_time - start_time,
        'fitting_seq': fitting_seq
    }

def test_reproduction():
    # 1. Mock Data Generation
    num_pix = 50  # Small number for fast testing
    delta_pix = 0.05
    
    # Define the coordinate grid
    kwargs_data = {
        'ra_at_xy_0': -num_pix/2 * delta_pix,
        'dec_at_xy_0': -num_pix/2 * delta_pix,
        'transform_pix2angle': np.array([[delta_pix, 0], [0, delta_pix]]),
        'image_data': np.zeros((num_pix, num_pix))  # Empty image (fitting noise)
    }
    
    # Define PSF
    kwargs_psf = {
        'psf_type': 'GAUSSIAN',
        'fwhm': 0.1,
        'pixel_size': delta_pix
    }
    
    # Define Numerics
    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }
    
    # 2. Define Model Lists
    # These must match the parameter setup inside run_inversion
    lens_model_list = ['SIE', 'SHEAR']
    source_model_list = ['SERSIC']
    lens_light_model_list = ['SERSIC']
    
    # 3. Pack into preprocessed_data
    preprocessed_data = {
        'kwargs_data': kwargs_data,
        'kwargs_psf': kwargs_psf,
        'kwargs_numerics': kwargs_numerics,
        'lens_model_list': lens_model_list,
        'source_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list
    }
    
    # 4. Define Fitting Strategy
    # We use a very short PSO chain just to verify the code runs
    fitting_kwargs_list = [
        ['PSO', {'sigma_scale': 1., 'n_particles': 10, 'n_iterations': 10}]
    ]
    
    # 5. Run the function
    print("Running reproduction test...")
    result = run_inversion(preprocessed_data, fitting_kwargs_list)
    
    # 6. Basic Assertions
    assert 'kwargs_result' in result
    assert 'chain_list' in result
    assert len(result['chain_list']) == 1 # One PSO run
    
    print("Test Passed! Code is reproducible.")
    print("Best fit theta_E:", result['kwargs_result']['kwargs_lens'][0]['theta_E'])

if __name__ == "__main__":
    test_reproduction()