import numpy as np
import time
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence


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
    image_model_counts[image_model_counts < 0] = 0
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


def forward_operator(
    kwargs_lens: list,
    kwargs_source: list,
    kwargs_lens_light: list,
    data_class: ImageData,
    psf_class: PSF,
    lens_model_class: LensModel,
    source_model_class: LightModel,
    lens_light_model_class: LightModel,
    kwargs_numerics: dict
) -> np.ndarray:
    """
    Forward operator: compute model image given lens, source, and lens light parameters.
    
    This creates an ImageModel and computes the predicted image.
    """
    imageModel = ImageModel(
        data_class, psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        kwargs_numerics=kwargs_numerics
    )
    
    y_pred = imageModel.image(
        kwargs_lens, kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_ps=None
    )
    
    return y_pred


def run_inversion(preprocessed_data: dict, fitting_kwargs_list: list) -> dict:
    """
    Run the inversion/fitting sequence using lenstronomy's FittingSequence.
    
    Returns the best fit parameters and chain list.
    """
    # Extract needed data
    kwargs_data = preprocessed_data['kwargs_data']
    kwargs_psf = preprocessed_data['kwargs_psf']
    kwargs_numerics = preprocessed_data['kwargs_numerics']
    lens_model_list = preprocessed_data['lens_model_list']
    source_model_list = preprocessed_data['source_model_list']
    lens_light_model_list = preprocessed_data['lens_light_model_list']
    
    # Setup lens parameters
    fixed_lens = []
    kwargs_lens_init = []
    kwargs_lens_sigma = []
    kwargs_lower_lens = []
    kwargs_upper_lens = []
    
    # SIE parameters
    fixed_lens.append({})
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
    
    # SHEAR parameters
    fixed_lens.append({'ra_0': 0, 'dec_0': 0})
    kwargs_lens_init.append({'gamma1': 0., 'gamma2': 0.0})
    kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1})
    kwargs_lower_lens.append({'gamma1': -0.2, 'gamma2': -0.2})
    kwargs_upper_lens.append({'gamma1': 0.2, 'gamma2': 0.2})
    
    lens_params = [
        kwargs_lens_init, kwargs_lens_sigma, fixed_lens,
        kwargs_lower_lens, kwargs_upper_lens
    ]
    
    # Setup source parameters
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
    
    source_params = [
        kwargs_source_init, kwargs_source_sigma, fixed_source,
        kwargs_lower_source, kwargs_upper_source
    ]
    
    # Setup lens light parameters
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
    
    lens_light_params = [
        kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light,
        kwargs_lower_lens_light, kwargs_upper_lens_light
    ]
    
    # Combine parameters
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
    
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    kwargs_data_joint = {
        'multi_band_list': multi_band_list,
        'multi_band_type': 'single-band'
    }
    kwargs_constraints = {'linear_solver': True}
    
    # Create fitting sequence
    fitting_seq = FittingSequence(
        kwargs_data_joint, kwargs_model, kwargs_constraints,
        kwargs_likelihood, kwargs_params, verbose=True
    )
    
    print("Starting Fitting Sequence...")
    start_time = time.time()
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


def evaluate_results(
    inversion_result: dict,
    preprocessed_data: dict
) -> dict:
    """
    Evaluate the fitting results by comparing to true parameters and computing residuals.
    """
    kwargs_result = inversion_result['kwargs_result']
    
    print("Best Fit Result:")
    print(kwargs_result)
    
    # Extract fitted parameters
    fitted_kwargs_lens = kwargs_result.get('kwargs_lens', [])
    fitted_kwargs_source = kwargs_result.get('kwargs_source', [])
    fitted_kwargs_lens_light = kwargs_result.get('kwargs_lens_light', [])
    
    # Get true parameters
    true_kwargs_lens = preprocessed_data['true_kwargs_lens']
    true_kwargs_source = preprocessed_data['true_kwargs_source']
    true_kwargs_lens_light = preprocessed_data['true_kwargs_lens_light']
    
    # Compute model image with fitted parameters
    model_image = forward_operator(
        fitted_kwargs_lens,
        fitted_kwargs_source,
        fitted_kwargs_lens_light,
        preprocessed_data['data_class'],
        preprocessed_data['psf_class'],
        preprocessed_data['lens_model_class'],
        preprocessed_data['source_model_class'],
        preprocessed_data['lens_light_model_class'],
        preprocessed_data['kwargs_numerics']
    )
    
    # Compute residuals
    observed_image = preprocessed_data['image_data']
    residuals = observed_image - model_image
    
    # Compute chi-squared
    background_rms = preprocessed_data['background_rms']
    chi_squared = np.sum((residuals / background_rms) ** 2)
    reduced_chi_squared = chi_squared / (observed_image.size - 1)
    
    # Compare key lens parameters
    print("\n--- Parameter Comparison ---")
    if len(fitted_kwargs_lens) > 0 and len(true_kwargs_lens) > 0:
        print("Lens Model (SIE):")
        for key in ['theta_E', 'e1', 'e2', 'center_x', 'center_y']:
            if key in fitted_kwargs_lens[0] and key in true_kwargs_lens[0]:
                fitted_val = fitted_kwargs_lens[0][key]
                true_val = true_kwargs_lens[0][key]
                print(f"  {key}: True={true_val:.4f}, Fitted={fitted_val:.4f}, Diff={fitted_val - true_val:.4f}")
    
    if len(fitted_kwargs_lens) > 1 and len(true_kwargs_lens) > 1:
        print("Shear Model:")
        for key in ['gamma1', 'gamma2']:
            if key in fitted_kwargs_lens[1] and key in true_kwargs_lens[1]:
                fitted_val = fitted_kwargs_lens[1][key]
                true_val = true_kwargs_lens[1][key]
                print(f"  {key}: True={true_val:.4f}, Fitted={fitted_val:.4f}, Diff={fitted_val - true_val:.4f}")
    
    print(f"\nChi-squared: {chi_squared:.2f}")
    print(f"Reduced Chi-squared: {reduced_chi_squared:.4f}")
    print(f"Residual RMS: {np.std(residuals):.6f}")
    print(f"Fitting time: {inversion_result['fitting_time']:.2f} seconds")
    
    return {
        'model_image': model_image,
        'residuals': residuals,
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'residual_rms': np.std(residuals),
        'fitted_kwargs_lens': fitted_kwargs_lens,
        'fitted_kwargs_source': fitted_kwargs_source,
        'fitted_kwargs_lens_light': fitted_kwargs_lens_light
    }


if __name__ == '__main__':
    # Configuration parameters
    background_rms = 0.005
    exp_time = 500.0
    numPix = 60
    pixel_scale = 0.05
    fwhm = 0.05
    psf_type = 'GAUSSIAN'
    
    # Lens model configuration
    lens_model_list = ['SIE', 'SHEAR']
    kwargs_spemd = {
        'theta_E': 0.66, 'center_x': 0.05, 'center_y': 0,
        'e1': 0.07, 'e2': -0.03
    }
    kwargs_shear = {'gamma1': 0.0, 'gamma2': -0.05}
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    
    # Source model configuration
    source_model_list = ['SERSIC_ELLIPSE']
    kwargs_sersic = {
        'amp': 16, 'R_sersic': 0.1, 'n_sersic': 1,
        'e1': -0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0
    }
    kwargs_source = [kwargs_sersic]
    
    # Lens light model configuration
    lens_light_model_list = ['SERSIC_ELLIPSE']
    kwargs_sersic_lens = {
        'amp': 16, 'R_sersic': 0.6, 'n_sersic': 2,
        'e1': -0.1, 'e2': 0.1, 'center_x': 0.05, 'center_y': 0
    }
    kwargs_lens_light = [kwargs_sersic_lens]
    
    # Fitting configuration (reduced iterations for demonstration)
    fitting_kwargs_list = [
        ['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 10}],
        ['MCMC', {'n_burn': 10, 'n_run': 10, 'n_walkers': 50, 'sigma_scale': 0.1}]
    ]
    
    # Step 1: Load and preprocess data
    preprocessed_data = load_and_preprocess_data(
        numPix=numPix,
        pixel_scale=pixel_scale,
        background_rms=background_rms,
        exp_time=exp_time,
        fwhm=fwhm,
        psf_type=psf_type,
        lens_model_list=lens_model_list,
        kwargs_lens=kwargs_lens,
        source_model_list=source_model_list,
        kwargs_source=kwargs_source,
        lens_light_model_list=lens_light_model_list,
        kwargs_lens_light=kwargs_lens_light,
        random_seed=42
    )
    
    # Step 2: Run inversion/fitting
    inversion_result = run_inversion(preprocessed_data, fitting_kwargs_list)
    
    # Step 3: Evaluate results
    evaluation = evaluate_results(inversion_result, preprocessed_data)
    
    print("\nOPTIMIZATION_FINISHED_SUCCESSFULLY")