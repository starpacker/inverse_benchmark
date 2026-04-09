import numpy as np
import time
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence


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


def forward_operator(data_dict, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
    """
    Compute the forward model image given the model parameters.
    
    Args:
        data_dict: Dictionary containing model classes and data configuration.
        kwargs_lens: Lens model parameters.
        kwargs_source: Source light parameters.
        kwargs_lens_light: Lens light parameters.
        kwargs_ps: Point source parameters.
    
    Returns:
        np.ndarray: Predicted model image.
    """
    # Create ImageModel
    imageModel = ImageModel(
        data_dict['data_class'],
        data_dict['psf_class'],
        data_dict['lens_model_class'],
        data_dict['source_model_class'],
        data_dict['lens_light_model_class'],
        data_dict['point_source_class'],
        kwargs_numerics=data_dict['kwargs_numerics']
    )
    
    # Compute forward model
    image_model = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    
    return image_model


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
    
    # Define models for fitting
    kwargs_model = {
        'lens_model_list': data_dict['lens_model_list'],
        'source_light_model_list': data_dict['source_model_list'],
        'lens_light_model_list': data_dict['lens_light_model_list'],
        'point_source_model_list': data_dict['point_source_list'],
        'fixed_magnification_list': [False],
    }
    
    # Define constraints
    kwargs_constraints = {
        'joint_source_with_point_source': [[0, 0]],
        'num_point_source_list': [len(x_image)],
        'solver_type': 'THETA_E_PHI',
    }
    
    # Define Likelihood
    kwargs_likelihood = {
        'check_bounds': True,
        'force_no_add_image': False,
        'source_marg': False,
        'image_position_uncertainty': 0.004,
        'source_position_tolerance': 0.001
    }
    
    # Prepare Data for FittingSequence
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
    
    # Initial Params - Lens
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
    
    # Source
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
    
    # Lens Light
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
    
    # Point Source
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
    
    # Create FittingSequence
    fitting_seq = FittingSequence(
        kwargs_data_joint, kwargs_model, kwargs_constraints,
        kwargs_likelihood, kwargs_params
    )
    
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


def evaluate_results(data_dict, inversion_result):
    """
    Evaluate the inversion results by comparing to true parameters and computing residuals.
    
    Args:
        data_dict: Dictionary containing true parameters and data.
        inversion_result: Dictionary containing best fit parameters.
    
    Returns:
        dict: Evaluation metrics.
    """
    kwargs_result = inversion_result['kwargs_result']
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nBest fit result:")
    print(kwargs_result)
    
    # Extract fitted parameters
    kwargs_lens_fit = kwargs_result['kwargs_lens']
    kwargs_source_fit = kwargs_result['kwargs_source']
    kwargs_lens_light_fit = kwargs_result['kwargs_lens_light']
    kwargs_ps_fit = kwargs_result['kwargs_ps']
    
    # True parameters
    kwargs_lens_true = data_dict['kwargs_lens_true']
    kwargs_source_true = data_dict['kwargs_source_true']
    kwargs_lens_light_true = data_dict['kwargs_lens_light_true']
    
    # Compare lens parameters
    print("\n--- Lens Model Comparison ---")
    lens_params_to_compare = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lens_residuals = {}
    for param in lens_params_to_compare:
        if param in kwargs_lens_true[0] and param in kwargs_lens_fit[0]:
            true_val = kwargs_lens_true[0][param]
            fit_val = kwargs_lens_fit[0][param]
            residual = fit_val - true_val
            lens_residuals[param] = residual
            print(f"  {param}: True={true_val:.4f}, Fit={fit_val:.4f}, Residual={residual:.4f}")
    
    # Shear comparison
    print("\n--- Shear Comparison ---")
    shear_params = ['gamma1', 'gamma2']
    shear_residuals = {}
    for param in shear_params:
        if param in kwargs_lens_true[1] and param in kwargs_lens_fit[1]:
            true_val = kwargs_lens_true[1][param]
            fit_val = kwargs_lens_fit[1][param]
            residual = fit_val - true_val
            shear_residuals[param] = residual
            print(f"  {param}: True={true_val:.4f}, Fit={fit_val:.4f}, Residual={residual:.4f}")
    
    # Compute model image with best fit
    model_image = forward_operator(
        data_dict, kwargs_lens_fit, kwargs_source_fit, kwargs_lens_light_fit, kwargs_ps_fit
    )
    
    # Compute residual image
    observed_image = data_dict['image_noisy']
    residual_image = observed_image - model_image
    
    # Compute chi-squared
    background_rms = data_dict['background_rms']
    chi2 = np.sum((residual_image / background_rms) ** 2)
    reduced_chi2 = chi2 / (observed_image.size - 1)
    
    print(f"\n--- Image Residuals ---")
    print(f"  Chi-squared: {chi2:.2f}")
    print(f"  Reduced Chi-squared: {reduced_chi2:.4f}")
    print(f"  Residual RMS: {np.std(residual_image):.4f}")
    print(f"  Max absolute residual: {np.max(np.abs(residual_image)):.4f}")
    
    # Fitting time
    print(f"\n--- Performance ---")
    print(f"  Fitting time: {inversion_result['fitting_time']:.2f} seconds")
    
    return {
        'lens_residuals': lens_residuals,
        'shear_residuals': shear_residuals,
        'chi2': chi2,
        'reduced_chi2': reduced_chi2,
        'residual_rms': np.std(residual_image),
        'model_image': model_image,
        'residual_image': residual_image,
        'fitting_time': inversion_result['fitting_time']
    }


if __name__ == '__main__':
    # Step 1: Load and preprocess data
    data_dict = load_and_preprocess_data(
        background_rms=0.5,
        exp_time=100,
        numPix=100,
        deltaPix=0.05,
        fwhm=0.2,
        seed=42
    )
    
    # Step 2: Run inversion (PSO + MCMC)
    inversion_result = run_inversion(
        data_dict,
        n_particles=50,
        n_iterations=10,
        n_burn=10,
        n_run=10
    )
    
    # Step 3: Evaluate results
    evaluation = evaluate_results(data_dict, inversion_result)
    
    print("\nOPTIMIZATION_FINISHED_SUCCESSFULLY")