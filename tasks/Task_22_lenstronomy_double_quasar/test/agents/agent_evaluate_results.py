import numpy as np
from lenstronomy.ImSim.image_model import ImageModel

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