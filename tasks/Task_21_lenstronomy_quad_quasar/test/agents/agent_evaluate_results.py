import numpy as np
from lenstronomy.ImSim.image_model import ImageModel

def forward_operator(params, image_model, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
    """
    Forward operator: compute the predicted image given model parameters.
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
    
    # Compute the forward model using the lenstronomy ImageModel class
    y_pred = image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    
    return y_pred

def evaluate_results(data_dict, inversion_result):
    """
    Evaluate the fitting results by comparing with true parameters and computing residuals.
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
        None, # No dynamic updates needed, we pass the fitted kwargs directly
        image_model,
        kwargs_lens_fit,
        kwargs_source_fit,
        kwargs_lens_light_fit,
        kwargs_ps_fit
    )
    
    # Compute residuals
    residuals = image_sim - image_reconstructed
    residual_rms = np.sqrt(np.mean(residuals**2))
    
    # Compare key parameters (assuming the main lens is at index 0)
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