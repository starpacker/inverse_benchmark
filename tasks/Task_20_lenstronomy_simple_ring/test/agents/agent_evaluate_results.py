import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel

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