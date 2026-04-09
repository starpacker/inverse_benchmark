import numpy as np
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit

def evaluate_results(data_class, psf_class, lightModel, pointSource, kwargs_numerics,
                     kwargs_result, image_sim):
    """
    Evaluate the fitting results by computing the reconstructed image and comparing
    with the observed data.
    
    Args:
        data_class: ImageData object
        psf_class: PSF object
        lightModel: LightModel object
        pointSource: PointSource object
        kwargs_numerics: numerics configuration
        kwargs_result: best fit parameters from inversion
        image_sim: observed/simulated image
    
    Returns:
        dict containing:
            - image_reconstructed: reconstructed image
            - residual: difference between observed and reconstructed
            - reconstructed_sum: sum of reconstructed image
            - true_sum: sum of observed image
            - residual_rms: RMS of residuals
    """
    # Initialize the linear solver with the system configuration
    imageLinearFit = ImageLinearFit(
        data_class=data_class,
        psf_class=psf_class,
        lens_light_model_class=lightModel,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics
    )
    
    # Solve for the linear amplitudes to generate the model image
    image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(
        kwargs_lens_light=kwargs_result['kwargs_lens_light'],
        kwargs_ps=kwargs_result['kwargs_ps']
    )
    
    # Calculate metrics
    residual = image_sim - image_reconstructed
    reconstructed_sum = np.sum(image_reconstructed)
    true_sum = np.sum(image_sim)
    residual_rms = np.sqrt(np.mean(residual**2))
    
    return {
        'image_reconstructed': image_reconstructed,
        'residual': residual,
        'reconstructed_sum': reconstructed_sum,
        'true_sum': true_sum,
        'residual_rms': residual_rms
    }