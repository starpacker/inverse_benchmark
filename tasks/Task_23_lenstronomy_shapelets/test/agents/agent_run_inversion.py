import numpy as np
import time
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit

def run_inversion(
    data_class: ImageData,
    psf_class: PSF,
    lens_model_class: LensModel,
    kwargs_lens: list,
    kwargs_numerics: dict,
    n_max: int,
    beta: float,
    center_x: float = 0.0,
    center_y: float = 0.0
) -> dict:
    """
    Run the linear inversion to reconstruct the source using shapelets.
    
    Uses lenstronomy's ImageLinearFit to solve for shapelet coefficients.
    
    Parameters:
        data_class: Configured ImageData object containing the observation.
        psf_class: Configured PSF object.
        lens_model_class: Configured LensModel object.
        kwargs_lens: List of dictionaries containing lens mass parameters.
        kwargs_numerics: Dictionary of numerical settings (e.g., supersampling).
        n_max: Maximum order of the Shapelet expansion.
        beta: Scale radius of the Shapelet basis.
        center_x: Centroid of the Shapelet basis in source plane.
        center_y: Centroid of the Shapelet basis in source plane.
        
    Returns:
        Dictionary containing the reconstructed model image, coefficients, and statistics.
    """
    # 1. Define Reconstruction Model (Shapelets)
    source_model_list_reconstruct = ['SHAPELETS']
    source_model_class_reconstruct = LightModel(light_model_list=source_model_list_reconstruct)
    
    # 2. Initialize ImageLinearFit
    imageLinearFit = ImageLinearFit(
        data_class=data_class,
        psf_class=psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class_reconstruct,
        kwargs_numerics=kwargs_numerics
    )
    
    # 3. Constraints for Shapelets (center and scale)
    kwargs_source_reconstruct = [{
        'n_max': n_max,
        'beta': beta,
        'center_x': center_x,
        'center_y': center_y
    }]
    
    start_time = time.time()
    
    # 4. Solve the Linear System
    wls_model, error_map, cov_param, param = imageLinearFit.image_linear_solve(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source_reconstruct,
        kwargs_lens_light=None,
        kwargs_ps=None,
        inv_bool=False
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 5. Calculate Reduced Chi-Square
    chi2_reduced = imageLinearFit.reduced_chi2(wls_model, error_map)
    
    # 6. Extract Shapelet Coefficients
    shapelet_coeffs = np.array(param)
    
    return {
        'model_image': wls_model,
        'error_map': error_map,
        'cov_param': cov_param,
        'shapelet_coeffs': shapelet_coeffs,
        'chi2_reduced': chi2_reduced,
        'elapsed_time': elapsed_time,
        'kwargs_source_reconstruct': kwargs_source_reconstruct,
        'n_max': n_max,
        'beta': beta
    }