import numpy as np
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

def forward_operator(
    shapelet_coeffs: np.ndarray,
    data_class: ImageData,
    psf_class: PSF,
    lens_model_class: LensModel,
    kwargs_lens: list,
    kwargs_source_template: list,
    kwargs_numerics: dict
) -> np.ndarray:
    """
    Forward operator: Given shapelet coefficients, compute the predicted lensed image.
    
    This implements the forward model: source -> lensing -> convolution with PSF -> image.
    """
    # Create source model for shapelets
    source_model_list = ['SHAPELETS']
    source_model_class = LightModel(light_model_list=source_model_list)
    
    # No lens light
    lens_light_model_class = LightModel(light_model_list=[])
    
    # Create ImageModel
    imageModel = ImageModel(
        data_class, psf_class, lens_model_class, source_model_class,
        lens_light_model_class, kwargs_numerics=kwargs_numerics
    )
    
    # Build kwargs_source with amplitudes
    n_max = kwargs_source_template[0]['n_max']
    beta = kwargs_source_template[0]['beta']
    center_x = kwargs_source_template[0]['center_x']
    center_y = kwargs_source_template[0]['center_y']
    
    # The ShapeletSet uses 'amp' as a list of coefficients
    kwargs_source = [{
        'n_max': n_max,
        'beta': beta,
        'center_x': center_x,
        'center_y': center_y,
        'amp': shapelet_coeffs
    }]
    
    # Compute predicted image
    y_pred = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
    
    return y_pred