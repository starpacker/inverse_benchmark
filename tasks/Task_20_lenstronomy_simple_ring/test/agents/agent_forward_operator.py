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