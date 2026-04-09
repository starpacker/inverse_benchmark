import numpy as np
from lenstronomy.ImSim.image_model import ImageModel

def forward_operator(params, image_model, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
    """
    Forward operator: compute the predicted image given model parameters.
    
    Args:
        params: dictionary of parameters to update (optional, can be None to use defaults)
        image_model: ImageModel instance
        kwargs_lens: lens model parameters
        kwargs_source: source model parameters
        kwargs_lens_light: lens light model parameters
        kwargs_ps: point source parameters
    
    Returns:
        y_pred: predicted image as numpy array
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
    
    # Compute the forward model
    y_pred = image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
    
    return y_pred