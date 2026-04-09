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