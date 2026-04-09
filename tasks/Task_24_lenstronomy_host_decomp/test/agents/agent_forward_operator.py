from lenstronomy.ImSim.image_model import ImageModel

def forward_operator(data_class, psf_class, lightModel, pointSource, kwargs_numerics,
                     kwargs_lens_light, kwargs_ps):
    """
    Forward operator: given model parameters, compute the predicted image.
    
    Args:
        data_class: ImageData object
        psf_class: PSF object
        lightModel: LightModel object
        pointSource: PointSource object
        kwargs_numerics: numerics configuration
        kwargs_lens_light: lens light (host galaxy) parameters
        kwargs_ps: point source parameters
    
    Returns:
        y_pred: predicted image as numpy array
    """
    # Step 1: Instantiate the ImageModel with the provided classes and numerics
    imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                            point_source_class=pointSource, kwargs_numerics=kwargs_numerics)
    
    # Step 2: Generate the image using the specific parameters for light and point sources
    y_pred = imageModel.image(kwargs_lens_light=kwargs_lens_light, kwargs_ps=kwargs_ps)
    
    return y_pred