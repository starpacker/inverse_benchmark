import numpy as np

def forward_operator(x_params, image_shape, background_image=None):
    """
    Generates an image from a list of Gaussian parameters.
    x_params: List of [background, height, center_x, center_y, width]
    image_shape: tuple (h, w)
    background_image: Optional 2D array representing global background.
    
    Returns: y_pred (simulated image)
    """
    h, w = image_shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    y_pred = np.zeros(image_shape)
    
    if background_image is not None:
        y_pred += background_image

    for p in x_params:
        # p: [local_bg, height, cx, cy, wid]
        # We assume local_bg is handled by the global background_image for the full forward model,
        # or we just render the gaussian lobes here.
        # Following the logic of the original code's generate_gaussian_image:
        # Unpack
        _, h_val, cx, cy, wid = p
        
        # Render this gaussian lobe
        g = h_val * np.exp(-2 * (((cx - x_grid) / wid) ** 2 + ((cy - y_grid) / wid) ** 2))
        y_pred += g
        
    return y_pred
