import numpy as np

def symmetric_gaussian_2d(xy, background, height, center_x, center_y, width):
    """
    Explicit mathematical definition of a 2D Symmetric Gaussian.
    f(x,y) = background + height * exp( -2 * ( ((x-cx)/w)^2 + ((y-cy)/w)^2 ) )
    """
    x, y = xy
    g = background + height * np.exp(-2 * (((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2))
    return g.ravel()
