import numpy as np

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cupy = None

HAS_CUPY = True

try:
    from pyolaf.geometry import LFM_computeGeometryParameters, LFM_setCameraParams_v2
    from pyolaf.lf import LFM_computeLFMatrixOperators
    from pyolaf.transform import LFM_retrieveTransformation, format_transform, get_transformed_shape, transform_img
    from pyolaf.project import LFM_forwardProject, LFM_backwardProject
    from pyolaf.aliasing import lanczosfft, LFM_computeDepthAdaptiveWidth
    HAS_PYOLAF = True
except ImportError:
    HAS_PYOLAF = False

def forward_operator(volume, H, lenslet_centers, resolution, img_size, crange, use_gpu=False, step=10):
    """
    Computes the forward projection from volume to light field image.

    Args:
        volume: Input volume (numpy or cupy array).
        H: Forward projection operator.
        lenslet_centers (dict): Geometry information.
        resolution (dict): Resolution information.
        img_size: Size of the output image.
        crange: Camera range.
        use_gpu (bool): Whether to use GPU.
        step (int): Step parameter for projection.

    Returns:
        Projected light field image.
    """
    xp = cupy if (use_gpu and HAS_CUPY) else np
    volume_arr = xp.asarray(volume) if use_gpu and HAS_CUPY else np.asarray(volume)
    
    lf_image = LFM_forwardProject(H, volume_arr, lenslet_centers, resolution, img_size, crange, step=step)
    
    return lf_image
