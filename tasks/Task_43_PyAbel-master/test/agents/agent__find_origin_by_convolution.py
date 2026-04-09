import numpy as np

def _find_origin_by_convolution(IM, axes=(0, 1)):
    """
    Find the image origin as the maximum of autoconvolution of its projections.
    """
    if isinstance(axes, int):
        axes = [axes]
    conv = [None, None]
    origin = [IM.shape[0] // 2, IM.shape[1] // 2]
    for a in axes:
        proj = IM.sum(axis=1 - a)
        if proj.size == 0:
             continue
        conv[a] = np.convolve(proj, proj, mode='full')
        origin[a] = np.argmax(conv[a]) / 2
    return tuple(origin)
