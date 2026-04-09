import numpy as np

import scipy.ndimage

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

def _center_image(IM, odd_size=True, square=True):
    rows, cols = IM.shape
    if odd_size and cols % 2 == 0:
        IM = IM[:, :-1]
        rows, cols = IM.shape
    if square and rows != cols:
        if rows > cols:
            diff = rows - cols
            trim = diff // 2
            if trim > 0:
                IM = IM[trim: -trim]
            if diff % 2: IM = IM[: -1]
        else:
            if odd_size and rows % 2 == 0:
                IM = IM[:-1, :]
                rows -= 1
            xs = (cols - rows) // 2
            if xs > 0:
                IM = IM[:, xs:-xs]
        rows, cols = IM.shape

    origin = _find_origin_by_convolution(IM)
    
    # set_center logic merged here for conciseness and scope safety
    center = np.array(IM.shape) // 2
    
    # Check if origin is close to integer for precise shift
    if all(abs(o - round(o)) < 1e-3 for o in origin):
        origin = [int(round(o)) for o in origin]
        out = np.zeros_like(IM)
        src = [slice(None), slice(None)]
        dst = [slice(None), slice(None)]
        for a in range(2):
            d = int(center[a] - origin[a])
            if d > 0:
                dst[a] = slice(d, IM.shape[a])
                src[a] = slice(0, IM.shape[a] - d)
            elif d < 0:
                dst[a] = slice(0, IM.shape[a] + d)
                src[a] = slice(-d, IM.shape[a])
        out[tuple(dst)] = IM[tuple(src)]
        return out
    else:
        delta = [center[a] - origin[a] for a in range(2)]
        return scipy.ndimage.shift(IM, delta, order=3, mode='constant', cval=0.0)
