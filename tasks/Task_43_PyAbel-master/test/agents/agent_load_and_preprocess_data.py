import bz2

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

def _get_image_quadrants(IM, reorient=True, symmetry_axis=None):
    IM = np.atleast_2d(IM)
    n, m = IM.shape
    n_c = n // 2 + n % 2
    m_c = m // 2 + m % 2

    Q0 = IM[:n_c, -m_c:]
    Q1 = IM[:n_c, :m_c]
    Q2 = IM[-n_c:, :m_c]
    Q3 = IM[-n_c:, -m_c:]

    if reorient:
        Q1 = np.fliplr(Q1)
        Q3 = np.flipud(Q3)
        Q2 = np.fliplr(np.flipud(Q2))

    # Average symmetrization
    if symmetry_axis==(0, 1):
        Q = (Q0 + Q1 + Q2 + Q3)/4.0
        return Q, Q, Q, Q
    return Q0, Q1, Q2, Q3

def load_and_preprocess_data(file_path):
    """
    Loads text/bz2 data, centers it, splits into quadrants, 
    and returns the top-right quadrant (Q0) for processing.
    """
    print(f"Loading data from {file_path}...")
    
    # 1. Load
    if file_path.endswith('.bz2'):
        with bz2.open(file_path, 'rt') as f:
            raw_im = np.loadtxt(f)
    else:
        raw_im = np.loadtxt(file_path)

    # 2. Center
    # Use odd_size=True, square=True as per original workflow
    centered_im = _center_image(raw_im, odd_size=True, square=True)
    
    # 3. Quadrants
    # Symmetry axis (0,1) averages all 4 quadrants into one representation
    # This effectively boosts SNR for the inversion
    Q_tuple = _get_image_quadrants(centered_im, reorient=True, symmetry_axis=(0, 1))
    
    # Return the processed quadrant (Q0) and the full centered image for ref
    return Q_tuple[0], centered_im
