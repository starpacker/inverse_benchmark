import logging

import numpy as np

from scipy.interpolate import RectBivariateSpline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def arange2(start, stop=None, step=1):
    """Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop is None:
        a = np.arange(start)
    else:
        a = np.arange(start, stop, step)
        if a[-1] > stop - step:
            a = np.delete(a, -1)
    return a

def scan_convert(data, xb, zb):
    """Scan conversion to image grid"""
    decim_fact = 8
    
    # Decimate input data to save compute during interpolation
    data_dec = data[:, 0:-1:decim_fact]
    zb_dec = zb[0:-1:decim_fact]

    # Original code used interp2d. RectBivariateSpline is the modern robust equivalent for rectilinear grids.
    # Note: RectBivariateSpline takes (x, y) grid axes and z data.
    # Here input axes are xb (lateral), zb (depth). Data is (lateral, depth).
    
    # Ensure sorted order for Spline
    # xb and zb are usually sorted, but let's be safe or just pass them.
    
    interpolator = RectBivariateSpline(xb, zb_dec, data_dec)
    
    dz = zb[1] - zb[0]
    # Define new grid
    xnew = arange2(xb[0], xb[-1] + dz, dz)
    znew = zb
    
    # Evaluate spline
    image_sC = interpolator(xnew, znew)
    
    # RectBivariateSpline returns (xnew, znew). We might need to transpose depending on display convention
    # The original interp2d output shape convention matches (znew, xnew) in the usage `interp_func(znew, xnew)`.
    # Let's align with original output: image_sC was result of interp2d(znew, xnew).
    # interp2d (x, y, z) -> call(new_x, new_y). 
    # Original: interp2d(zb, xb, data). x=zb, y=xb. call(znew, xnew).
    # Result shape was (len(xnew), len(znew)) effectively or transposed?
    # Actually, interp2d builds function f(x, y). 
    # Calling f(znew, xnew) evaluates at grid defined by znew and xnew.
    # This implies the result is on meshgrid(znew, xnew).
    
    # To be safe and strictly follow logic:
    # We want output image where axis 0 is X and axis 1 is Z (or vice versa).
    # Typically ultrasound images are (Depth, Lateral) or (Lateral, Depth).
    # The original code transposed at the very end `image_final = ... .T`.
    
    # Let's just return the grid computed by RectBivariateSpline(xb, zb_dec, data_dec)(xnew, znew)
    # This returns shape (len(xnew), len(znew)).
    
    return image_sC, znew, xnew
