import numpy as np

import matplotlib

matplotlib.use('Agg')

from ceviche import fdfd_ez

from scipy.ndimage import gaussian_filter

def load_and_preprocess_data(nx, ny, npml, wg_width, eps_min, eps_max, omega, dl):
    """
    Create ground truth dielectric structure, source, design mask, and solve forward problem
    to obtain target field.
    
    Returns:
        dict containing:
            - eps_gt: ground truth permittivity
            - source: electromagnetic source
            - design_mask: mask for designable region
            - Ez_gt: ground truth electric field
            - params: dictionary of physical parameters
    """
    # Create ground truth structure: waveguide with scatterers
    eps_gt = np.ones((nx, ny)) * eps_min
    
    cx = nx // 2
    cy = ny // 2
    hw = wg_width // 2
    
    # Straight waveguide from left to right (through center)
    eps_gt[cx - hw : cx + hw + 1, npml : ny - npml] = eps_max
    
    # Add two rectangular scatterers/defects along the waveguide
    d1_y = cy - 10
    eps_gt[cx - hw - 3 : cx + hw + 4, d1_y - 2 : d1_y + 3] = eps_max
    
    d2_y = cy + 10
    eps_gt[cx - hw - 3 : cx + hw + 4, d2_y - 2 : d2_y + 3] = eps_max
    
    # Add a small side-coupled stub
    stub_len = 6
    eps_gt[cx + hw + 1 : cx + hw + 1 + stub_len, cy - 1 : cy + 2] = eps_max
    
    # Light smoothing
    eps_gt = gaussian_filter(eps_gt, sigma=0.5)
    eps_gt = np.clip(eps_gt, eps_min, eps_max)
    
    # Create line source at left side of waveguide
    source = np.zeros((nx, ny), dtype=complex)
    src_y = npml + 2
    source[cx - hw : cx + hw + 1, src_y] = 1.0
    
    # Create design mask: entire non-PML region is designable
    design_mask = np.zeros((nx, ny))
    margin = npml + 2
    design_mask[margin : nx - margin, margin : ny - margin] = 1.0
    
    # Solve forward problem for ground truth
    F = fdfd_ez(omega, dl, eps_gt, [npml, npml])
    _, _, Ez_gt = F.solve(source)
    Ez_gt = np.array(Ez_gt)
    
    params = {
        'nx': nx,
        'ny': ny,
        'npml': npml,
        'wg_width': wg_width,
        'eps_min': eps_min,
        'eps_max': eps_max,
        'omega': omega,
        'dl': dl
    }
    
    return {
        'eps_gt': eps_gt,
        'source': source,
        'design_mask': design_mask,
        'Ez_gt': Ez_gt,
        'params': params
    }
