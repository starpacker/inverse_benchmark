import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_velocity_field_on_grid(grid_x, grid_y, img_size):
    """
    Generate ground truth velocity field on a given grid.
    Uses a combination of:
    - Lamb-Oseen vortex (rotational flow)
    - Uniform translation
    - Strain field
    """
    cx, cy = img_size / 2.0, img_size / 2.0
    
    vortex_strength = 5.0
    vortex_radius = img_size / 6.0
    
    dx = grid_x - cx
    dy = grid_y - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-10
    
    v_theta = vortex_strength * (1 - np.exp(-r**2 / (2 * vortex_radius**2))) / r
    u_vortex = -v_theta * dy
    v_vortex = v_theta * dx
    
    u_uniform = 2.0
    v_uniform = 1.0
    
    strain_rate = 0.005
    u_strain = strain_rate * dx
    v_strain = -strain_rate * dy
    
    u_total = u_vortex + u_uniform + u_strain
    v_total = v_vortex + v_uniform + v_strain
    
    return u_total, v_total
