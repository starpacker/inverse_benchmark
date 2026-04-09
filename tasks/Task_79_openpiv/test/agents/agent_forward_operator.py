import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_gaussian_particle(img, x0, y0, diameter, intensity=255.0):
    """Render a single Gaussian particle onto the image."""
    sigma = diameter / 4.0
    r = int(3 * sigma) + 1
    y_min = max(0, int(y0) - r)
    y_max = min(img.shape[0], int(y0) + r + 1)
    x_min = max(0, int(x0) - r)
    x_max = min(img.shape[1], int(x0) + r + 1)
    
    for iy in range(y_min, y_max):
        for ix in range(x_min, x_max):
            dist2 = (ix - x0)**2 + (iy - y0)**2
            img[iy, ix] += intensity * np.exp(-dist2 / (2 * sigma**2))
    return img

def forward_operator(velocity_field_u, velocity_field_v, 
                     particle_positions_x, particle_positions_y, 
                     img_size, particle_diameter, dt):
    """
    Forward model: Given velocity field, displace particles to create frame_b.
    y = A(x): velocity field -> displaced particle image
    
    In PIV, the forward model is:
        x'_i = x_i + u(x_i, y_i) * dt
        y'_i = y_i + v(x_i, y_i) * dt
    where (x_i, y_i) are particle positions and u, v is the velocity field.
    
    Args:
        velocity_field_u: u component of velocity field (2D array on grid)
        velocity_field_v: v component of velocity field (2D array on grid)
        particle_positions_x: x positions of particles
        particle_positions_y: y positions of particles
        img_size: Image size in pixels
        particle_diameter: Particle diameter in pixels
        dt: Time step
    
    Returns:
        displaced_image: The resulting particle image after displacement
    """
    # Interpolate velocity field at particle positions
    from scipy.interpolate import RegularGridInterpolator
    
    # Create grid for interpolation
    ny, nx = velocity_field_u.shape
    y_coords = np.linspace(0, img_size, ny)
    x_coords = np.linspace(0, img_size, nx)
    
    interp_u = RegularGridInterpolator(
        (y_coords, x_coords), velocity_field_u, 
        method='linear', bounds_error=False, fill_value=0
    )
    interp_v = RegularGridInterpolator(
        (y_coords, x_coords), velocity_field_v, 
        method='linear', bounds_error=False, fill_value=0
    )
    
    # Get velocity at particle positions
    points = np.column_stack([particle_positions_y, particle_positions_x])
    u_at_particles = interp_u(points)
    v_at_particles = interp_v(points)
    
    # Compute displaced positions
    x_displaced = particle_positions_x + u_at_particles * dt
    y_displaced = particle_positions_y + v_at_particles * dt
    
    # Render displaced image
    n_particles = len(particle_positions_x)
    intensities = np.random.uniform(180, 255, n_particles)
    
    displaced_image = np.zeros((img_size, img_size), dtype=np.float64)
    for i in range(n_particles):
        if (5 < x_displaced[i] < img_size - 5 and 
            5 < y_displaced[i] < img_size - 5):
            displaced_image = generate_gaussian_particle(
                displaced_image, x_displaced[i], y_displaced[i],
                particle_diameter, intensities[i]
            )
    
    displaced_image = np.clip(displaced_image, 0, 255).astype(np.int32)
    
    return displaced_image
