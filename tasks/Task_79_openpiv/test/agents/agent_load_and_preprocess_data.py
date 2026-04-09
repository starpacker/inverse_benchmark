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

def load_and_preprocess_data(img_size, n_particles, particle_diameter, 
                              noise_level, window_size, overlap, 
                              search_area_size, dt, seed=42):
    """
    Generate synthetic particle image pair and ground truth velocity field.
    
    Args:
        img_size: Image size in pixels
        n_particles: Number of tracer particles
        particle_diameter: Particle diameter in pixels
        noise_level: Image noise level (fraction of max intensity)
        window_size: Interrogation window size
        overlap: Overlap between windows
        search_area_size: Search area size
        dt: Time between frames
        seed: Random seed
    
    Returns:
        dict containing:
            - frame_a: first frame (2D array)
            - frame_b: second frame (particles displaced by velocity field)
            - gt_u: ground truth u velocity on PIV grid
            - gt_v: ground truth v velocity on PIV grid
            - x_grid: x coordinates of PIV grid
            - y_grid: y coordinates of PIV grid
            - params: dict of parameters
    """
    np.random.seed(seed)
    
    # Generate random particle positions for frame_a
    x_particles = np.random.uniform(10, img_size - 10, n_particles)
    y_particles = np.random.uniform(10, img_size - 10, n_particles)
    
    # Compute velocity at each particle position
    u_at_particles, v_at_particles = generate_velocity_field_on_grid(
        x_particles, y_particles, img_size
    )
    
    # Displaced positions for frame_b
    x_displaced = x_particles + u_at_particles * dt
    y_displaced = y_particles + v_at_particles * dt
    
    # Random intensities for particles
    intensities = np.random.uniform(180, 255, n_particles)
    
    # Render frame_a
    frame_a = np.zeros((img_size, img_size), dtype=np.float64)
    for i in range(n_particles):
        frame_a = generate_gaussian_particle(
            frame_a, x_particles[i], y_particles[i], 
            particle_diameter, intensities[i]
        )
    
    # Render frame_b (particles displaced by velocity field)
    frame_b = np.zeros((img_size, img_size), dtype=np.float64)
    for i in range(n_particles):
        if (5 < x_displaced[i] < img_size - 5 and 
            5 < y_displaced[i] < img_size - 5):
            frame_b = generate_gaussian_particle(
                frame_b, x_displaced[i], y_displaced[i],
                particle_diameter, intensities[i]
            )
    
    # Clip to valid range and add noise
    frame_a = np.clip(frame_a, 0, 255)
    frame_b = np.clip(frame_b, 0, 255)
    
    # Add Gaussian noise
    noise_a = np.random.normal(0, noise_level * 255, frame_a.shape)
    noise_b = np.random.normal(0, noise_level * 255, frame_b.shape)
    frame_a = np.clip(frame_a + noise_a, 0, 255).astype(np.int32)
    frame_b = np.clip(frame_b + noise_b, 0, 255).astype(np.int32)
    
    # Compute ground truth on PIV grid
    from openpiv.pyprocess import get_coordinates
    x_grid, y_grid = get_coordinates(
        image_size=frame_a.shape, 
        search_area_size=search_area_size, 
        overlap=overlap
    )
    gt_u, gt_v = generate_velocity_field_on_grid(x_grid, y_grid, img_size)
    
    params = {
        'img_size': img_size,
        'n_particles': n_particles,
        'particle_diameter': particle_diameter,
        'noise_level': noise_level,
        'window_size': window_size,
        'overlap': overlap,
        'search_area_size': search_area_size,
        'dt': dt
    }
    
    return {
        'frame_a': frame_a,
        'frame_b': frame_b,
        'gt_u': gt_u,
        'gt_v': gt_v,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'params': params
    }
