import numpy as np

import matplotlib

matplotlib.use('Agg')

def create_spiral_array(n_mics, radius):
    """Archimedean spiral microphone array in z=0 plane."""
    angles = np.linspace(0, 4 * np.pi, n_mics, endpoint=False)
    radii = np.linspace(0.05, radius, n_mics)
    return np.column_stack([radii * np.cos(angles),
                            radii * np.sin(angles),
                            np.zeros(n_mics)])

def create_focus_grid(grid_span, grid_res, z_focus):
    """2D focus grid at distance z_focus."""
    coords = np.linspace(-grid_span / 2, grid_span / 2, grid_res)
    gx, gy = np.meshgrid(coords, coords)
    grid_points = np.column_stack([gx.ravel(), gy.ravel(),
                                   np.full(grid_res**2, z_focus)])
    return grid_points, coords

def create_source_distribution(grid_points, grid_res):
    """3 Gaussian blob sources."""
    sources = [
        {'x': -0.12, 'y': 0.15,  'strength': 1.0},
        {'x': 0.18,  'y': -0.08, 'strength': 0.7},
        {'x': 0.0,   'y': -0.20, 'strength': 0.5},
    ]
    q = np.zeros(grid_res * grid_res)
    sigma = 0.04
    for s in sources:
        r2 = (grid_points[:, 0] - s['x'])**2 + (grid_points[:, 1] - s['y'])**2
        q += s['strength'] * np.exp(-r2 / (2 * sigma**2))
    return q

def load_and_preprocess_data(n_mics, array_radius, grid_span, grid_res, z_focus, seed=42):
    """
    Load and preprocess data for acoustic beamforming.
    
    Creates:
    - Spiral microphone array
    - Focus grid
    - Ground truth source distribution
    
    Parameters
    ----------
    n_mics : int
        Number of microphones in the array
    array_radius : float
        Radius of the spiral array [m]
    grid_span : float
        Width/height of the focus grid [m]
    grid_res : int
        Resolution of the focus grid (grid_res x grid_res points)
    z_focus : float
        Distance from array to focus plane [m]
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - mic_positions: (n_mics, 3) array of microphone positions
        - grid_points: (n_grid, 3) array of focus grid points
        - coords: 1D array of grid coordinates
        - q_gt: (n_grid,) ground truth source distribution
        - grid_res: grid resolution
    """
    np.random.seed(seed)
    
    # Create spiral microphone array
    mic_positions = create_spiral_array(n_mics, array_radius)
    
    # Create focus grid
    grid_points, coords = create_focus_grid(grid_span, grid_res, z_focus)
    
    # Create ground truth source distribution
    q_gt = create_source_distribution(grid_points, grid_res)
    
    return {
        'mic_positions': mic_positions,
        'grid_points': grid_points,
        'coords': coords,
        'q_gt': q_gt,
        'grid_res': grid_res
    }
