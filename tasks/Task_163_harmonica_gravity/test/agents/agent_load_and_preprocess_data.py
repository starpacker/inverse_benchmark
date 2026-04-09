import numpy as np

import matplotlib

matplotlib.use("Agg")

import harmonica as hm

import verde as vd

def load_and_preprocess_data(
    prisms: list,
    densities: np.ndarray,
    region: tuple,
    shape: tuple,
    observation_height: float,
    noise_level: float,
    seed: int
) -> dict:
    """
    Load and preprocess data for gravity field inversion.
    
    This function:
    1. Sets up the observation grid coordinates
    2. Computes the true gravity anomaly from the prism model (forward problem)
    3. Converts units to mGal if necessary
    4. Adds Gaussian noise to simulate noisy observations
    
    Parameters
    ----------
    prisms : list
        List of prism bounds [west, east, south, north, bottom, top] in meters
    densities : np.ndarray
        Density contrasts for each prism in kg/m³
    region : tuple
        Region bounds (west, east, south, north) in meters
    shape : tuple
        Grid shape (n_north, n_east)
    observation_height : float
        Height of observations in meters
    noise_level : float
        Standard deviation of Gaussian noise in mGal
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'coordinates': tuple of (easting, northing) 2D arrays
        - 'gravity_true_mgal': true gravity anomaly in mGal (2D array)
        - 'gravity_noisy': noisy observations in mGal (2D array)
        - 'observation_height': height of observations
        - 'region': region bounds
        - 'shape': grid shape
        - 'unit_label': string label for units
        - 'noise_level': noise level used
        - 'prisms': prism definitions
        - 'densities': density contrasts
    """
    np.random.seed(seed)
    
    # Create observation grid
    coordinates = vd.grid_coordinates(
        region, shape=shape, extra_coords=observation_height
    )
    
    print(f"[INFO] Observation grid: {shape[0]}×{shape[1]} points, region={region}")
    print(f"[INFO] Observation height: {observation_height} m")
    print(f"[INFO] Defined {len(prisms)} subsurface prisms")
    for i, (p, d) in enumerate(zip(prisms, densities)):
        print(f"  Prism {i}: bounds={p}, density_contrast={d} kg/m³")
    
    # Compute true gravity using prism forward model
    gravity_true = hm.prism_gravity(
        coordinates=coordinates,
        prisms=prisms,
        density=densities,
        field="g_z",
    )
    
    gravity_max = np.max(np.abs(gravity_true))
    print(f"[INFO] True gravity range: [{gravity_true.min():.6e}, {gravity_true.max():.6e}]")
    
    # Convert to mGal if values are in SI units (m/s²)
    if gravity_max < 1e-2:
        gravity_true_mgal = gravity_true * 1e5  # convert m/s² → mGal
        unit_label = "mGal"
        print(f"[INFO] Converted to mGal. Range: [{gravity_true_mgal.min():.4f}, {gravity_true_mgal.max():.4f}] mGal")
    else:
        gravity_true_mgal = gravity_true
        unit_label = "mGal" if gravity_max < 100 else "m/s²"
        print(f"[INFO] Values in native units. Range: [{gravity_true_mgal.min():.4f}, {gravity_true_mgal.max():.4f}] {unit_label}")
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, gravity_true_mgal.shape)
    gravity_noisy = gravity_true_mgal + noise
    print(f"[INFO] Added Gaussian noise: σ = {noise_level} {unit_label}")
    print(f"[INFO] Noisy gravity range: [{gravity_noisy.min():.4f}, {gravity_noisy.max():.4f}] {unit_label}")
    
    return {
        'coordinates': coordinates,
        'gravity_true_mgal': gravity_true_mgal,
        'gravity_noisy': gravity_noisy,
        'observation_height': observation_height,
        'region': region,
        'shape': shape,
        'unit_label': unit_label,
        'noise_level': noise_level,
        'prisms': prisms,
        'densities': densities,
    }
