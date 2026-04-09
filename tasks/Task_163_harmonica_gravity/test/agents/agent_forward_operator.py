import numpy as np

import matplotlib

matplotlib.use("Agg")

import harmonica as hm

def forward_operator(
    coordinates: tuple,
    prisms: list,
    densities: np.ndarray,
    convert_to_mgal: bool = True
) -> np.ndarray:
    """
    Forward operator: compute gravity anomaly from subsurface prism model.
    
    This function computes the vertical component of the gravitational
    acceleration (g_z) at observation points due to rectangular prisms
    with specified density contrasts.
    
    Parameters
    ----------
    coordinates : tuple
        Tuple of (easting, northing, height) arrays for observation points
    prisms : list
        List of prism bounds [west, east, south, north, bottom, top] in meters
    densities : np.ndarray
        Density contrasts for each prism in kg/m³
    convert_to_mgal : bool
        If True, convert from SI (m/s²) to mGal
    
    Returns
    -------
    np.ndarray
        Computed gravity anomaly at observation points
    """
    # Compute gravity using harmonica's prism forward model
    gravity = hm.prism_gravity(
        coordinates=coordinates,
        prisms=prisms,
        density=densities,
        field="g_z",
    )
    
    if convert_to_mgal:
        gravity_max = np.max(np.abs(gravity))
        if gravity_max < 1e-2:
            gravity = gravity * 1e5  # convert m/s² → mGal
    
    return gravity
