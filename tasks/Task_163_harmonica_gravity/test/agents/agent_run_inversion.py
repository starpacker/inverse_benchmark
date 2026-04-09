import numpy as np

import matplotlib

matplotlib.use("Agg")

import harmonica as hm

import verde as vd

def run_inversion(
    data: dict,
    depth: float = 5000.0,
    damping: float = 1e-3
) -> dict:
    """
    Run gravity field inversion using Equivalent Sources method.
    
    This function fits an EquivalentSources model to the noisy gravity
    observations and predicts the reconstructed gravity field.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_and_preprocess_data containing:
        - 'coordinates': observation coordinates
        - 'gravity_noisy': noisy observations
        - 'observation_height': height of observations
        - 'shape': grid shape
        - 'region': region bounds
    depth : float
        Depth of equivalent sources below observation points (meters)
    damping : float
        Tikhonov regularization parameter
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'gravity_reconstructed': reconstructed gravity field (2D array)
        - 'eqs_model': fitted EquivalentSources model
        - 'predicted_upward': upward continued gravity field
        - 'upward_height': height of upward continuation
    """
    print("[INFO] Fitting EquivalentSources model...")
    
    coordinates = data['coordinates']
    gravity_noisy = data['gravity_noisy']
    observation_height = data['observation_height']
    shape = data['shape']
    region = data['region']
    unit_label = data['unit_label']
    
    # Flatten coordinates for fitting
    easting_flat = coordinates[0].ravel()
    northing_flat = coordinates[1].ravel()
    height_flat = np.full_like(easting_flat, observation_height)
    data_flat = gravity_noisy.ravel()
    
    fit_coords = (easting_flat, northing_flat, height_flat)
    
    # Create and fit EquivalentSources model
    eqs = hm.EquivalentSources(
        depth=depth,
        damping=damping,
    )
    eqs.fit(fit_coords, data_flat)
    print("[INFO] EquivalentSources model fitted successfully")
    
    # Predict at observation locations
    predicted_flat = eqs.predict(fit_coords)
    gravity_reconstructed = predicted_flat.reshape(shape)
    
    print(f"[INFO] Reconstructed gravity range: [{gravity_reconstructed.min():.4f}, {gravity_reconstructed.max():.4f}] {unit_label}")
    
    # Predict on upward-continued surface
    upward_height = 2000.0  # 2 km above surface
    coords_up = vd.grid_coordinates(region, shape=shape, extra_coords=upward_height)
    easting_up = coords_up[0].ravel()
    northing_up = coords_up[1].ravel()
    height_up = np.full_like(easting_up, upward_height)
    predicted_upward = eqs.predict((easting_up, northing_up, height_up)).reshape(shape)
    print(f"[INFO] Upward continued ({upward_height}m) range: [{predicted_upward.min():.4f}, {predicted_upward.max():.4f}] {unit_label}")
    
    return {
        'gravity_reconstructed': gravity_reconstructed,
        'eqs_model': eqs,
        'predicted_upward': predicted_upward,
        'upward_height': upward_height,
    }
