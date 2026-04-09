import numpy as np

import matplotlib

matplotlib.use('Agg')

def run_inversion(frame_a, frame_b, window_size, overlap, search_area_size, dt):
    """
    PIV reconstruction: recover velocity field from particle image pair.
    
    Uses OpenPIV's extended search area PIV with:
    1. Cross-correlation in interrogation windows
    2. Sub-pixel peak finding (Gaussian fit)
    3. Signal-to-noise ratio filtering
    4. Outlier replacement via local mean interpolation
    
    Args:
        frame_a: First frame (2D array)
        frame_b: Second frame (2D array)
        window_size: Interrogation window size
        overlap: Overlap between windows
        search_area_size: Search area size
        dt: Time between frames
    
    Returns:
        dict containing:
            - u_recon: reconstructed u velocity field
            - v_recon: reconstructed v velocity field
            - x_grid: x coordinates of PIV grid
            - y_grid: y coordinates of PIV grid
            - sig2noise: signal-to-noise ratio map
            - flags: outlier flags
    """
    from openpiv import pyprocess, validation, filters
    
    # Step 1: Cross-correlation PIV
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        correlation_method='circular',
        subpixel_method='gaussian',
        sig2noise_method='peak2peak'
    )
    
    # Step 2: Get coordinates
    x_grid, y_grid = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=search_area_size,
        overlap=overlap
    )
    
    # Step 3: Validation - signal-to-noise ratio filter
    flags_s2n = validation.sig2noise_val(sig2noise, threshold=1.05)
    
    # Step 4: Validation - global velocity range filter
    u_max = np.max(np.abs(u)) * 1.5 + 1
    flags_g = validation.global_val(u, v, (-u_max, u_max), (-u_max, u_max))
    
    # Combine flags
    flags = flags_s2n | flags_g
    
    # Step 5: Replace outliers using local mean interpolation
    u_filtered, v_filtered = filters.replace_outliers(
        u, v, flags, 
        method='localmean', 
        max_iter=10, 
        kernel_size=2
    )
    
    # Convert MaskedArray to regular ndarray
    if hasattr(u_filtered, 'data'):
        u_filtered = np.array(u_filtered)
    if hasattr(v_filtered, 'data'):
        v_filtered = np.array(v_filtered)
    
    print(f"  [PIV] Grid shape: {u_filtered.shape}")
    print(f"  [PIV] Outliers replaced: {np.sum(flags)} / {flags.size} "
          f"({100*np.sum(flags)/flags.size:.1f}%)")
    print(f"  [PIV] Velocity range: u=[{u_filtered.min():.2f}, {u_filtered.max():.2f}], "
          f"v=[{v_filtered.min():.2f}, {v_filtered.max():.2f}]")
    
    return {
        'u_recon': u_filtered,
        'v_recon': v_filtered,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'sig2noise': sig2noise,
        'flags': flags
    }
