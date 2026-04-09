import numpy as np
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Zernike import Zernike

def zernike_mode_explicit(n, m, X, Y, D):
    """
    Generates a Zernike mode Z_n^m on the grid (X, Y)
    """
    # Normalized coordinates
    R = np.sqrt(X**2 + Y**2) / (D / 2)
    Theta = np.arctan2(Y, X)
    
    # Mask outside pupil
    mask = R <= 1.0
    
    # Initialize Z
    Z = np.zeros_like(X)
    
    # Calculate R_nm only inside pupil
    R_vals = R[mask]
    Theta_vals = Theta[mask]
    
    # Radial function values
    Rad = np.zeros_like(R_vals)
    if (n - m) % 2 == 0:
        for k in range((n - m) // 2 + 1):
            if (n - k) < 0 or ((n + m) // 2 - k) < 0 or ((n - m) // 2 - k) < 0:
                continue
            num = ((-1)**k) * np.math.factorial(n - k)
            denom = (np.math.factorial(k) * 
                     np.math.factorial((n + m) // 2 - k) * 
                     np.math.factorial((n - m) // 2 - k))
            Rad += (num / denom) * (R_vals**(n - 2 * k))
            
    # Azimuthal part
    if m == 0:
        Z[mask] = np.sqrt(n + 1) * Rad
    elif m > 0:
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.cos(m * Theta_vals)
    else:  # m < 0
        Z[mask] = np.sqrt(2 * (n + 1)) * Rad * np.sin(-m * Theta_vals)
        
    return Z

def load_and_preprocess_data(resolution, diameter, sampling_time, central_obstruction,
                             opt_band, magnitude, n_zernike_modes, n_iterations,
                             r0, L0, wind_speed, wind_direction, altitude):
    """
    Initialize all simulation components: telescope, source, atmosphere, and Zernike basis.
    
    Returns:
        data_dict: Dictionary containing all initialized objects and precomputed data
    """
    print("\n[1] Initializing System...")
    
    # Initialize telescope
    tel = Telescope(resolution=resolution, diameter=diameter, 
                    samplingTime=sampling_time, centralObstruction=central_obstruction)
    
    # Initialize source
    ngs = Source(optBand=opt_band, magnitude=magnitude)
    ngs * tel
    
    print("\n[2] Generating Zernike Basis Explicitly...")
    
    # Create coordinate grid
    y, x = np.indices((tel.resolution, tel.resolution))
    y = (y - tel.resolution / 2) * tel.pixelSize
    x = (x - tel.resolution / 2) * tel.pixelSize
    
    # Generate first few modes explicitly for demonstration
    n_explicit_modes = 6
    zernike_basis_2d = np.zeros((n_explicit_modes, tel.resolution, tel.resolution))
    
    # Mapping Noll Index (j) to (n, m)
    noll_indices = [
        (0, 0),   # Piston (j=1)
        (1, 1),   # Tilt X (j=2)
        (1, -1),  # Tilt Y (j=3)
        (2, 0),   # Defocus (j=4)
        (2, -2),  # Astigmatism (j=5)
        (2, 2),   # Astigmatism (j=6)
    ]
    
    print("    Generating modes using explicit radial polynomials...")
    for j, (n, m) in enumerate(noll_indices):
        mode = zernike_mode_explicit(n, m, x, y, tel.D)
        zernike_basis_2d[j] = mode
        
    # Use OOPAO for the full set to ensure coverage for decomposition
    Z = Zernike(telObject=tel, J=n_zernike_modes)
    Z.computeZernike(tel)
    Z_inv = np.linalg.pinv(Z.modes)  # Pseudoinverse of the basis
    
    # Initialize atmosphere
    print("\n[3] Initializing Atmosphere...")
    atm = Atmosphere(telescope=tel, r0=r0, L0=L0, 
                     fractionalR0=[1], windSpeed=[wind_speed], 
                     windDirection=[wind_direction], altitude=[altitude])
    atm.initializeAtmosphere(tel)
    
    # Create phase map for forward model demonstration
    # 0.5 rad of Defocus (index 3) + 0.5 rad of Astigmatism (index 5)
    phase_map = 0.5 * zernike_basis_2d[3] + 0.5 * zernike_basis_2d[5]
    
    # Convert to OPD [m] for consistency
    # Phase = 2*pi*OPD / lambda => OPD = Phase * lambda / (2*pi)
    opd_map = phase_map * ngs.wavelength / (2 * np.pi)
    
    data_dict = {
        'telescope': tel,
        'source': ngs,
        'atmosphere': atm,
        'zernike_object': Z,
        'zernike_inverse': Z_inv,
        'zernike_basis_2d': zernike_basis_2d,
        'phase_map': phase_map,
        'opd_map': opd_map,
        'coordinate_x': x,
        'coordinate_y': y,
        'n_iterations': n_iterations,
    }
    
    return data_dict