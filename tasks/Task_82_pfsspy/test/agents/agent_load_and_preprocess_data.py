import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def load_and_preprocess_data(nphi, ntheta, noise_level):
    """
    Generate a synthetic photospheric magnetogram with dipole + quadrupole
    components, simulating a simplified solar magnetic field.
    
    Args:
        nphi: Number of longitude points
        ntheta: Number of latitude points (sin(lat) grid)
        noise_level: Noise level for magnetogram (fraction of max)
    
    Returns:
        br_map: sunpy Map of radial magnetic field at photosphere
        br_clean: clean magnetogram (without noise)
        br_noisy: noisy magnetogram
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.map
    
    # Create coordinate grid
    # For CEA projection, y-axis is proportional to sin(latitude)
    phi = np.linspace(0, 2 * np.pi, nphi + 1)[:-1]  # longitude
    sin_lat = np.linspace(-1, 1, ntheta)  # sin(latitude) for CEA
    theta = np.arcsin(sin_lat)  # latitude in radians
    
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Synthesize magnetogram using spherical harmonics
    # Dipole component (l=1, m=0): B_r ~ cos(θ) (tilted slightly)
    dipole_strength = 5.0  # Gauss
    br_dipole = dipole_strength * np.sin(theta_grid)  # axial dipole
    
    # Tilted dipole component (l=1, m=1)
    tilt_strength = 2.0
    br_tilt = tilt_strength * np.cos(theta_grid) * np.cos(phi_grid - 0.5)
    
    # Quadrupole component (l=2, m=0)
    quad_strength = 1.5
    br_quad = quad_strength * (3 * np.sin(theta_grid)**2 - 1) / 2
    
    # Active region spots (localized bipolar regions)
    # Spot 1: positive polarity
    lat1, lon1 = np.deg2rad(20), np.deg2rad(60)
    sigma_spot = np.deg2rad(10)
    dist1 = np.sqrt((theta_grid - lat1)**2 + (phi_grid - lon1)**2)
    br_spot1 = 15.0 * np.exp(-dist1**2 / (2 * sigma_spot**2))
    
    # Spot 2: negative polarity (nearby)
    lat2, lon2 = np.deg2rad(25), np.deg2rad(75)
    dist2 = np.sqrt((theta_grid - lat2)**2 + (phi_grid - lon2)**2)
    br_spot2 = -12.0 * np.exp(-dist2**2 / (2 * sigma_spot**2))
    
    # Another active region in southern hemisphere
    lat3, lon3 = np.deg2rad(-15), np.deg2rad(200)
    dist3 = np.sqrt((theta_grid - lat3)**2 + (phi_grid - lon3)**2)
    br_spot3 = -10.0 * np.exp(-dist3**2 / (2 * sigma_spot**2))
    
    lat4, lon4 = np.deg2rad(-10), np.deg2rad(220)
    dist4 = np.sqrt((theta_grid - lat4)**2 + (phi_grid - lon4)**2)
    br_spot4 = 8.0 * np.exp(-dist4**2 / (2 * sigma_spot**2))
    
    # Combined clean magnetogram
    br_clean = br_dipole + br_tilt + br_quad + br_spot1 + br_spot2 + br_spot3 + br_spot4
    
    # Add noise
    noise = noise_level * br_clean.max() * np.random.randn(*br_clean.shape)
    br_noisy = br_clean + noise
    
    # Create a SunPy Map with CEA projection (required by pfsspy)
    # CEA: Cylindrical Equal Area
    # pfsspy validation: shape[1]*CDELT1 ≈ 360°, shape[0]*CDELT2*π/2 ≈ 180°
    # So CDELT2 = 360 / (NTHETA * π)
    cdelt1 = 360.0 / nphi
    cdelt2 = 360.0 / (ntheta * np.pi)
    header = {
        'NAXIS1': nphi,
        'NAXIS2': ntheta,
        'CDELT1': cdelt1,
        'CDELT2': cdelt2,
        'CRPIX1': (nphi + 1) / 2.0,
        'CRPIX2': (ntheta + 1) / 2.0,
        'CRVAL1': 0.0,
        'CRVAL2': 0.0,
        'CTYPE1': 'CRLN-CEA',
        'CTYPE2': 'CRLT-CEA',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'DATE-OBS': '2024-01-01T00:00:00',
        'BUNIT': 'G',
    }
    br_map = sunpy.map.Map(br_noisy, header)
    
    return br_map, br_clean, br_noisy
