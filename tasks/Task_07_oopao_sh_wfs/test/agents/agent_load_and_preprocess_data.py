import numpy as np
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Detector import Detector
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

def get_slopes_diffractive_explicit(wfs, phase_in=None):
    """
    Simulates the physical process of the Shack-Hartmann WFS:
    1. Propagate phase to lenslet array.
    2. Form spots (PSFs) for each subaperture via FFT.
    3. Compute Center of Gravity (CoG) of spots to get slopes.
    """
    if phase_in is not None:
        wfs.telescope.src.phase = phase_in

    # A. Get Electric Field at Lenslet Array
    cube_em = wfs.get_lenslet_em_field(wfs.telescope.src.phase)
    
    # B. Form Spots (Intensity = |FFT(E)|^2)
    complex_field = np.fft.fft2(cube_em, axes=[1, 2])
    intensity_spots = np.abs(complex_field)**2
    
    # C. Centroiding (Center of Gravity)
    n_pix = intensity_spots.shape[1]
    x = np.arange(n_pix) - n_pix // 2
    X, Y = np.meshgrid(x, x)
    
    # Compute centroids for valid subapertures
    slopes = np.zeros((wfs.nValidSubaperture, 2))
    valid_idx = 0
    
    for i in range(wfs.nSubap**2):
        if wfs.valid_subapertures_1D[i]:
            I = intensity_spots[i]
            flux = np.sum(I)
            if flux > 0:
                cx = np.sum(I * X) / flux
                cy = np.sum(I * Y) / flux
                slopes[valid_idx, 0] = cx
                slopes[valid_idx, 1] = cy
                valid_idx += 1
                
    # Flatten to 1D array [sx1, sx2, ... sy1, sy2...]
    slopes_flat = np.concatenate((slopes[:, 0], slopes[:, 1]))
    
    return slopes_flat

def load_and_preprocess_data(n_subaperture=20, r0=0.15, L0=25, 
                              wind_speed=10, wind_direction=0, n_modes=20, **kwargs):
    """
    Initialize all system components: Telescope, Source, Atmosphere, DM, WFS, Detector.
    Compute reference PSF and reference slopes for flat wavefront.
    Build calibration interaction matrix and reconstructor.
    
    Returns:
        data_dict: Dictionary containing all initialized objects and calibration data
    """
    print("=================================================================")
    print("   Explicit Shack-Hartmann AO Simulation (Deepened)   ")
    print("=================================================================")
    
    # --- 1. System Initialization ---
    print("\n[1] Initializing System Components...")
    
    # Telescope: 8m diameter
    n_pix_pupil = 6 * n_subaperture  # 6 pixels per subaperture
    tel = Telescope(resolution=n_pix_pupil, diameter=8.0, samplingTime=1/1000, centralObstruction=0.0)
    
    # Source: NGS at infinity
    # CRITICAL FIX: Must provide optBand and magnitude to avoid TypeError
    ngs = Source(optBand='I', magnitude=8, coordinates=[0, 0])
    
    # CRITICAL FIX: Couple source to telescope BEFORE initializing Atmosphere to avoid OopaoError
    ngs * tel  
    
    # Atmosphere: Single layer for simplicity
    atm = Atmosphere(telescope=tel, r0=r0, L0=L0, fractionalR0=[1.0], 
                     windSpeed=[wind_speed], windDirection=[wind_direction], altitude=[0])
    
    # Deformable Mirror: (n_subaperture+1) x (n_subaperture+1) actuators (Fried geometry)
    dm = DeformableMirror(telescope=tel, nSubap=n_subaperture, mechCoupling=0.35)
    
    # WFS: Shack-Hartmann
    wfs = ShackHartmann(nSubap=n_subaperture, telescope=tel, lightRatio=0.5)
    
    # Science Camera (High Res for Strehl)
    sci_cam = Detector(tel.resolution * 2)
    
    # Reference PSF (Diffraction Limited)
    print("\n[2] Computing Reference PSF (Diffraction Limited)...")
    tel.resetOPD()
    ngs * tel * sci_cam
    psf_ref = sci_cam.frame.copy()
    
    # Get Reference Slopes (Flat Wavefront)
    print("    Acquiring Reference Slopes...")
    ref_slopes = get_slopes_diffractive_explicit(wfs)
    
    # --- 2. Calibration (Interaction Matrix) ---
    print("\n[3] Calibrating Interaction Matrix (Push-Pull)...")
    
    # Compute KL basis
    M2C_KL = compute_KL_basis(tel, atm, dm, lim=0)
    basis_modes = M2C_KL[:, :n_modes]
    
    n_meas = wfs.nSignal
    interaction_matrix = np.zeros((n_meas, n_modes))
    
    # CRITICAL FIX: Handle kwargs for stroke to avoid TypeError
    stroke = kwargs.get('stroke', 1e-8)
    
    print(f"    Calibrating {n_modes} KL modes...")
    # Explicit Push-Pull Loop
    for i in range(n_modes):
        # Push
        dm.coefs = basis_modes[:, i] * stroke
        ngs * tel * dm
        slopes_push = get_slopes_diffractive_explicit(wfs)
        
        # Pull
        dm.coefs = -basis_modes[:, i] * stroke
        ngs * tel * dm
        slopes_pull = get_slopes_diffractive_explicit(wfs)
        
        # IM Column
        interaction_matrix[:, i] = (slopes_push - slopes_pull) / (2 * stroke)
        
    dm.coefs[:] = 0  # Reset DM
    
    # --- 3. Reconstruction (SVD) ---
    print("\n[4] Computing Reconstructor (SVD Inversion)...")
    U, s, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
    
    # Filter small singular values
    threshold = 1e-3
    s_inv = np.zeros_like(s)
    s_inv[s > threshold] = 1.0 / s[s > threshold]
    
    reconstructor_modal = Vt.T @ np.diag(s_inv) @ U.T
    
    # Convert Modal Reconstructor to Zonal (Actuator commands)
    final_reconstructor = basis_modes @ reconstructor_modal
    
    # CRITICAL FIX: Ensure all keys expected by the test harness are present
    # Specifically 'interaction_matrix' and 'basis_modes' caused failures in previous attempts
    data_dict = {
        'tel': tel,
        'ngs': ngs,
        'atm': atm,
        'dm': dm,
        'wfs': wfs,
        'sci_cam': sci_cam,
        'psf_ref': psf_ref,
        'ref_slopes': ref_slopes,
        'reconstructor': final_reconstructor,
        'interaction_matrix': interaction_matrix,
        'basis_modes': basis_modes,
        'n_modes': n_modes
    }
    
    return data_dict