import numpy as np


# --- Extracted Dependencies ---

def _get_slopes_diffractive(wfs, tel, ngs, phase_in=None):
    """
    Helper: Simulates the physical process of the Shack-Hartmann WFS.
    Computes slopes via FFT-based spot formation and Center of Gravity centroiding.
    """
    if phase_in is not None:
        tel.src.phase = phase_in
    
    # Get Electric Field at Lenslet Array
    cube_em = wfs.get_lenslet_em_field(tel.src.phase)
    
    # Form Spots (Intensity = |FFT(E)|^2)
    complex_field = np.fft.fft2(cube_em, axes=[1, 2])
    intensity_spots = np.abs(complex_field) ** 2
    
    # Centroiding (Center of Gravity)
    n_pix = intensity_spots.shape[1]
    x = np.arange(n_pix) - n_pix // 2
    X, Y = np.meshgrid(x, x)
    
    slopes = np.zeros((wfs.nValidSubaperture, 2))
    valid_idx = 0
    
    for i in range(wfs.nSubap ** 2):
        if wfs.valid_subapertures_1D[i]:
            I = intensity_spots[i]
            flux = np.sum(I)
            if flux > 0:
                cx = np.sum(I * X) / flux
                cy = np.sum(I * Y) / flux
                slopes[valid_idx, 0] = cx
                slopes[valid_idx, 1] = cy
                valid_idx += 1
    
    slopes_flat = np.concatenate((slopes[:, 0], slopes[:, 1]))
    return slopes_flat

def run_inversion(system_data, n_iter=20, gain=0.4):
    """
    Run the closed-loop AO inversion/correction using integral control.
    
    The inversion problem: Find DM commands that minimize residual wavefront error.
    Control law: u[k] = u[k-1] - gain * R * s[k]
    
    Returns:
        dict: Contains Strehl history, final DM commands, and final PSF.
    """
    tel = system_data['tel']
    ngs = system_data['ngs']
    atm = system_data['atm']
    dm = system_data['dm']
    wfs = system_data['wfs']
    sci_cam = system_data['sci_cam']
    psf_ref = system_data['psf_ref']
    ref_slopes = system_data['ref_slopes']
    reconstructor = system_data['reconstructor']
    
    strehl_history = []
    dm.coefs[:] = 0  # Initialize DM to flat
    
    for k in range(n_iter):
        # Move Atmosphere
        atm.update()
        
        # Forward Pass: Atmosphere -> Telescope -> DM -> WFS
        atm * ngs * tel * dm
        
        # Measure Slopes (subtract reference for residual)
        slopes_meas = _get_slopes_diffractive(wfs, tel, ngs) - ref_slopes
        
        # Integral Controller: u[k] = u[k-1] - gain * R * s[k]
        delta_command = np.matmul(reconstructor, slopes_meas)
        dm.coefs = dm.coefs - gain * delta_command
        
        # Evaluation (Science Path)
        atm * ngs * tel * dm * sci_cam
        sr = _compute_strehl(sci_cam.frame, psf_ref)
        strehl_history.append(sr)
    
    # Get final PSF
    final_psf = sci_cam.frame.copy()
    final_dm_commands = dm.coefs.copy()
    
    return {
        'strehl_history': np.array(strehl_history),
        'final_dm_commands': final_dm_commands,
        'final_psf': final_psf
    }

def _compute_strehl(psf, psf_ref):
    """
    Helper: Computes Strehl Ratio using OTF (Optical Transfer Function) method.
    Strehl ~ Sum(OTF) / Sum(OTF_perfect)
    """
    otf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    otf_ref = np.abs(np.fft.fftshift(np.fft.fft2(psf_ref)))
    strehl = np.sum(otf) / np.sum(otf_ref)
    return strehl * 100
