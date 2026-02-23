# main.py
```python
import sys
import os
import time

# Ensure OOPAO is in path
sys.path.append('/home/yjh/OOPAO')

import numpy as np
import matplotlib.pyplot as plt
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Detector import Detector

# =============================================================================
# 1. HELPER FUNCTIONS (Explicit Implementation of Algorithms)
# =============================================================================

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
    # E = A * exp(i * phi)
    # The OOPAO helper does this splitting for us
    cube_em = wfs.get_lenslet_em_field(wfs.telescope.src.phase)
    
    # B. Form Spots (Intensity = |FFT(E)|^2)
    # Perform FFT on each subaperture (vectorized)
    # This simulates the physical formation of spots on the detector
    complex_field = np.fft.fft2(cube_em, axes=[1, 2])
    intensity_spots = np.abs(complex_field)**2
    
    # C. Centroiding (Center of Gravity)
    # CoG_x = Sum(I * x) / Sum(I)
    # CoG_y = Sum(I * y) / Sum(I)
    
    # Create coordinate grids for the subaperture pixels
    n_pix = intensity_spots.shape[1]
    x = np.arange(n_pix) - n_pix//2
    X, Y = np.meshgrid(x, x)
    
    # Compute centroids for valid subapertures
    slopes = np.zeros((wfs.nValidSubaperture, 2))
    valid_idx = 0
    
    # Iterate over all subapertures (simplified for clarity, could be vectorized)
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
    # Normalize by pixel scale (units of pixels -> arcsec or radians)
    slopes_flat = np.concatenate((slopes[:, 0], slopes[:, 1]))
    
    # In practice, we subtract the reference (flat wavefront) slopes
    # For this demo, we assume ref is 0 or handled externally
    
    return slopes_flat

def compute_strehl_explicit(psf, psf_ref):
    """
    Computes Strehl Ratio explicitly using OTF (Optical Transfer Function) method.
    Strehl = Peak(PSF) / Peak(PSF_perfect)
           ~ Sum(OTF) / Sum(OTF_perfect)
    """
    # 1. Compute OTF (Fourier Transform of PSF)
    # OTF is the Autocorrelation of the pupil
    otf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    otf_ref = np.abs(np.fft.fftshift(np.fft.fft2(psf_ref)))
    
    # 2. Sum of OTF (Energy conservation check, effectively peak of PSF in freq domain)
    # This is more robust than single pixel peak in image domain
    strehl = np.sum(otf) / np.sum(otf_ref)
    
    return strehl * 100 # In percent

def integral_controller_explicit(current_command, slopes, reconstructor, gain):
    """
    Explicit Implementation of Integral Controller:
    u[k] = u[k-1] - g * R * s[k]
    """
    # 1. Matrix-Vector Multiply: Map Slopes (s) to Actuator Commands (delta_u)
    # R: Control Matrix (Reconstructor)
    delta_command = np.matmul(reconstructor, slopes)
    
    # 2. Integrator Update
    # The negative sign corrects the error
    next_command = current_command - gain * delta_command
    
    return next_command

# =============================================================================
# 2. MAIN SIMULATION
# =============================================================================

def main():
    print("=================================================================")
    print("   Explicit Shack-Hartmann AO Simulation (Deepened)   ")
    print("=================================================================")
    
    # --- 1. System Initialization ---
    print("\n[1] Initializing System Components...")
    
    # Telescope: 8m diameter, 20x20 subapertures (so ~40cm subaperture size)
    n_subaperture = 20
    d_subap = 8.0 / n_subaperture
    n_pix_pupil = 6 * n_subaperture # 6 pixels per subaperture
    tel = Telescope(resolution=n_pix_pupil, diameter=8.0, samplingTime=1/1000, centralObstruction=0.0)
    
    # Source: NGS at infinity
    ngs = Source(optBand='I', magnitude=8, coordinates=[0,0])
    ngs * tel # Couple source to telescope
    
    # Atmosphere: Single layer for simplicity in this demo, but moving
    atm = Atmosphere(telescope=tel, r0=0.15, L0=25, fractionalR0=[1.0], 
                     windSpeed=[10], windDirection=[0], altitude=[0])
    atm.initializeAtmosphere(tel)
    
    # Deformable Mirror: 21x21 actuators (Fried geometry)
    n_act = n_subaperture + 1
    dm = DeformableMirror(telescope=tel, nSubap=n_act-1, mechCoupling=0.35)
    
    # WFS: Shack-Hartmann
    wfs = ShackHartmann(nSubap=n_subaperture, telescope=tel, lightRatio=0.5)
    
    # Science Camera (High Res for Strehl)
    sci_cam = Detector(tel.resolution*2)
    
    # Reference PSF (Diffraction Limited)
    print("\n[2] Computing Reference PSF (Diffraction Limited)...")
    tel.resetOPD()
    ngs * tel * sci_cam
    psf_ref = sci_cam.frame.copy()
    
    # Get Reference Slopes (Flat Wavefront)
    print("    Acquiring Reference Slopes...")
    # We use our explicit function here to verify it works
    ref_slopes = get_slopes_diffractive_explicit(wfs)
    
    # --- 2. Calibration (Interaction Matrix) ---
    print("\n[3] Calibrating Interaction Matrix (Push-Pull)...")
    
    # We calibrate Zonal Modes (Actuators directly)
    # Or Modal? Let's do Zonal for explicit demonstration of D matrix
    # To save time, we just calibrate the central few actuators or use a pre-computed basis helper
    # For this demo, let's use the helper to get a KL basis, but manually build the IM
    
    from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
    M2C_KL = compute_KL_basis(tel, atm, dm, lim=0)
    n_modes = 20 # Keep it small for speed
    basis_modes = M2C_KL[:, :n_modes]
    
    n_meas = wfs.nSignal
    interaction_matrix = np.zeros((n_meas, n_modes))
    stroke = 1e-8
    
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
        
    dm.coefs[:] = 0 # Reset DM
    
    # --- 3. Reconstruction (SVD) ---
    print("\n[4] Computing Reconstructor (SVD Inversion)...")
    # D = U S V^T
    # R = V S_inv U^T
    U, s, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
    
    # Filter small singular values
    threshold = 1e-3
    s_inv = np.zeros_like(s)
    s_inv[s > threshold] = 1.0 / s[s > threshold]
    
    reconstructor_modal = Vt.T @ np.diag(s_inv) @ U.T
    
    # Convert Modal Reconstructor to Zonal (Actuator commands)
    # Cmd = M2C * R_modal * Slopes
    final_reconstructor = basis_modes @ reconstructor_modal
    
    # --- 4. Closed Loop Loop ---
    print("\n[5] Starting Closed Loop AO...")
    n_iter = 20
    gain = 0.4
    
    strehl_history = []
    
    for k in range(n_iter):
        # A. Move Atmosphere
        atm.update()
        
        # B. Forward Pass: Atmosphere -> Telescope -> DM -> WFS
        # Note: OOPAO overloads * operator for propagation
        atm * ngs * tel * dm
        
        # C. Measure Slopes (Explicit Function)
        # Note: We subtract ref_slopes to get residual aberrations
        slopes_meas = get_slopes_diffractive_explicit(wfs) - ref_slopes
        
        # D. Control (Explicit Integral Controller)
        dm.coefs = integral_controller_explicit(dm.coefs, slopes_meas, final_reconstructor, gain)
        
        # E. Evaluation (Science Path)
        atm * ngs * tel * dm * sci_cam
        sr = compute_strehl_explicit(sci_cam.frame, psf_ref)
        strehl_history.append(sr)
        
        print(f"    Iter {k+1:02d}: Strehl = {sr:.2f}%")
        
    # --- 5. Results ---
    print("\n[6] Simulation Finished.")
    plt.figure()
    plt.plot(strehl_history, 'o-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Strehl Ratio [%]')
    plt.title('Explicit Shack-Hartmann AO Performance')
    plt.savefig('sh_explicit_results.png')
    print("    Saved plot to sh_explicit_results.png")

if __name__ == "__main__":
    main()
```
