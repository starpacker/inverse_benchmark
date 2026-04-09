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
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis


def load_and_preprocess_data(n_subaperture=20, n_modes=20, stroke=1e-8):
    """
    Initialize all system components: telescope, source, atmosphere, DM, WFS, detector.
    Compute reference PSF, reference slopes, and calibrate the interaction matrix.
    
    Returns:
        dict: Contains all initialized components and calibration data.
    """
    # Telescope: 8m diameter
    n_pix_pupil = 6 * n_subaperture  # 6 pixels per subaperture
    tel = Telescope(resolution=n_pix_pupil, diameter=8.0, samplingTime=1/1000, centralObstruction=0.0)
    
    # Source: NGS at infinity
    ngs = Source(optBand='I', magnitude=8, coordinates=[0, 0])
    ngs * tel  # Couple source to telescope
    
    # Atmosphere: Single layer
    atm = Atmosphere(telescope=tel, r0=0.15, L0=25, fractionalR0=[1.0],
                     windSpeed=[10], windDirection=[0], altitude=[0])
    atm.initializeAtmosphere(tel)
    
    # Deformable Mirror: (n_subaperture+1) x (n_subaperture+1) actuators
    dm = DeformableMirror(telescope=tel, nSubap=n_subaperture, mechCoupling=0.35)
    
    # WFS: Shack-Hartmann
    wfs = ShackHartmann(nSubap=n_subaperture, telescope=tel, lightRatio=0.5)
    
    # Science Camera (High Res for Strehl)
    sci_cam = Detector(tel.resolution * 2)
    
    # Compute Reference PSF (Diffraction Limited)
    tel.resetOPD()
    ngs * tel * sci_cam
    psf_ref = sci_cam.frame.copy()
    
    # Get Reference Slopes (Flat Wavefront)
    ref_slopes = _get_slopes_diffractive(wfs, tel, ngs)
    
    # Compute KL basis for calibration
    M2C_KL = compute_KL_basis(tel, atm, dm, lim=0)
    basis_modes = M2C_KL[:, :n_modes]
    
    # Build Interaction Matrix via Push-Pull
    n_meas = wfs.nSignal
    interaction_matrix = np.zeros((n_meas, n_modes))
    
    for i in range(n_modes):
        # Push
        dm.coefs = basis_modes[:, i] * stroke
        ngs * tel * dm
        slopes_push = _get_slopes_diffractive(wfs, tel, ngs)
        
        # Pull
        dm.coefs = -basis_modes[:, i] * stroke
        ngs * tel * dm
        slopes_pull = _get_slopes_diffractive(wfs, tel, ngs)
        
        # IM Column
        interaction_matrix[:, i] = (slopes_push - slopes_pull) / (2 * stroke)
    
    dm.coefs[:] = 0  # Reset DM
    
    # Compute Reconstructor via SVD
    U, s, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
    threshold = 1e-3
    s_inv = np.zeros_like(s)
    s_inv[s > threshold] = 1.0 / s[s > threshold]
    reconstructor_modal = Vt.T @ np.diag(s_inv) @ U.T
    
    # Convert Modal Reconstructor to Zonal (Actuator commands)
    final_reconstructor = basis_modes @ reconstructor_modal
    
    return {
        'tel': tel,
        'ngs': ngs,
        'atm': atm,
        'dm': dm,
        'wfs': wfs,
        'sci_cam': sci_cam,
        'psf_ref': psf_ref,
        'ref_slopes': ref_slopes,
        'reconstructor': final_reconstructor,
        'n_modes': n_modes
    }


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


def forward_operator(system_data, dm_commands=None):
    """
    Forward operator: Propagate light through the optical system.
    Atmosphere -> Source -> Telescope -> DM -> Science Camera
    
    Returns the resulting PSF on the science camera.
    """
    tel = system_data['tel']
    ngs = system_data['ngs']
    atm = system_data['atm']
    dm = system_data['dm']
    sci_cam = system_data['sci_cam']
    
    if dm_commands is not None:
        dm.coefs = dm_commands
    
    # Propagate through optical train
    atm * ngs * tel * dm * sci_cam
    
    return sci_cam.frame.copy()


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
    return strehl * 100  # In percent


def evaluate_results(inversion_results, psf_ref, output_path='sh_explicit_results.png'):
    """
    Evaluate and visualize the AO correction results.
    
    Args:
        inversion_results: dict from run_inversion
        psf_ref: reference diffraction-limited PSF
        output_path: path to save the figure
    
    Returns:
        dict: Summary statistics
    """
    strehl_history = inversion_results['strehl_history']
    final_psf = inversion_results['final_psf']
    final_dm_commands = inversion_results['final_dm_commands']
    
    # Compute statistics
    initial_strehl = strehl_history[0] if len(strehl_history) > 0 else 0.0
    final_strehl = strehl_history[-1] if len(strehl_history) > 0 else 0.0
    mean_strehl = np.mean(strehl_history)
    max_strehl = np.max(strehl_history)
    min_strehl = np.min(strehl_history)
    
    # RMS of DM commands
    dm_rms = np.sqrt(np.mean(final_dm_commands ** 2))
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Strehl history
    axes[0].plot(np.arange(1, len(strehl_history) + 1), strehl_history, 'o-', linewidth=2, markersize=4)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Strehl Ratio [%]')
    axes[0].set_title('Strehl Ratio Evolution')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, len(strehl_history) + 1])
    
    # Plot 2: Final PSF (log scale)
    psf_display = np.log10(final_psf + 1e-10)
    im1 = axes[1].imshow(psf_display, cmap='hot', origin='lower')
    axes[1].set_title(f'Final PSF (log scale)\nStrehl = {final_strehl:.2f}%')
    axes[1].set_xlabel('Pixels')
    axes[1].set_ylabel('Pixels')
    plt.colorbar(im1, ax=axes[1], label='log10(Intensity)')
    
    # Plot 3: Reference PSF (log scale)
    psf_ref_display = np.log10(psf_ref + 1e-10)
    im2 = axes[2].imshow(psf_ref_display, cmap='hot', origin='lower')
    axes[2].set_title('Reference PSF (Diffraction Limited)')
    axes[2].set_xlabel('Pixels')
    axes[2].set_ylabel('Pixels')
    plt.colorbar(im2, ax=axes[2], label='log10(Intensity)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print summary
    print("\n" + "=" * 60)
    print("                    EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Initial Strehl:     {initial_strehl:.2f}%")
    print(f"  Final Strehl:       {final_strehl:.2f}%")
    print(f"  Mean Strehl:        {mean_strehl:.2f}%")
    print(f"  Max Strehl:         {max_strehl:.2f}%")
    print(f"  Min Strehl:         {min_strehl:.2f}%")
    print(f"  DM RMS Command:     {dm_rms:.2e} m")
    print(f"  Number of Iters:    {len(strehl_history)}")
    print(f"  Output saved to:    {output_path}")
    print("=" * 60)
    
    return {
        'initial_strehl': initial_strehl,
        'final_strehl': final_strehl,
        'mean_strehl': mean_strehl,
        'max_strehl': max_strehl,
        'min_strehl': min_strehl,
        'dm_rms': dm_rms,
        'n_iterations': len(strehl_history)
    }


if __name__ == '__main__':
    print("=================================================================")
    print("   Explicit Shack-Hartmann AO Simulation (Refactored)   ")
    print("=================================================================")
    
    # Configuration parameters
    N_SUBAPERTURE = 20
    N_MODES = 20
    STROKE = 1e-8
    N_ITER = 20
    GAIN = 0.4
    OUTPUT_PATH = 'sh_explicit_results.png'
    
    # Step 1: Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    print("    - Initializing telescope, source, atmosphere, DM, WFS")
    print("    - Computing reference PSF and slopes")
    print("    - Calibrating interaction matrix")
    system_data = load_and_preprocess_data(
        n_subaperture=N_SUBAPERTURE,
        n_modes=N_MODES,
        stroke=STROKE
    )
    print("    Data loading complete.")
    
    # Step 2: Demonstrate forward operator
    print("\n[2] Testing forward operator...")
    test_psf = forward_operator(system_data, dm_commands=None)
    print(f"    Forward operator output shape: {test_psf.shape}")
    print(f"    PSF max value: {np.max(test_psf):.4e}")
    
    # Step 3: Run inversion (closed-loop AO)
    print("\n[3] Running closed-loop AO inversion...")
    print(f"    Iterations: {N_ITER}, Gain: {GAIN}")
    inversion_results = run_inversion(
        system_data,
        n_iter=N_ITER,
        gain=GAIN
    )
    
    # Print iteration-by-iteration results
    for k, sr in enumerate(inversion_results['strehl_history']):
        print(f"    Iter {k+1:02d}: Strehl = {sr:.2f}%")
    
    # Step 4: Evaluate results
    print("\n[4] Evaluating results...")
    evaluation_summary = evaluate_results(
        inversion_results,
        psf_ref=system_data['psf_ref'],
        output_path=OUTPUT_PATH
    )
    
    print("\nOPTIMIZATION_FINISHED_SUCCESSFULLY")