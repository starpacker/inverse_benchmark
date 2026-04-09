"""
RM-Tools — Faraday Rotation Measure Synthesis & CLEAN
======================================================
Task: Recover Faraday depth spectrum F(φ) from broadband polarization observations
Repo: https://github.com/CIRADA-Tools/RM-Tools
Paper: Brentjens & de Bruyn (2005), "Faraday rotation measure synthesis"

Inverse Problem:
    Forward: F(φ) → P(λ²) = ∫ F(φ) exp(2iφλ²) dφ
             A Faraday depth spectrum produces complex polarization as function of λ²
    Inverse: P(λ²) → F̂(φ) via RM synthesis (discrete Fourier transform in λ² space)
             followed by CLEAN deconvolution (Högbom-like) to deconvolve the RMSF

Usage:
    /data/yjh/rmtools_env/bin/python rmtools_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import gaussian_filter1d

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# Observation parameters
FREQ_MIN_MHZ = 800       # Minimum frequency (MHz)
FREQ_MAX_MHZ = 2000      # Maximum frequency (MHz)
N_CHANNELS = 300          # Number of frequency channels
NOISE_SIGMA_JY = 0.005   # Noise level in Jy/beam per channel

# Faraday depth parameters
PHI_MAX = 500.0           # Max Faraday depth (rad/m²)
D_PHI = 1.0               # Faraday depth resolution (rad/m²)
N_SAMPLES = 10.0          # Oversampling of RMSF

# Source parameters (ground truth Faraday depth spectrum)
# Two Faraday-thin components
COMPONENTS = [
    {'phi': 50.0,  'amplitude': 1.0, 'chi0': 0.3},  # RM=50, amp=1 Jy, chi0=0.3 rad
    {'phi': -120.0, 'amplitude': 0.5, 'chi0': 1.2},  # RM=-120, amp=0.5 Jy, chi0=1.2 rad
]
STOKES_I_FLUX = 10.0     # Total intensity (Jy) - flat spectrum assumed

# CLEAN parameters
CLEAN_CUTOFF_SIGMA = 2.0  # CLEAN down to 2-sigma noise level
CLEAN_GAIN = 0.05          # CLEAN loop gain (smaller for finer convergence)
CLEAN_MAX_ITER = 5000      # Max CLEAN iterations


# ═══════════════════════════════════════════════════════════
# 2. Forward Operator
# ═══════════════════════════════════════════════════════════
def forward_operator(phi_values, amplitudes, chi0_values, freq_hz):
    """
    Forward model: Faraday depth spectrum → complex polarization P(λ²).
    
    P(λ²) = Σ_i A_i · exp(2i·(χ0_i + φ_i·λ²))
    
    For Faraday-thin components:
        Q(ν) + iU(ν) = Σ_i A_i · exp(2i·φ_i·λ² + 2i·χ0_i)
    
    Args:
        phi_values: Faraday depths (rad/m²)
        amplitudes: Component amplitudes (Jy)
        chi0_values: Initial polarization angles (rad)
        freq_hz: Observation frequencies (Hz)
    
    Returns:
        P_complex: Complex polarization spectrum (Q + iU)
    """
    c = 2.998e8  # speed of light m/s
    lambda_sq = (c / freq_hz) ** 2  # λ² in m²
    
    P = np.zeros(len(freq_hz), dtype=complex)
    for phi, amp, chi0 in zip(phi_values, amplitudes, chi0_values):
        P += amp * np.exp(2j * (chi0 + phi * lambda_sq))
    
    return P, lambda_sq


def generate_faraday_depth_spectrum(phi_arr):
    """
    Generate ground truth Faraday depth spectrum F(φ) on a given φ grid.
    Each component is a delta function convolved with a narrow Gaussian
    for numerical representation.
    """
    F_gt = np.zeros(len(phi_arr), dtype=complex)
    
    for comp in COMPONENTS:
        phi0 = comp['phi']
        amp = comp['amplitude']
        chi0 = comp['chi0']
        
        # Represent as narrow Gaussian (delta-like)
        sigma_phi = D_PHI * 0.5
        gaussian = amp * np.exp(-(phi_arr - phi0)**2 / (2 * sigma_phi**2))
        phase = np.exp(2j * chi0)
        F_gt += gaussian * phase
    
    return F_gt


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def generate_data():
    """Generate synthetic broadband polarization observations."""
    # Frequency array
    freq_hz = np.linspace(FREQ_MIN_MHZ * 1e6, FREQ_MAX_MHZ * 1e6, N_CHANNELS)
    c = 2.998e8
    lambda_sq = (c / freq_hz) ** 2
    
    # Forward model: compute clean polarization
    phi_vals = [c['phi'] for c in COMPONENTS]
    amp_vals = [c['amplitude'] for c in COMPONENTS]
    chi0_vals = [c['chi0'] for c in COMPONENTS]
    
    P_clean, _ = forward_operator(phi_vals, amp_vals, chi0_vals, freq_hz)
    
    # Extract Q and U
    Q_clean = P_clean.real
    U_clean = P_clean.imag
    
    # Add noise
    Q_noisy = Q_clean + np.random.normal(0, NOISE_SIGMA_JY, N_CHANNELS)
    U_noisy = U_clean + np.random.normal(0, NOISE_SIGMA_JY, N_CHANNELS)
    
    # Stokes I (flat spectrum)
    I_arr = np.ones(N_CHANNELS) * STOKES_I_FLUX
    dI_arr = np.ones(N_CHANNELS) * NOISE_SIGMA_JY * 0.1  # I noise much lower
    
    # Noise arrays
    dQ_arr = np.ones(N_CHANNELS) * NOISE_SIGMA_JY
    dU_arr = np.ones(N_CHANNELS) * NOISE_SIGMA_JY
    
    observations = {
        'freq_hz': freq_hz,
        'lambda_sq': lambda_sq,
        'Q': Q_noisy,
        'U': U_noisy,
        'dQ': dQ_arr,
        'dU': dU_arr,
        'I': I_arr,
        'dI': dI_arr,
    }
    
    ground_truth = {
        'Q_clean': Q_clean,
        'U_clean': U_clean,
        'components': COMPONENTS,
    }
    
    print(f"  [FORWARD] Frequency range: {FREQ_MIN_MHZ}-{FREQ_MAX_MHZ} MHz, "
          f"{N_CHANNELS} channels")
    print(f"  [FORWARD] λ² range: [{lambda_sq.min():.6f}, {lambda_sq.max():.6f}] m²")
    print(f"  [FORWARD] Components: {len(COMPONENTS)}")
    for i, c in enumerate(COMPONENTS):
        print(f"    Component {i+1}: φ={c['phi']:.1f} rad/m², "
              f"A={c['amplitude']:.2f} Jy, χ₀={c['chi0']:.2f} rad")
    
    return observations, ground_truth


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver: RM Synthesis + CLEAN
# ═══════════════════════════════════════════════════════════
def reconstruct(observations):
    """
    RM Synthesis + Hogbom CLEAN to recover Faraday depth spectrum.
    
    Steps:
    1. RM Synthesis: Discrete Fourier transform of P(λ²) to get dirty F(φ)
    2. Compute RMSF (Rotation Measure Spread Function)
    3. CLEAN deconvolution to remove sidelobes
    
    Uses RM-Tools library functions.
    """
    from RMtools_1D.do_RMsynth_1D import do_rmsynth_planes, get_rmsf_planes
    from RMtools_1D.do_RMclean_1D import do_rmclean_hogbom
    
    lambda_sq = observations['lambda_sq']
    Q = observations['Q']
    U = observations['U']
    dQ = observations['dQ']
    dU = observations['dU']
    
    # Create Faraday depth array
    phi_arr = np.arange(-PHI_MAX, PHI_MAX + D_PHI, D_PHI)
    
    # Weight array (inverse variance)
    weight_arr = 1.0 / (dQ**2 + dU**2)
    
    # Step 1: Compute RMSF
    print("  [RM] Computing Rotation Measure Spread Function (RMSF)...")
    # phi2 array must be twice the length of phi_arr (for RMSF deconvolution)
    phi2_arr = np.arange(-2 * PHI_MAX, 2 * PHI_MAX + D_PHI, D_PHI)
    rmsf_results = get_rmsf_planes(
        lambdaSqArr_m2=lambda_sq,
        phiArr_radm2=phi2_arr,
        weightArr=weight_arr,
        lam0Sq_m2=None,
        nBits=64
    )
    RMSFArr = rmsf_results.RMSFcube
    phi2_arr = rmsf_results.phi2Arr  # Use the actual phi2 from RMSF result
    fwhmRMSF = float(rmsf_results.fwhmRMSFArr)
    print(f"  [RM] RMSF FWHM: {fwhmRMSF:.2f} rad/m²")
    print(f"  [RM] RMSF shape: {RMSFArr.shape}, phi2 shape: {phi2_arr.shape}")
    
    # Step 2: RM Synthesis (dirty FDF)
    print("  [RM] Computing dirty Faraday Dispersion Function (FDF)...")
    synth_results = do_rmsynth_planes(
        dataQ=Q,
        dataU=U,
        lambdaSqArr_m2=lambda_sq,
        phiArr_radm2=phi_arr,
        weightArr=weight_arr,
        lam0Sq_m2=None,
        nBits=64
    )
    dirtyFDF = synth_results.FDFcube
    lam0Sq = synth_results.lam0Sq_m2
    
    print(f"  [RM] Dirty FDF shape: {dirtyFDF.shape}")
    print(f"  [RM] Reference λ²: {lam0Sq:.6f} m²")
    
    # Step 3: CLEAN deconvolution
    print("  [RM] Running RM-CLEAN (Hogbom)...")
    noise_level = NOISE_SIGMA_JY / np.sqrt(len(lambda_sq))
    cutoff = CLEAN_CUTOFF_SIGMA * noise_level
    
    clean_results = do_rmclean_hogbom(
        dirtyFDF=dirtyFDF,
        phiArr_radm2=phi_arr,
        RMSFArr=RMSFArr,
        phi2Arr_radm2=phi2_arr,
        fwhmRMSFArr=np.array([fwhmRMSF]),
        cutoff=cutoff,
        maxIter=CLEAN_MAX_ITER,
        gain=CLEAN_GAIN,
        nBits=64,
        verbose=False,
        doPlots=False
    )
    
    # Handle both named tuple and regular tuple returns
    if hasattr(clean_results, 'cleanFDF'):
        cleanFDF = clean_results.cleanFDF
        ccArr = clean_results.ccArr
        iterCount = clean_results.iterCountArr
        residFDF = clean_results.residFDF
    else:
        # Fallback: tuple return (cleanFDF, ccArr, iterCount, residFDF)
        cleanFDF, ccArr, iterCount, residFDF = clean_results
    
    print(f"  [RM] CLEAN iterations: {iterCount}")
    print(f"  [RM] Clean FDF shape: {cleanFDF.shape}")
    
    return {
        'phi_arr': phi_arr,
        'dirtyFDF': dirtyFDF,
        'cleanFDF': cleanFDF,
        'ccArr': ccArr,
        'residFDF': residFDF,
        'RMSFArr': RMSFArr,
        'phi2_arr': phi2_arr,
        'fwhmRMSF': fwhmRMSF,
        'lam0Sq': lam0Sq,
    }


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(ground_truth, reconstruction):
    """Compute Faraday depth spectrum recovery metrics.
    
    Following standard radio interferometry practice, the GT spectrum is
    convolved with a Gaussian 'CLEAN beam' whose FWHM matches the RMSF
    before comparison.  This is analogous to comparing a CLEAN image with
    the model convolved with the restoring beam.
    """
    phi_arr = reconstruction['phi_arr']
    cleanFDF = reconstruction['cleanFDF']
    fwhmRMSF = reconstruction['fwhmRMSF']
    
    # Generate GT Faraday depth spectrum on same grid
    gt_FDF = generate_faraday_depth_spectrum(phi_arr)
    
    # Convolve GT amplitude with CLEAN beam (Gaussian with FWHM = RMSF FWHM)
    # sigma_beam in pixel units: FWHM / (2*sqrt(2*ln2)) / D_PHI
    sigma_beam_pix = fwhmRMSF / (2.0 * np.sqrt(2.0 * np.log(2.0))) / D_PHI
    gt_amp_raw = np.abs(gt_FDF)
    gt_amp = gaussian_filter1d(gt_amp_raw, sigma_beam_pix)
    
    # Use absolute values (amplitude spectrum)
    clean_amp = np.abs(cleanFDF)
    dirty_amp = np.abs(reconstruction['dirtyFDF'])
    
    # Normalize for comparison
    gt_amp_norm = gt_amp / (gt_amp.max() + 1e-10)
    clean_amp_norm = clean_amp / (clean_amp.max() + 1e-10)
    
    # RMSE
    rmse = np.sqrt(np.mean((gt_amp_norm - clean_amp_norm)**2))
    
    # Correlation coefficient
    cc = np.corrcoef(gt_amp_norm, clean_amp_norm)[0, 1]
    
    # Peak detection accuracy
    gt_peaks = []
    clean_peaks = []
    for comp in COMPONENTS:
        phi0 = comp['phi']
        # Find GT peak
        idx_gt = np.argmin(np.abs(phi_arr - phi0))
        gt_peaks.append(phi_arr[idx_gt])
        
        # Find clean peak within ±fwhmRMSF of true position
        fwhm = reconstruction['fwhmRMSF']
        mask = np.abs(phi_arr - phi0) < 2 * fwhm
        if np.any(mask):
            idx_clean = np.argmax(clean_amp[mask])
            clean_peaks.append(phi_arr[mask][idx_clean])
        else:
            clean_peaks.append(phi0)  # fallback
    
    # Peak position errors
    peak_errors = [abs(g - c) for g, c in zip(gt_peaks, clean_peaks)]
    avg_peak_error = np.mean(peak_errors)
    
    # Amplitude recovery at peak positions
    amp_recoveries = []
    for comp in COMPONENTS:
        phi0 = comp['phi']
        fwhm = reconstruction['fwhmRMSF']
        mask = np.abs(phi_arr - phi0) < fwhm
        if np.any(mask):
            peak_amp = np.max(clean_amp[mask])
            amp_recoveries.append(peak_amp / comp['amplitude'])
    avg_amp_recovery = np.mean(amp_recoveries) if amp_recoveries else 0.0
    
    # PSNR
    data_range = gt_amp_norm.max() - gt_amp_norm.min()
    mse = np.mean((gt_amp_norm - clean_amp_norm)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    return {
        'psnr': float(psnr),
        'rmse': float(rmse),
        'cc': float(cc),
        'avg_peak_position_error': float(avg_peak_error),
        'peak_position_errors': [float(e) for e in peak_errors],
        'avg_amplitude_recovery': float(avg_amp_recovery),
        'fwhm_rmsf': float(reconstruction['fwhmRMSF']),
        'gt_peaks': [float(p) for p in gt_peaks],
        'clean_peaks': [float(p) for p in clean_peaks],
    }


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(observations, ground_truth, reconstruction, metrics, save_path):
    """Generate comprehensive RM synthesis visualization."""
    phi_arr = reconstruction['phi_arr']
    dirtyFDF = reconstruction['dirtyFDF']
    cleanFDF = reconstruction['cleanFDF']
    phi2_arr = reconstruction['phi2_arr']
    RMSFArr = reconstruction['RMSFArr']
    
    # GT spectrum — convolve with CLEAN beam for fair comparison
    gt_FDF_raw = generate_faraday_depth_spectrum(phi_arr)
    fwhmRMSF = reconstruction['fwhmRMSF']
    sigma_beam_pix = fwhmRMSF / (2.0 * np.sqrt(2.0 * np.log(2.0))) / D_PHI
    gt_FDF_amp = gaussian_filter1d(np.abs(gt_FDF_raw), sigma_beam_pix)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (a) Input Q and U spectra
    freq_ghz = observations['freq_hz'] / 1e9
    axes[0, 0].plot(freq_ghz, observations['Q'], 'b.', ms=1, alpha=0.3, label='Q(ν)')
    axes[0, 0].plot(freq_ghz, observations['U'], 'r.', ms=1, alpha=0.3, label='U(ν)')
    axes[0, 0].plot(freq_ghz, ground_truth['Q_clean'], 'b-', lw=1, alpha=0.7)
    axes[0, 0].plot(freq_ghz, ground_truth['U_clean'], 'r-', lw=1, alpha=0.7)
    axes[0, 0].set_xlabel('Frequency (GHz)')
    axes[0, 0].set_ylabel('Flux (Jy/beam)')
    axes[0, 0].set_title('Stokes Q, U Spectra')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) RMSF
    center = len(phi2_arr) // 2
    width = min(200, len(phi2_arr) // 2)
    rmsf_slice = slice(center - width, center + width)
    axes[0, 1].plot(phi2_arr[rmsf_slice], np.abs(RMSFArr[rmsf_slice]), 'k-', lw=1.5)
    axes[0, 1].set_xlabel('φ (rad/m²)')
    axes[0, 1].set_ylabel('|RMSF|')
    axes[0, 1].set_title(f'RMSF (FWHM={metrics["fwhm_rmsf"]:.1f} rad/m²)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) Dirty FDF amplitude
    axes[0, 2].plot(phi_arr, np.abs(dirtyFDF), 'gray', lw=1, label='Dirty |F(φ)|')
    axes[0, 2].plot(phi_arr, gt_FDF_amp, 'b-', lw=2, label='GT |F(φ)| (convolved)')
    for comp in COMPONENTS:
        axes[0, 2].axvline(x=comp['phi'], color='b', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('φ (rad/m²)')
    axes[0, 2].set_ylabel('|F(φ)| (Jy/beam/RMSF)')
    axes[0, 2].set_title('Dirty Faraday Dispersion Function')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # (d) Clean FDF vs GT
    axes[1, 0].plot(phi_arr, gt_FDF_amp, 'b-', lw=2, label='GT |F(φ)| (convolved)')
    axes[1, 0].plot(phi_arr, np.abs(cleanFDF), 'r-', lw=1.5, label='CLEAN |F(φ)|')
    for i, comp in enumerate(COMPONENTS):
        axes[1, 0].axvline(x=comp['phi'], color='b', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=metrics['clean_peaks'][i], color='r', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('φ (rad/m²)')
    axes[1, 0].set_ylabel('|F(φ)| (Jy/beam/RMSF)')
    axes[1, 0].set_title(f'CLEAN FDF (CC={metrics["cc"]:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # (e) Zoom on components
    for i, comp in enumerate(COMPONENTS):
        phi0 = comp['phi']
        fwhm = metrics['fwhm_rmsf']
        zoom_range = (-fwhm * 5, fwhm * 5)
        mask = (phi_arr > phi0 + zoom_range[0]) & (phi_arr < phi0 + zoom_range[1])
        
        ax = axes[1, 1] if i == 0 else axes[1, 2]
        ax.plot(phi_arr[mask], gt_FDF_amp[mask], 'b-', lw=2, label='GT (convolved)')
        ax.plot(phi_arr[mask], np.abs(cleanFDF[mask]), 'r--', lw=2, label='CLEAN')
        ax.plot(phi_arr[mask], np.abs(dirtyFDF[mask]), 'gray', lw=1, alpha=0.5, label='Dirty')
        ax.axvline(x=phi0, color='b', linestyle='--', alpha=0.5, label=f'True φ={phi0}')
        ax.set_xlabel('φ (rad/m²)')
        ax.set_ylabel('|F(φ)|')
        ax.set_title(f'Component {i+1}: φ={phi0}, Δφ={metrics["peak_position_errors"][i]:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"RM-Tools — Faraday Rotation Measure Synthesis & CLEAN\n"
        f"PSNR={metrics['psnr']:.2f} dB | CC={metrics['cc']:.4f} | "
        f"Avg Peak Error={metrics['avg_peak_position_error']:.2f} rad/m² | "
        f"Amp Recovery={metrics['avg_amplitude_recovery']:.3f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  RM-Tools — Faraday Rotation Measure Synthesis")
    print("=" * 60)
    
    # (a) Generate data
    print("\n[DATA] Generating synthetic broadband polarization data...")
    observations, ground_truth = generate_data()
    
    # (b) Reconstruct
    print("\n[RECON] Running RM Synthesis + CLEAN...")
    reconstruction = reconstruct(observations)
    
    # (c) Evaluate
    print("\n[EVAL] Computing evaluation metrics...")
    metrics = compute_metrics(ground_truth, reconstruction)
    
    print(f"[EVAL] PSNR = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] CC = {metrics['cc']:.6f}")
    print(f"[EVAL] RMSE = {metrics['rmse']:.6f}")
    print(f"[EVAL] Avg peak position error = {metrics['avg_peak_position_error']:.4f} rad/m²")
    for i, (gt_p, cl_p, err) in enumerate(zip(metrics['gt_peaks'], metrics['clean_peaks'], 
                                                metrics['peak_position_errors'])):
        print(f"  Component {i+1}: GT φ={gt_p:.1f}, CLEAN φ={cl_p:.1f}, Δφ={err:.2f} rad/m²")
    print(f"[EVAL] Avg amplitude recovery = {metrics['avg_amplitude_recovery']:.4f}")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Save arrays
    phi_arr = reconstruction['phi_arr']
    gt_FDF = generate_faraday_depth_spectrum(phi_arr)
    # Convolve GT with CLEAN beam for saved arrays (consistent with metrics)
    fwhmRMSF = reconstruction['fwhmRMSF']
    sigma_beam_pix = fwhmRMSF / (2.0 * np.sqrt(2.0 * np.log(2.0))) / D_PHI
    gt_amp_conv = gaussian_filter1d(np.abs(gt_FDF), sigma_beam_pix)
    gt_real_conv = gaussian_filter1d(gt_FDF.real, sigma_beam_pix)
    gt_imag_conv = gaussian_filter1d(gt_FDF.imag, sigma_beam_pix)
    
    gt_data = np.stack([gt_amp_conv, gt_real_conv, gt_imag_conv], axis=0)
    recon_data = np.stack([np.abs(reconstruction['cleanFDF']),
                           reconstruction['cleanFDF'].real,
                           reconstruction['cleanFDF'].imag], axis=0)
    input_data = np.stack([observations['Q'], observations['U']], axis=0)
    
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_data)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_data)
    np.save(os.path.join(RESULTS_DIR, "input.npy"), input_data)
    print(f"[SAVE] GT FDF shape: {gt_data.shape} → ground_truth.npy")
    print(f"[SAVE] Clean FDF shape: {recon_data.shape} → reconstruction.npy")
    print(f"[SAVE] Input QU shape: {input_data.shape} → input.npy")
    
    # (f) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(observations, ground_truth, reconstruction, metrics, vis_path)
    
    print("\n" + "=" * 60)
    print("  DONE — RM-Tools Faraday RM Synthesis")
    print("=" * 60)
