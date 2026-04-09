"""
pyilc_cmb - CMB Component Separation via Internal Linear Combination
=====================================================================
From multi-frequency sky maps, extract the CMB signal using ILC.

Physics:
  - Simulate CMB (Gaussian random field) + synchrotron + dust at 6 frequencies
  - Forward: d_ν = a_ν * CMB + foregrounds(ν) + noise
  - Inverse: ILC weights  w = (a^T C^{-1} a)^{-1} C^{-1} a
  - Recovered CMB = w^T d
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time
from skimage.metrics import structural_similarity as ssim

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_101_pyilc_cmb"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
N_PIX     = 128          # map side length (pixels)
N_FREQ    = 6            # observation frequencies (GHz)
FREQS_GHZ = np.array([30.0, 44.0, 70.0, 100.0, 143.0, 217.0])
CMB_RMS   = 70.0         # μK — typical CMB RMS
NOISE_RMS = np.array([5.0, 5.0, 4.0, 3.0, 3.0, 5.0])   # μK per freq
SEED      = 42

np.random.seed(SEED)


# ====================================================================
# 1. Simulate sky components
# ====================================================================
def gaussian_random_field(N, power_law_index=-2.0, rms=1.0):
    """Generate a 2D Gaussian random field with power-law power spectrum."""
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0  # avoid division by zero
    power = K ** power_law_index
    power[0, 0] = 0.0  # zero mean
    phases = np.random.uniform(0, 2 * np.pi, (N, N))
    amplitudes = np.sqrt(power) * np.exp(1j * phases)
    field = np.fft.ifft2(amplitudes).real
    field = field / field.std() * rms
    return field


def create_cmb(N, rms=CMB_RMS):
    """Simulate CMB as Gaussian random field ~ l^(-2) power spectrum."""
    return gaussian_random_field(N, power_law_index=-2.0, rms=rms)


def synchrotron_template(N, freq_ghz, ref_freq=0.408):
    """Synchrotron emission: power law ∝ ν^{-3.0} from ref_freq GHz."""
    template = gaussian_random_field(N, power_law_index=-2.5, rms=200.0)
    return template * (freq_ghz / ref_freq) ** (-3.0)


def dust_template(N, freq_ghz, ref_freq=545.0):
    """Thermal dust emission: modified blackbody ∝ ν^{1.5} × B_ν."""
    template = gaussian_random_field(N, power_law_index=-2.5, rms=150.0)
    return template * (freq_ghz / ref_freq) ** 1.5


def simulate_observations(N, freqs):
    """
    Generate multi-frequency sky maps.
    
    Returns
    -------
    cmb_gt : (N, N) ground truth CMB map
    data   : (N_freq, N, N) observed maps
    """
    cmb_gt = create_cmb(N)
    data = np.zeros((len(freqs), N, N))
    
    for i, nu in enumerate(freqs):
        fg_sync = synchrotron_template(N, nu)
        fg_dust = dust_template(N, nu)
        noise = np.random.normal(0, NOISE_RMS[i], (N, N))
        # CMB has flat SED in thermodynamic temperature units: a_ν = 1
        data[i] = cmb_gt + fg_sync + fg_dust + noise
    
    return cmb_gt, data


# ====================================================================
# 2. ILC Inverse
# ====================================================================
def ilc_weights(data):
    """
    Compute ILC weights: w = (a^T C^{-1} a)^{-1} C^{-1} a
    where a = [1, 1, ..., 1] (CMB has unit response at all frequencies
    in thermodynamic temperature units).
    
    Parameters
    ----------
    data : (N_freq, N_pix_total) — flattened frequency maps
    
    Returns
    -------
    w : (N_freq,) ILC weights
    """
    n_freq = data.shape[0]
    a = np.ones(n_freq)
    
    # Covariance matrix
    C = np.cov(data)
    
    # Regularise slightly for numerical stability
    C += 1e-10 * np.eye(n_freq)
    
    C_inv = np.linalg.inv(C)
    
    # ILC weights
    w = C_inv @ a / (a @ C_inv @ a)
    return w


def recover_cmb(data):
    """
    Apply ILC to recover CMB map.
    
    Parameters
    ----------
    data : (N_freq, N, N) multi-frequency maps
    
    Returns
    -------
    cmb_rec : (N, N) recovered CMB
    w : (N_freq,) ILC weights
    """
    n_freq, ny, nx = data.shape
    data_flat = data.reshape(n_freq, -1)
    
    w = ilc_weights(data_flat)
    
    cmb_flat = w @ data_flat
    cmb_rec = cmb_flat.reshape(ny, nx)
    
    return cmb_rec, w


# ====================================================================
# 3. Metrics
# ====================================================================
def compute_metrics(cmb_gt, cmb_rec):
    """PSNR, SSIM, CC of recovered CMB."""
    mse = np.mean((cmb_gt - cmb_rec)**2)
    data_range = cmb_gt.max() - cmb_gt.min()
    psnr = 10.0 * np.log10(data_range**2 / mse) if mse > 0 else 100.0
    
    ssim_val = ssim(cmb_gt, cmb_rec, data_range=data_range)
    
    cc = float(np.corrcoef(cmb_gt.ravel(), cmb_rec.ravel())[0, 1])
    
    rmse = float(np.sqrt(mse))
    
    return psnr, ssim_val, cc, rmse


# ====================================================================
# 4. Visualization
# ====================================================================
def plot_results(freqs, data, cmb_gt, cmb_rec, weights, metrics_dict):
    """Multi-panel figure."""
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: selected frequency maps (3 of 6)
    sel = [0, 2, 5]  # 30, 70, 217 GHz
    for idx, si in enumerate(sel):
        ax = fig.add_subplot(3, 3, idx + 1)
        vmax = np.percentile(np.abs(data[si]), 99)
        ax.imshow(data[si], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"{freqs[si]:.0f} GHz", fontsize=10)
        ax.axis('off')
    
    # Row 2: GT CMB, recovered CMB, residual
    vmax_cmb = np.percentile(np.abs(cmb_gt), 99)
    
    ax = fig.add_subplot(3, 3, 4)
    ax.imshow(cmb_gt, cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title("GT CMB", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(3, 3, 5)
    ax.imshow(cmb_rec, cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title("Recovered CMB (ILC)", fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(3, 3, 6)
    residual = cmb_gt - cmb_rec
    vmax_res = np.percentile(np.abs(residual), 99)
    ax.imshow(residual, cmap='RdBu_r', vmin=-vmax_res, vmax=vmax_res)
    ax.set_title(f"Residual (RMS={np.std(residual):.1f} μK)", fontsize=10)
    ax.axis('off')
    
    # Row 3 left: ILC weights
    ax = fig.add_subplot(3, 3, 7)
    ax.bar(range(len(freqs)), weights, color='steelblue')
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f:.0f}" for f in freqs], fontsize=8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Weight")
    ax.set_title("ILC Weights")
    ax.axhline(0, color='k', lw=0.5)
    
    # Row 3 middle: power spectra comparison
    ax = fig.add_subplot(3, 3, 8)
    ps_gt = np.abs(np.fft.fft2(cmb_gt))**2
    ps_rec = np.abs(np.fft.fft2(cmb_rec))**2
    k = np.arange(1, N_PIX // 2)
    kx = np.fft.fftfreq(N_PIX, d=1.0) * N_PIX
    ky = np.fft.fftfreq(N_PIX, d=1.0) * N_PIX
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    cl_gt = np.zeros(len(k))
    cl_rec = np.zeros(len(k))
    for i, ki in enumerate(k):
        mask = (K >= ki - 0.5) & (K < ki + 0.5)
        if mask.sum() > 0:
            cl_gt[i] = ps_gt[mask].mean()
            cl_rec[i] = ps_rec[mask].mean()
    
    ax.loglog(k, cl_gt, 'b-', label='GT', lw=1.5)
    ax.loglog(k, cl_rec, 'r--', label='ILC', lw=1.5)
    ax.set_xlabel("Multipole ℓ")
    ax.set_ylabel("C_ℓ")
    ax.set_title("Angular Power Spectrum")
    ax.legend(fontsize=8)
    
    # Row 3 right: metrics text
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    txt = (f"PSNR = {metrics_dict['PSNR']:.2f} dB\n"
           f"SSIM = {metrics_dict['SSIM']:.4f}\n"
           f"CC   = {metrics_dict['CC']:.4f}\n"
           f"RMSE = {metrics_dict['RMSE']:.2f} μK\n"
           f"\nΣ weights = {weights.sum():.6f}")
    ax.text(0.1, 0.5, txt, fontsize=14, family='monospace',
            transform=ax.transAxes, verticalalignment='center')
    ax.set_title("Metrics Summary")
    
    plt.tight_layout()
    for path in [os.path.join(RESULTS_DIR, "vis_result.png"),
                 os.path.join(ASSETS_DIR, "vis_result.png")]:
        fig.savefig(path, dpi=150)
    plt.close(fig)


# ====================================================================
# 5. Main
# ====================================================================
def main():
    print("=" * 60)
    print("Task 101: CMB Component Separation (pyilc_cmb)")
    print("=" * 60)

    t0 = time.time()

    # Simulate
    print("\n[1] Simulating multi-frequency sky maps ...")
    cmb_gt, data = simulate_observations(N_PIX, FREQS_GHZ)

    # ILC recovery
    print("[2] Applying ILC ...")
    cmb_rec, weights = recover_cmb(data)

    elapsed = time.time() - t0
    print(f"    Elapsed: {elapsed:.1f} s")
    print(f"    ILC weights: {weights}")
    print(f"    Sum of weights: {weights.sum():.6f}")

    # Metrics
    print("[3] Computing metrics ...")
    psnr, ssim_val, cc, rmse = compute_metrics(cmb_gt, cmb_rec)

    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    SSIM = {ssim_val:.4f}")
    print(f"    CC   = {cc:.4f}")
    print(f"    RMSE = {rmse:.2f} μK")

    metrics = {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "CC": float(cc),
        "RMSE": float(rmse),
    }

    # Save
    print("[4] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), cmb_gt)
        np.save(os.path.join(d, "recon_output.npy"), cmb_rec)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Plot
    print("[5] Plotting ...")
    plot_results(FREQS_GHZ, data, cmb_gt, cmb_rec, weights, metrics)

    print(f"\n{'='*60}")
    print("Task 101 COMPLETE")
    print(f"{'='*60}")
    return metrics


if __name__ == "__main__":
    metrics = main()
