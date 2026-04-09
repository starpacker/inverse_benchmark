import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from phasorpy.phasor import (
    phasor_from_lifetime,
    phasor_from_signal,
    phasor_semicircle,
)

# Import target function
from agent_run_inversion import run_inversion


# === Inject Referee (evaluate_results) verbatim from Reference B ===
def evaluate_results(
    f1_gt: np.ndarray,
    f1_recon: np.ndarray,
    tau1_ns: float,
    tau2_ns: float,
    freq_mhz: float,
    total_photons: int,
    nx: int,
    ny: int,
    G_meas: np.ndarray,
    S_meas: np.ndarray,
    G_ref: np.ndarray,
    S_ref: np.ndarray,
    outdir: str,
) -> dict:
    """
    Evaluate reconstruction quality and generate visualizations.
    """
    os.makedirs(outdir, exist_ok=True)

    f2_gt = 1.0 - f1_gt

    # Compute metrics
    psnr_val = psnr(f1_gt, f1_recon, data_range=1.0)
    ssim_val = ssim(f1_gt, f1_recon, data_range=1.0)

    # Lifetime-related relative errors
    tau_eff_gt = f1_gt * tau1_ns + f2_gt * tau2_ns
    tau_eff_recon = f1_recon * tau1_ns + (1 - f1_recon) * tau2_ns
    tau_re = np.mean(np.abs(tau_eff_gt - tau_eff_recon) / tau_eff_gt)

    # Mean absolute error of fraction
    mae_f1 = np.mean(np.abs(f1_gt - f1_recon))

    metrics = {
        "PSNR_dB": round(float(psnr_val), 2),
        "SSIM": round(float(ssim_val), 6),
        "fraction_MAE": round(float(mae_f1), 6),
        "lifetime_eff_RE": round(float(tau_re), 6),
        "tau1_ns": tau1_ns,
        "tau2_ns": tau2_ns,
        "frequency_MHz": freq_mhz,
        "image_size": [nx, ny],
        "total_photons_per_pixel": total_photons,
    }

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arrays
    np.save(os.path.join(outdir, "ground_truth.npy"), f1_gt)
    np.save(os.path.join(outdir, "recon_output.npy"), f1_recon)

    print("\nSaved ground_truth.npy, recon_output.npy, metrics.json")

    # Visualization: 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # Panel 1: GT fraction map
    im0 = axes[0, 0].imshow(f1_gt, cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[0, 0].set_title("Ground Truth: Species-1 Fraction", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], label="$f_1$")

    # Panel 2: Phasor plot
    ax_ph = axes[0, 1]
    sc_g, sc_s = phasor_semicircle()
    ax_ph.plot(sc_g, sc_s, "k-", linewidth=1.5, label="Universal semicircle")
    step = max(1, nx * ny // 3000)
    g_flat = G_meas.ravel()[::step]
    s_flat = S_meas.ravel()[::step]
    f1_flat = f1_gt.ravel()[::step]
    sc = ax_ph.scatter(g_flat, s_flat, c=f1_flat, cmap="viridis", s=3, alpha=0.5,
                       vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax_ph, label="$f_1$ (GT)")
    ax_ph.plot(G_ref[0], S_ref[0], "r^", markersize=12, label=f"τ₁={tau1_ns} ns")
    ax_ph.plot(G_ref[1], S_ref[1], "bs", markersize=12, label=f"τ₂={tau2_ns} ns")
    ax_ph.set_xlabel("G (real)", fontsize=11)
    ax_ph.set_ylabel("S (imaginary)", fontsize=11)
    ax_ph.set_title("Phasor Plot", fontsize=12)
    ax_ph.set_xlim(-0.05, 1.05)
    ax_ph.set_ylim(-0.05, 0.6)
    ax_ph.set_aspect("equal")
    ax_ph.legend(fontsize=9, loc="upper right")

    # Panel 3: Reconstructed fraction map
    im2 = axes[1, 0].imshow(f1_recon, cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[1, 0].set_title(
        f"Reconstructed: Species-1 Fraction\nPSNR={psnr_val:.1f} dB, SSIM={ssim_val:.4f}",
        fontsize=12,
    )
    plt.colorbar(im2, ax=axes[1, 0], label="$f_1$")

    # Panel 4: Error map
    error = f1_recon - f1_gt
    emax = max(abs(error.min()), abs(error.max()), 0.05)
    im3 = axes[1, 1].imshow(error, cmap="RdBu_r", vmin=-emax, vmax=emax, origin="lower")
    axes[1, 1].set_title(f"Error (Recon − GT), MAE={mae_f1:.4f}", fontsize=12)
    plt.colorbar(im3, ax=axes[1, 1], label="$\\Delta f_1$")

    fig.suptitle(
        "Task 142: Phasor-based FLIM Lifetime Component Analysis\n"
        f"τ₁={tau1_ns} ns, τ₂={tau2_ns} ns, freq={freq_mhz} MHz, "
        f"{total_photons} photons/px",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figpath = os.path.join(outdir, "reconstruction_result.png")
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {figpath}")

    return metrics


def main():
    data_paths = ['/data/yjh/phasorpy_flim_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of positional args: {len(args)}")
    print(f"Keyword args keys: {list(kwargs.keys())}")

    # Pattern detection
    if len(inner_paths) > 0:
        # Chained execution
        print("\n=== Chained Execution Pattern Detected ===")
        print("Running outer function to get operator...")
        agent_operator = run_inversion(*args, **kwargs)

        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        print("Running operator on inner data...")
        agent_result = agent_operator(*inner_args, **inner_kwargs)
    else:
        # Direct execution
        print("\n=== Direct Execution Pattern ===")
        print("Running run_inversion...")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        std_result = std_output

    # Now we need to evaluate both agent_result and std_result
    # We need f1_gt and other params for evaluate_results
    # Extract parameters from the input args
    # From the function signature: run_inversion(noisy_signal, freq_mhz, tau1_ns, tau2_ns)
    if len(args) >= 4:
        noisy_signal = args[0]
        freq_mhz = args[1]
        tau1_ns = args[2]
        tau2_ns = args[3]
    else:
        noisy_signal = args[0] if len(args) > 0 else kwargs.get('noisy_signal')
        freq_mhz = args[1] if len(args) > 1 else kwargs.get('freq_mhz')
        tau1_ns = args[2] if len(args) > 2 else kwargs.get('tau1_ns')
        tau2_ns = args[3] if len(args) > 3 else kwargs.get('tau2_ns')

    print(f"\nParameters: freq_mhz={freq_mhz}, tau1_ns={tau1_ns}, tau2_ns={tau2_ns}")
    if hasattr(noisy_signal, 'shape'):
        print(f"Signal shape: {noisy_signal.shape}")

    # We need f1_gt for evaluation. It's not directly in the inputs.
    # We'll use the standard output's f1_recon as the "ground truth" proxy
    # OR we can compare the two results' quality metrics against each other.
    
    # Since evaluate_results requires f1_gt which we don't have in the pkl directly,
    # we use the standard result's f1_recon as our ground truth reference.
    # This is reasonable: the standard implementation's output is our baseline.

    # Extract results
    if isinstance(agent_result, dict):
        agent_f1 = agent_result.get('f1_recon')
        agent_G_meas = agent_result.get('G_meas')
        agent_S_meas = agent_result.get('S_meas')
        agent_G_ref = agent_result.get('G_ref')
        agent_S_ref = agent_result.get('S_ref')
    else:
        print("ERROR: agent_result is not a dict")
        sys.exit(1)

    if isinstance(std_result, dict):
        std_f1 = std_result.get('f1_recon')
        std_G_meas = std_result.get('G_meas')
        std_S_meas = std_result.get('S_meas')
        std_G_ref = std_result.get('G_ref')
        std_S_ref = std_result.get('S_ref')
    else:
        print("ERROR: std_result is not a dict")
        sys.exit(1)

    nx, ny = std_f1.shape[0], std_f1.shape[1]
    # Estimate total_photons from the signal
    if hasattr(noisy_signal, 'shape') and len(noisy_signal.shape) == 3:
        total_photons = int(np.mean(np.sum(noisy_signal, axis=-1)))
    else:
        total_photons = 1000  # fallback

    print(f"Image size: ({nx}, {ny}), Estimated photons/px: {total_photons}")

    # Use std_f1 as ground truth to evaluate agent
    # This checks if the agent produces results consistent with the standard
    outdir_agent = '/tmp/eval_agent'
    outdir_std = '/tmp/eval_std'

    print("\n--- Evaluating Agent Output (against standard as GT) ---")
    try:
        metrics_agent = evaluate_results(
            f1_gt=std_f1,
            f1_recon=agent_f1,
            tau1_ns=tau1_ns,
            tau2_ns=tau2_ns,
            freq_mhz=freq_mhz,
            total_photons=total_photons,
            nx=nx,
            ny=ny,
            G_meas=agent_G_meas,
            S_meas=agent_S_meas,
            G_ref=agent_G_ref,
            S_ref=agent_S_ref,
            outdir=outdir_agent,
        )
    except Exception as e:
        print(f"ERROR evaluating agent: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Also evaluate std against itself (should be perfect)
    print("\n--- Evaluating Standard Output (self-reference) ---")
    try:
        metrics_std = evaluate_results(
            f1_gt=std_f1,
            f1_recon=std_f1,
            tau1_ns=tau1_ns,
            tau2_ns=tau2_ns,
            freq_mhz=freq_mhz,
            total_photons=total_photons,
            nx=nx,
            ny=ny,
            G_meas=std_G_meas,
            S_meas=std_S_meas,
            G_ref=std_G_ref,
            S_ref=std_S_ref,
            outdir=outdir_std,
        )
    except Exception as e:
        print(f"ERROR evaluating standard: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract primary metrics
    agent_psnr = metrics_agent['PSNR_dB']
    agent_ssim = metrics_agent['SSIM']
    agent_mae = metrics_agent['fraction_MAE']

    std_psnr = metrics_std['PSNR_dB']
    std_ssim = metrics_std['SSIM']
    std_mae = metrics_std['fraction_MAE']

    print(f"\n{'='*60}")
    print(f"Scores -> Agent PSNR: {agent_psnr} dB, Standard PSNR: {std_psnr} dB")
    print(f"Scores -> Agent SSIM: {agent_ssim}, Standard SSIM: {std_ssim}")
    print(f"Scores -> Agent MAE: {agent_mae}, Standard MAE: {std_mae}")
    print(f"{'='*60}")

    # Verification: Since std vs std is perfect (infinite PSNR, SSIM=1, MAE=0),
    # we need agent results to be very close. 
    # For a correct implementation, agent_f1 should match std_f1 almost exactly.
    # Check PSNR is high enough (>= 40 dB means very close match)
    # and MAE is very small

    success = True

    # PSNR check: if agent matches standard well, PSNR should be very high
    # A PSNR > 30 dB indicates very good match
    if agent_psnr < 30.0:
        print(f"FAIL: Agent PSNR ({agent_psnr} dB) is below 30 dB threshold")
        success = False
    else:
        print(f"PASS: Agent PSNR ({agent_psnr} dB) >= 30 dB")

    # SSIM check
    if agent_ssim < 0.90:
        print(f"FAIL: Agent SSIM ({agent_ssim}) is below 0.90 threshold")
        success = False
    else:
        print(f"PASS: Agent SSIM ({agent_ssim}) >= 0.90")

    # MAE check
    if agent_mae > 0.05:
        print(f"FAIL: Agent MAE ({agent_mae}) exceeds 0.05 threshold")
        success = False
    else:
        print(f"PASS: Agent MAE ({agent_mae}) <= 0.05")

    # Additional sanity checks
    # Check that G_ref and S_ref match
    g_ref_diff = np.max(np.abs(agent_G_ref - std_G_ref))
    s_ref_diff = np.max(np.abs(agent_S_ref - std_S_ref))
    print(f"\nReference phasor diffs: G_ref max diff = {g_ref_diff:.8f}, S_ref max diff = {s_ref_diff:.8f}")

    if g_ref_diff > 1e-6 or s_ref_diff > 1e-6:
        print("WARNING: Reference phasor coordinates differ significantly")
        # Not necessarily a failure, but worth noting

    # Check shapes match
    if agent_f1.shape != std_f1.shape:
        print(f"FAIL: Shape mismatch: agent {agent_f1.shape} vs std {std_f1.shape}")
        success = False

    if success:
        print("\n=== ALL CHECKS PASSED ===")
        sys.exit(0)
    else:
        print("\n=== SOME CHECKS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)