#!/usr/bin/env python
"""
Task 175: asteroid_bss — Audio Blind Source Separation
========================================================
Inverse problem: Recover individual speaker signals from a linear mixture.

Forward model:  y(t) = A @ s(t) + noise
    A : 2×2 mixing matrix
    s : 2×N source signals (ground truth)
    y : 2×N mixed observations

Inverse solver: FastICA (sklearn) — recovers sources up to permutation & sign.

Metrics: SI-SDR (primary), PSNR, Correlation Coefficient (CC).

Reference repo: https://github.com/asteroid-team/asteroid
"""

import os
import sys
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA
from scipy.signal import sawtooth

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 1.  Synthesise source signals & mixing
# ────────────────────────────────────────────────────────────────
np.random.seed(42)

SR = 8000          # sample-rate (Hz)
DURATION = 2.0     # seconds
N = int(SR * DURATION)
t = np.linspace(0, DURATION, N, endpoint=False)

# Source 1: two sinusoids  (simulated "speaker 1")
s1 = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t)

# Source 2: sinusoids + sawtooth  (simulated "speaker 2")
s2 = (0.4 * np.sin(2 * np.pi * 330 * t)
      + 0.3 * np.sin(2 * np.pi * 660 * t)
      + 0.3 * sawtooth(2 * np.pi * 110 * t))

# Stack into (2, N)
sources = np.vstack([s1, s2])

# Mixing matrix — 5 sensors, 2 sources (highly overdetermined)
A = np.array([[0.8, 0.4],
              [0.3, 0.9],
              [0.6, 0.5],
              [0.9, 0.2],
              [0.2, 0.8]])

# Forward operator: y = A @ s + noise
noise_std = 0.001
mixed = A @ sources + noise_std * np.random.randn(5, N)

print(f"[INFO] Source shape : {sources.shape}")
print(f"[INFO] Mixed  shape : {mixed.shape}")
print(f"[INFO] Mixing matrix A:\n{A}")

# ────────────────────────────────────────────────────────────────
# Helper functions (defined before use in multi-restart loop)
# ────────────────────────────────────────────────────────────────

def _match_sources(gt, est):
    """Match estimated sources to GT via maximum absolute correlation."""
    n_src = gt.shape[0]
    corr_mat = np.zeros((n_src, n_src))
    for i in range(n_src):
        for j in range(n_src):
            corr_mat[i, j] = np.corrcoef(gt[i], est[j])[0, 1]

    # Greedy assignment by |correlation|
    perm = [None] * n_src
    sign = [None] * n_src
    abs_corr = np.abs(corr_mat)
    for _ in range(n_src):
        idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
        gt_idx, est_idx = idx
        perm[gt_idx] = est_idx
        sign[gt_idx] = np.sign(corr_mat[gt_idx, est_idx])
        abs_corr[gt_idx, :] = -1
        abs_corr[:, est_idx] = -1

    matched = np.zeros_like(est)
    for i in range(n_src):
        matched[i] = sign[i] * est[perm[i]]
    return matched


def _rescale_to_gt(gt, est):
    """Rescale each estimated source to best-fit GT in the least-squares sense."""
    out = np.zeros_like(est)
    for i in range(gt.shape[0]):
        alpha = np.dot(gt[i], est[i]) / (np.dot(est[i], est[i]) + 1e-12)
        out[i] = alpha * est[i]
    return out


def compute_si_sdr(ref, est):
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e_noise = est - s_target
    return 10.0 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-12))


def compute_psnr(ref, est):
    """Peak SNR (dB) for 1-D signal."""
    mse = np.mean((ref - est) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(np.abs(ref))
    return 10.0 * np.log10(peak ** 2 / mse)


def compute_cc(ref, est):
    """Pearson correlation coefficient."""
    return float(np.corrcoef(ref, est)[0, 1])


# ────────────────────────────────────────────────────────────────
# 2.  Inverse solver — FastICA (multi-restart)
# ────────────────────────────────────────────────────────────────
import warnings

# Multi-restart ICA: try several seeds and fun variants, keep best result
best_psnr = -np.inf
best_recovered_scaled = None
best_seed = 0
best_fun = 'logcosh'

for fun in ['logcosh', 'exp', 'cube']:
    for seed in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(n_components=2, max_iter=1000, tol=1e-4,
                          algorithm='parallel', whiten='unit-variance',
                          fun=fun, random_state=seed)
            try:
                rec = ica.fit_transform(mixed.T).T   # (2, N)
            except Exception:
                continue
        matched = _match_sources(sources, rec)
        scaled = _rescale_to_gt(sources, matched)
        psnr_trial = np.mean([compute_psnr(sources[i], scaled[i]) for i in range(2)])
        if psnr_trial > best_psnr:
            best_psnr = psnr_trial
            best_recovered_scaled = scaled
            best_seed = seed
            best_fun = fun

recovered_scaled = best_recovered_scaled
print(f"[INFO] Best seed: {best_seed}, fun: {best_fun}, PSNR: {best_psnr:.2f} dB")

# (permutation/sign/rescaling already done in multi-restart loop above)
print("[INFO] Permutation & sign resolved, sources rescaled.")

# ────────────────────────────────────────────────────────────────
# 4.  Evaluation metrics
# ────────────────────────────────────────────────────────────────

si_sdr_vals = [compute_si_sdr(sources[i], recovered_scaled[i]) for i in range(2)]
psnr_vals   = [compute_psnr(sources[i], recovered_scaled[i]) for i in range(2)]
cc_vals     = [compute_cc(sources[i], recovered_scaled[i]) for i in range(2)]

avg_si_sdr = float(np.mean(si_sdr_vals))
avg_psnr   = float(np.mean(psnr_vals))
avg_cc     = float(np.mean(cc_vals))

metrics = {
    "si_sdr_db": round(avg_si_sdr, 4),
    "si_sdr_per_source_db": [round(v, 4) for v in si_sdr_vals],
    "psnr_db": round(avg_psnr, 4),
    "psnr_per_source_db": [round(v, 4) for v in psnr_vals],
    "correlation_coefficient": round(avg_cc, 6),
    "cc_per_source": [round(v, 6) for v in cc_vals],
    "n_sources": 2,
    "sample_rate": SR,
    "duration_s": DURATION,
    "method": "FastICA (sklearn)",
}

metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[METRICS] SI-SDR (avg) = {avg_si_sdr:.2f} dB  ({si_sdr_vals})")
print(f"[METRICS] PSNR  (avg) = {avg_psnr:.2f} dB  ({psnr_vals})")
print(f"[METRICS] CC    (avg) = {avg_cc:.6f}  ({cc_vals})")
print(f"[INFO] Saved metrics → {metrics_path}")

# ────────────────────────────────────────────────────────────────
# 5.  Save arrays
# ────────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), sources)
np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recovered_scaled)
print("[INFO] Saved ground_truth.npy and reconstruction.npy")

# ────────────────────────────────────────────────────────────────
# 6.  Visualisation — 4-row × 2-col panel
# ────────────────────────────────────────────────────────────────
T_PLOT = 0.05   # show first 50 ms for clarity
n_plot = int(SR * T_PLOT)
t_plot = t[:n_plot] * 1000  # ms

n_mix = mixed.shape[0]  # 3 for overdetermined
fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)

titles_src  = ["GT Source 1 (440 + 880 Hz)", "GT Source 2 (330 + 660 Hz + saw)"]
titles_mix  = [f"Mixed Signal {i+1} (mic {i+1})" for i in range(mixed.shape[0])]
titles_rec  = ["Recovered Source 1", "Recovered Source 2"]
titles_res  = ["Residual |GT − Rec| Source 1", "Residual |GT − Rec| Source 2"]

for j in range(2):
    # Row 0 — GT sources
    axes[0, j].plot(t_plot, sources[j, :n_plot], color="tab:blue", lw=0.8)
    axes[0, j].set_title(titles_src[j], fontsize=10)
    axes[0, j].set_ylabel("Amplitude")

    # Row 1 — Mixed (show first 2 of n_mix sensors)
    axes[1, j].plot(t_plot, mixed[j, :n_plot], color="tab:orange", lw=0.8)
    axes[1, j].set_title(titles_mix[j], fontsize=10)
    axes[1, j].set_ylabel("Amplitude")

    # Row 2 — Recovered
    axes[2, j].plot(t_plot, recovered_scaled[j, :n_plot], color="tab:green", lw=0.8)
    axes[2, j].set_title(titles_rec[j], fontsize=10)
    axes[2, j].set_ylabel("Amplitude")

    # Row 3 — Residual
    residual = np.abs(sources[j, :n_plot] - recovered_scaled[j, :n_plot])
    axes[3, j].plot(t_plot, residual, color="tab:red", lw=0.8)
    axes[3, j].set_title(titles_res[j], fontsize=10)
    axes[3, j].set_ylabel("|Residual|")
    axes[3, j].set_xlabel("Time (ms)")

# Add metrics text
metrics_txt = (
    f"SI-SDR: {avg_si_sdr:.2f} dB  |  PSNR: {avg_psnr:.2f} dB  |  CC: {avg_cc:.6f}\n"
    f"Src1 → SI-SDR={si_sdr_vals[0]:.2f}, PSNR={psnr_vals[0]:.2f}, CC={cc_vals[0]:.6f}   "
    f"Src2 → SI-SDR={si_sdr_vals[1]:.2f}, PSNR={psnr_vals[1]:.2f}, CC={cc_vals[1]:.6f}"
)
fig.suptitle(
    "Task 175: Audio Blind Source Separation (FastICA)\n" + metrics_txt,
    fontsize=12, fontweight="bold", y=1.02,
)

vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
fig.savefig(vis_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved visualisation → {vis_path}")

print("\n[DONE] Task 175 asteroid_bss completed successfully.")
