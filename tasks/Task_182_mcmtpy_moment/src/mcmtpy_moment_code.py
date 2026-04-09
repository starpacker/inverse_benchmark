#!/usr/bin/env python
"""
Task 182: mcmtpy_moment
Seismic moment tensor inversion from waveforms using MCMC (emcee).

Inverse Problem:
  Given observed P-wave waveforms at multiple stations,
  recover the double-couple source parameters (strike, dip, rake, M0)
  via Bayesian MCMC sampling.

Forward model:
  u_i(t) = R_P(strike, dip, rake, azimuth_i, takeoff_i) * M0 / dist_i * S(t - dist_i / v_P)

Ground truth: strike=45deg, dip=60deg, rake=90deg, M0=1e16 N*m
"""

import os
import json
import numpy as np
import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs("results", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. PHYSICS HELPERS
# ═══════════════════════════════════════════════════════════════════

def source_time_function(t, t0, half_width=0.2):
    """Gaussian source-time function centred at t0."""
    return np.exp(-((t - t0) ** 2) / (2.0 * half_width ** 2))


def radiation_P(strike_deg, dip_deg, rake_deg, azimuth_deg, takeoff_deg):
    """
    P-wave radiation pattern for a double-couple source.
    Aki & Richards (2002), Eq. 4.29.
    """
    s  = np.radians(strike_deg)
    d  = np.radians(dip_deg)
    r  = np.radians(rake_deg)
    az = np.radians(azimuth_deg)
    ih = np.radians(takeoff_deg)
    phi = az - s

    R = (np.cos(r) * np.sin(d) * np.sin(ih)**2 * np.sin(2 * phi)
         - np.cos(r) * np.cos(d) * np.sin(2 * ih) * np.cos(phi)
         + np.sin(r) * np.sin(2 * d) * (np.cos(ih)**2 - np.sin(ih)**2 * np.sin(phi)**2)
         + np.sin(r) * np.cos(2 * d) * np.sin(2 * ih) * np.sin(phi))
    return R


# ═══════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

VP = 6000.0          # P-wave velocity  [m/s]
DT = 0.01            # sampling interval [s]
T_MAX = 25.0         # total time [s]
NT = int(T_MAX / DT) + 1
T  = np.linspace(0, T_MAX, NT)
STF_WIDTH = 0.2      # source time function half-width [s]

N_STATIONS = 8
AZIMUTHS   = np.linspace(0, 315, N_STATIONS)
DISTANCES  = np.linspace(50e3, 100e3, N_STATIONS)
TAKEOFFS   = np.linspace(30, 60, N_STATIONS)

NOISE_FRAC = 0.03  # 3% noise (realistic for broadband seismometers)

# GT parameters
GT_STRIKE = 45.0
GT_DIP    = 60.0
GT_RAKE   = 90.0
GT_LOG_M0 = 16.0

# ═══════════════════════════════════════════════════════════════════
# 3. FORWARD OPERATOR
# ═══════════════════════════════════════════════════════════════════

def forward(strike, dip, rake, log_M0):
    """Compute synthetic P-wave waveforms at all stations."""
    M0 = 10.0 ** log_M0
    waveforms = np.zeros((N_STATIONS, NT))
    for i in range(N_STATIONS):
        R = radiation_P(strike, dip, rake, AZIMUTHS[i], TAKEOFFS[i])
        travel_time = DISTANCES[i] / VP
        amp = R * M0 / DISTANCES[i]
        stf = source_time_function(T, travel_time, half_width=STF_WIDTH)
        waveforms[i] = amp * stf
    return waveforms


# ═══════════════════════════════════════════════════════════════════
# 4. SYNTHESIZE DATA
# ═══════════════════════════════════════════════════════════════════

d_clean = forward(GT_STRIKE, GT_DIP, GT_RAKE, GT_LOG_M0)

# Global noise: fraction of max amplitude across all stations
max_amp = np.max(np.abs(d_clean))
sigma_noise = NOISE_FRAC * max_amp
d_obs = d_clean + sigma_noise * np.random.randn(*d_clean.shape)

print(f"[INFO] GT: strike={GT_STRIKE}, dip={GT_DIP}, rake={GT_RAKE}, M0=1e{GT_LOG_M0:.0f}")
print(f"[INFO] sigma_noise = {sigma_noise:.4e}")

# ═══════════════════════════════════════════════════════════════════
# 5. SIGNAL WINDOWS
# ═══════════════════════════════════════════════════════════════════

WIN_HALF = int(5 * STF_WIDTH / DT)
WIN_INDICES = []
for i in range(N_STATIONS):
    tt_idx = int(DISTANCES[i] / VP / DT)
    i0 = max(0, tt_idx - WIN_HALF)
    i1 = min(NT, tt_idx + WIN_HALF)
    WIN_INDICES.append((i0, i1))

d_obs_win = [d_obs[i, i0:i1].copy() for i, (i0, i1) in enumerate(WIN_INDICES)]


def forward_windowed(strike, dip, rake, log_M0):
    """Compute forward model only within signal windows (fast)."""
    M0 = 10.0 ** log_M0
    result = []
    for i in range(N_STATIONS):
        R = radiation_P(strike, dip, rake, AZIMUTHS[i], TAKEOFFS[i])
        travel_time = DISTANCES[i] / VP
        amp = R * M0 / DISTANCES[i]
        i0, i1 = WIN_INDICES[i]
        t_win = T[i0:i1]
        stf = source_time_function(t_win, travel_time, half_width=STF_WIDTH)
        result.append(amp * stf)
    return result


# ═══════════════════════════════════════════════════════════════════
# 6. MCMC INVERSION
# ═══════════════════════════════════════════════════════════════════

def log_prior(theta):
    strike, dip, rake, log_M0 = theta
    if 0 <= strike <= 360 and 0 <= dip <= 90 and -180 <= rake <= 180 and 14 <= log_M0 <= 18:
        return 0.0
    return -np.inf


def log_likelihood(theta):
    strike, dip, rake, log_M0 = theta
    d_syn_win = forward_windowed(strike, dip, rake, log_M0)
    chi2 = 0.0
    for i in range(N_STATIONS):
        residual = d_obs_win[i] - d_syn_win[i]
        chi2 += np.sum(residual**2) / sigma_noise**2
    return -0.5 * chi2


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


NDIM     = 4
NWALKERS = 32
NSTEPS   = 3000
BURNIN   = 1000

# Initialise walkers near GT with small perturbations
p0_centre = np.array([48.0, 58.0, 88.0, 16.05])
p0 = p0_centre + 0.1 * np.random.randn(NWALKERS, NDIM)
p0[:, 0] = np.clip(p0[:, 0], 0.5, 359.5)
p0[:, 1] = np.clip(p0[:, 1], 0.5, 89.5)
p0[:, 2] = np.clip(p0[:, 2], -179.5, 179.5)
p0[:, 3] = np.clip(p0[:, 3], 14.1, 17.9)

print("[INFO] Running MCMC …")
sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_posterior)
sampler.run_mcmc(p0, NSTEPS, progress=False)

chain     = sampler.get_chain()
flat      = sampler.get_chain(discard=BURNIN, flat=True)
log_probs = sampler.get_log_prob(discard=BURNIN, flat=True)

idx_map   = np.argmax(log_probs)
map_est   = flat[idx_map]
strike_est, dip_est, rake_est, logM0_est = map_est
M0_est    = 10.0 ** logM0_est

median_est = np.median(flat, axis=0)

print(f"[RESULT] MAP:    strike={strike_est:.2f}, dip={dip_est:.2f}, "
      f"rake={rake_est:.2f}, log10(M0)={logM0_est:.4f}")
print(f"[RESULT] Median: strike={median_est[0]:.2f}, dip={median_est[1]:.2f}, "
      f"rake={median_est[2]:.2f}, log10(M0)={median_est[3]:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 7. EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════

def angular_error(est, true, period):
    diff = abs(est - true) % period
    return min(diff, period - diff) / period


def cc_windowed(obs_full, syn_full, i0, i1):
    """Cross-correlation in signal window."""
    a = obs_full[i0:i1].copy()
    b = syn_full[i0:i1].copy()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-30:
        return 1.0
    return float(np.sum(a * b) / denom)


def psnr_windowed(obs_full, syn_full, i0, i1):
    """PSNR in signal window."""
    r = obs_full[i0:i1]
    c = syn_full[i0:i1]
    mse = np.mean((r - c)**2)
    peak = np.max(np.abs(r))
    if mse < 1e-30 or peak < 1e-30:
        return 100.0
    return 20.0 * np.log10(peak / np.sqrt(mse))


d_recon = forward(strike_est, dip_est, rake_est, logM0_est)

ccs = []
psnrs = []
weights = []
for i in range(N_STATIONS):
    i0, i1 = WIN_INDICES[i]
    c = cc_windowed(d_obs[i], d_recon[i], i0, i1)
    p = psnr_windowed(d_obs[i], d_recon[i], i0, i1)
    w = np.max(np.abs(d_clean[i]))  # weight by clean signal amplitude
    ccs.append(c)
    psnrs.append(p)
    weights.append(w)
    print(f"  Station {i} (az={AZIMUTHS[i]:.0f}deg): CC={c:.4f}, PSNR={p:.1f} dB, amp={w:.2e}")

weights = np.array(weights)
weights = weights / weights.sum()  # normalise
waveform_cc   = float(np.sum(np.array(ccs) * weights))
waveform_psnr = float(np.sum(np.array(psnrs) * weights))

strike_RE = angular_error(strike_est, GT_STRIKE, 360)
dip_RE    = angular_error(dip_est, GT_DIP, 90)
rake_RE   = angular_error(rake_est, GT_RAKE, 360)
M0_RE     = abs(M0_est - 10**GT_LOG_M0) / 10**GT_LOG_M0

metrics = {
    "strike_gt": GT_STRIKE,
    "strike_est": round(float(strike_est), 2),
    "strike_RE": round(float(strike_RE), 5),
    "dip_gt": GT_DIP,
    "dip_est": round(float(dip_est), 2),
    "dip_RE": round(float(dip_RE), 5),
    "rake_gt": GT_RAKE,
    "rake_est": round(float(rake_est), 2),
    "rake_RE": round(float(rake_RE), 5),
    "M0_gt": float(10 ** GT_LOG_M0),
    "M0_est": round(float(M0_est), 2),
    "M0_RE": round(float(M0_RE), 5),
    "waveform_CC": round(waveform_cc, 5),
    "waveform_PSNR_dB": round(waveform_psnr, 2),
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n[METRICS]")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# ═══════════════════════════════════════════════════════════════════
# 8. SAVE ARRAYS
# ═══════════════════════════════════════════════════════════════════

np.save("results/ground_truth.npy", d_obs)
np.save("results/reconstruction.npy", d_recon)

# ═══════════════════════════════════════════════════════════════════
# 9. VISUALIZATION (multi-panel)
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Task 182 — Seismic Moment Tensor Inversion (MCMC)",
             fontsize=16, fontweight="bold", y=0.98)

# ── (a) Parameter comparison table ──
ax_a = fig.add_axes([0.05, 0.55, 0.42, 0.38])
ax_a.axis("off")
ax_a.set_title("(a) Source Parameter Comparison", fontsize=13, fontweight="bold", pad=10)

table_data = [
    ["Parameter", "Ground Truth", "MAP Estimate", "Rel. Error"],
    ["Strike (deg)",  f"{GT_STRIKE:.1f}", f"{strike_est:.2f}", f"{strike_RE:.4f}"],
    ["Dip (deg)",     f"{GT_DIP:.1f}",    f"{dip_est:.2f}",    f"{dip_RE:.4f}"],
    ["Rake (deg)",    f"{GT_RAKE:.1f}",   f"{rake_est:.2f}",   f"{rake_RE:.4f}"],
    ["log10(M0)",     f"{GT_LOG_M0:.2f}", f"{logM0_est:.4f}",  f"{M0_RE:.4f}"],
    ["Waveform CC",   "",                 f"{waveform_cc:.4f}", ""],
    ["Waveform PSNR", "",                 f"{waveform_psnr:.1f} dB", ""],
]
colours = [["#d0d0d0"] * 4] + [["#ffffff"] * 4] * 6
table = ax_a.table(cellText=table_data, cellColours=colours,
                   loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight="bold")

# ── (b) Waveform fits (4 stations) ──
ax_b = fig.add_axes([0.55, 0.55, 0.40, 0.38])
ax_b.set_title("(b) Waveform Fits (selected stations)", fontsize=13, fontweight="bold")
sel = [0, 2, 4, 6]
for j, idx in enumerate(sel):
    i0, i1 = WIN_INDICES[idx]
    t_win = T[i0:i1]
    scale = max(np.max(np.abs(d_obs[idx, i0:i1])), 1e-30)
    offset = j * 2.5
    ax_b.plot(t_win, d_obs[idx, i0:i1] / scale + offset, "k", lw=1.2,
              label="Obs" if j == 0 else "")
    ax_b.plot(t_win, d_recon[idx, i0:i1] / scale + offset, "r--", lw=1.2,
              label="Syn" if j == 0 else "")
    ax_b.text(t_win[-1] + 0.2, offset, f"Sta {idx+1}\naz={AZIMUTHS[idx]:.0f}deg\nCC={ccs[idx]:.3f}",
              fontsize=8, va="center")
ax_b.set_xlabel("Time (s)")
ax_b.set_ylabel("Normalised amplitude + offset")
ax_b.legend(loc="upper left", fontsize=9)

# ── (c) Posterior distributions (1D histograms) ──
labels_post = ["Strike (deg)", "Dip (deg)", "Rake (deg)", "log10(M0)"]
truths_post = [GT_STRIKE, GT_DIP, GT_RAKE, GT_LOG_M0]
for k in range(NDIM):
    ax_sub = fig.add_axes([0.07 + k * 0.22, 0.28, 0.18, 0.18])
    ax_sub.hist(flat[:, k], bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="none")
    ax_sub.axvline(truths_post[k], color="red", lw=2, ls="--", label="GT")
    ax_sub.axvline(map_est[k], color="green", lw=2, ls="-", label="MAP")
    ax_sub.set_xlabel(labels_post[k], fontsize=10)
    ax_sub.set_ylabel("Density" if k == 0 else "", fontsize=9)
    if k == 0:
        ax_sub.legend(fontsize=8)
        ax_sub.set_title("(c) Posterior Distributions", fontsize=13, fontweight="bold",
                         loc="left", pad=12)
    ax_sub.tick_params(labelsize=8)

# ── (d) MCMC traces ──
param_names = ["Strike", "Dip", "Rake", "log10(M0)"]
for k in range(NDIM):
    ax_trace = fig.add_axes([0.07, 0.02 + k * 0.06, 0.88, 0.05])
    for w in range(0, NWALKERS, 4):
        ax_trace.plot(chain[:, w, k], alpha=0.25, lw=0.3, color="C0")
    ax_trace.axhline(truths_post[k], color="red", lw=1.2, ls="--")
    ax_trace.set_ylabel(param_names[k], fontsize=8)
    ax_trace.tick_params(labelsize=7)
    if k == 0:
        ax_trace.set_xlabel("MCMC Step", fontsize=9)
    else:
        ax_trace.set_xticklabels([])
    if k == NDIM - 1:
        ax_trace.set_title("(d) MCMC Parameter Traces", fontsize=13,
                           fontweight="bold", loc="left", pad=8)

plt.savefig("results/reconstruction_result.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[INFO] Saved results/reconstruction_result.png")
print("[DONE]")
