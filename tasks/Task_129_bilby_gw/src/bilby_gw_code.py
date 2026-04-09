#!/usr/bin/env python
"""
Bilby GW Parameter Estimation Benchmark
========================================
Recover compact binary coalescence (CBC) parameters from simulated
gravitational-wave strain data.

Strategy (fast, benchmarkable):
  1. Use bilby + LAL to generate a CBC waveform with known parameters
  2. Add realistic Gaussian noise from LIGO PSD
  3. Use bilby's dynesty sampler with aggressive settings
     - Only 3 free parameters (chirp_mass, mass_ratio, luminosity_distance)
     - All other params fixed to truth
     - nlive=100, tight priors, loose convergence
  4. Compare recovered waveform vs injected (ground truth)
  5. Compute PSNR, correlation, waveform match, parameter errors

Inverse problem: h(t) + n(t) -> {theta_source}

Reference: Ashton et al., ApJS 2019 (bilby paper)
"""

import os
import sys
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bilby

# ── Configuration ──────────────────────────────────────────────────────
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTDIR, exist_ok=True)
bilby.core.utils.setup_logger(outdir=OUTDIR, label="bilby_gw", log_level="WARNING")

DURATION = 4          # seconds of data
SAMPLING_FREQ = 2048  # Hz
MINIMUM_FREQ = 20.0   # Hz low-frequency cutoff
REFERENCE_FREQ = 50.0 # Hz
APPROX = "IMRPhenomPv2"  # fast frequency-domain waveform

# ── Injection parameters (ground truth) ───────────────────────────────
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=500.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

m1, m2 = injection_parameters["mass_1"], injection_parameters["mass_2"]
true_chirp_mass = (m1 * m2) ** (3.0 / 5) / (m1 + m2) ** (1.0 / 5)
true_mass_ratio = m2 / m1
print(f"True chirp_mass = {true_chirp_mass:.4f}, mass_ratio = {true_mass_ratio:.4f}")

# ── Waveform generator ───────────────────────────────────────────────
waveform_arguments = dict(
    waveform_approximant=APPROX,
    reference_frequency=REFERENCE_FREQ,
    minimum_frequency=MINIMUM_FREQ,
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=DURATION,
    sampling_frequency=SAMPLING_FREQ,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# ── Interferometers with injected signal + Gaussian noise ─────────────
print("[1/7] Setting up interferometers with injected signal...")
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=SAMPLING_FREQ,
    duration=DURATION,
    start_time=injection_parameters["geocent_time"] - DURATION + 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
)

# ── Compute ground-truth signal ──────────────────────────────────────
print("[2/7] Computing ground-truth waveform...")
polarisations = waveform_generator.frequency_domain_strain(injection_parameters)
freq_array = waveform_generator.frequency_array
h1 = ifos[0]
gt_strain_fd = h1.get_detector_response(polarisations, injection_parameters)
noisy_strain_fd = h1.strain_data.frequency_domain_strain.copy()

# ── Priors (3 free params) ───────────────────────────────────────────
print("[3/7] Setting up priors (3 free parameters)...")
priors = bilby.gw.prior.BBHPriorDict()

# Fix everything except chirp_mass, mass_ratio, luminosity_distance
for key in ["ra", "dec", "psi", "phase", "tilt_1", "tilt_2",
            "phi_12", "phi_jl", "a_1", "a_2", "theta_jn", "geocent_time"]:
    priors[key] = injection_parameters[key]

priors["chirp_mass"] = bilby.core.prior.Uniform(
    name="chirp_mass", minimum=25.0, maximum=32.0,
    latex_label="$\\mathcal{M}$",
)
priors["mass_ratio"] = bilby.core.prior.Uniform(
    name="mass_ratio", minimum=0.5, maximum=1.0,
    latex_label="$q$",
)
priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
    name="luminosity_distance", minimum=200.0, maximum=1000.0, unit="Mpc",
)

# ── Likelihood ────────────────────────────────────────────────────────
print("[4/7] Setting up likelihood...")
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
)

# ── Run sampler ───────────────────────────────────────────────────────
print("[5/7] Running nested sampling (dynesty, nlive=100, 3 free params)...")
print("       Expected runtime: 2-8 minutes...")
t_start = time.time()

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=100,
    nact=3,
    maxmcmc=2000,
    walks=5,
    dlogz=0.5,
    outdir=OUTDIR,
    label="bilby_gw",
    injection_parameters=injection_parameters,
    save=True,
    resume=False,
    clean=True,
)

t_elapsed = time.time() - t_start
print(f"       Sampling completed in {t_elapsed:.1f} s ({t_elapsed/60:.1f} min)")

# ── Extract results ───────────────────────────────────────────────────
print("[6/7] Extracting results and computing metrics...")
posterior = result.posterior

# Derive mass_1, mass_2 from chirp_mass, mass_ratio
if "mass_1" not in posterior.columns:
    q = posterior["mass_ratio"]
    mc = posterior["chirp_mass"]
    eta = q / (1 + q) ** 2
    mtotal = mc / eta ** (3.0 / 5)
    posterior["mass_1"] = mtotal / (1 + q)
    posterior["mass_2"] = mtotal * q / (1 + q)

estimated_params = ["chirp_mass", "mass_ratio", "luminosity_distance"]
derived_params = ["mass_1", "mass_2"]
all_report_params = estimated_params + derived_params

map_idx = posterior["log_likelihood"].idxmax()
median_params = {}
map_params = {}
for p in all_report_params:
    if p in posterior.columns:
        median_params[p] = float(np.median(posterior[p]))
        map_params[p] = float(posterior[p].iloc[map_idx])

true_values = dict(injection_parameters)
true_values["chirp_mass"] = true_chirp_mass
true_values["mass_ratio"] = true_mass_ratio

print("\n=== Parameter Recovery ===")
print(f"{'Parameter':<25} {'True':>12} {'MAP':>12} {'Median':>12} {'RelErr%':>10}")
print("-" * 75)
relative_errors = {}
for p in all_report_params:
    true_val = true_values[p]
    map_val = map_params.get(p, np.nan)
    med_val = median_params.get(p, np.nan)
    rel_err = abs(med_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-10 else 0.0
    relative_errors[p] = rel_err
    print(f"{p:<25} {true_val:>12.4f} {map_val:>12.4f} {med_val:>12.4f} {rel_err:>9.2f}%")

# ── Reconstruct waveform from MAP parameters ─────────────────────────
recon_params = dict(injection_parameters)
for p in estimated_params:
    recon_params[p] = map_params[p]
q_map = map_params["mass_ratio"]
mc_map = map_params["chirp_mass"]
eta_map = q_map / (1 + q_map) ** 2
mtotal_map = mc_map / eta_map ** (3.0 / 5)
recon_params["mass_1"] = mtotal_map / (1 + q_map)
recon_params["mass_2"] = mtotal_map * q_map / (1 + q_map)

recon_polarisations = waveform_generator.frequency_domain_strain(recon_params)
recon_signal_h1 = h1.get_detector_response(recon_polarisations, recon_params)

# ── Signal-level metrics ─────────────────────────────────────────────
mask = (freq_array >= MINIMUM_FREQ) & (freq_array <= SAMPLING_FREQ / 2)
gt_signal = gt_strain_fd[mask]
recon_signal = recon_signal_h1[mask]

gt_abs = np.abs(gt_signal)
recon_abs = np.abs(recon_signal)
mse = np.mean((gt_abs - recon_abs) ** 2)
max_val = np.max(gt_abs)
psnr = 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float("inf")

corr_complex = np.abs(np.vdot(gt_signal, recon_signal)) / (
    np.linalg.norm(gt_signal) * np.linalg.norm(recon_signal)
)

def inner_product(a, b, psd, df):
    return 4.0 * df * np.real(np.sum(np.conj(a) * b / psd))

psd_array = h1.power_spectral_density_array[mask]
df = freq_array[1] - freq_array[0]
psd_safe = np.where(psd_array > 0, psd_array, np.inf)

overlap = inner_product(gt_signal, recon_signal, psd_safe, df)
norm_gt = np.sqrt(inner_product(gt_signal, gt_signal, psd_safe, df))
norm_recon = np.sqrt(inner_product(recon_signal, recon_signal, psd_safe, df))
match = overlap / (norm_gt * norm_recon) if (norm_gt > 0 and norm_recon > 0) else 0.0

mean_rel_error = np.mean(list(relative_errors.values()))

snr_list = [float(ifo.meta_data.get("optimal_SNR", 0.0)) for ifo in ifos]
network_snr = np.sqrt(sum(s ** 2 for s in snr_list))

print(f"\n=== Signal Metrics ===")
print(f"PSNR (freq-domain amplitude): {psnr:.2f} dB")
print(f"Correlation coefficient:       {corr_complex:.6f}")
print(f"Waveform match (overlap):      {match:.6f}")
print(f"Mean param relative error:     {mean_rel_error:.2f}%")
print(f"Network SNR (injected):        {network_snr:.1f}")
print(f"Runtime:                       {t_elapsed:.1f} s")

# ── Save metrics ──────────────────────────────────────────────────────
metrics = {
    "psnr_db": round(float(psnr), 2),
    "correlation": round(float(corr_complex), 6),
    "waveform_match": round(float(match), 6),
    "mean_relative_error_pct": round(float(mean_rel_error), 2),
    "network_snr": round(float(network_snr), 2),
    "runtime_s": round(float(t_elapsed), 1),
    "nlive": 100,
    "n_free_params": 3,
    "sampler": "dynesty",
    "parameter_errors": {p: round(float(v), 4) for p, v in relative_errors.items()},
    "recovered_parameters": {p: round(float(v), 4) for p, v in median_params.items()},
    "true_parameters": {p: round(float(true_values[p]), 4) for p in all_report_params},
}
metrics_path = os.path.join(OUTDIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to {metrics_path}")

# ── Save numpy arrays ────────────────────────────────────────────────
print("[7/7] Saving outputs and creating visualization...")
from scipy.fft import irfft

n_samples = int(DURATION * SAMPLING_FREQ)
gt_td = np.real(irfft(gt_strain_fd, n=n_samples))
recon_td = np.real(irfft(recon_signal_h1, n=n_samples))
noisy_td = np.real(irfft(noisy_strain_fd, n=n_samples))
time_array = np.arange(n_samples) / SAMPLING_FREQ

np.save(os.path.join(OUTDIR, "ground_truth.npy"), gt_td)
np.save(os.path.join(OUTDIR, "reconstruction.npy"), recon_td)
print("Saved ground_truth.npy and reconstruction.npy")

# ── Visualization ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
t_start_data = injection_parameters["geocent_time"] - DURATION + 2
t_plot = time_array + t_start_data - injection_parameters["geocent_time"]

ax = axes[0, 0]
ax.plot(t_plot, noisy_td, color="lightgray", alpha=0.5, linewidth=0.3,
        label="Noisy data", rasterized=True)
ax.plot(t_plot, gt_td, color="tab:blue", linewidth=1.0, label="True signal")
ax.set_xlabel("Time relative to merger (s)")
ax.set_ylabel("Strain")
ax.set_title("Detector Strain (H1): Data + Injected Signal")
ax.legend(loc="upper left")
ax.set_xlim(-0.5, 0.05)

ax = axes[0, 1]
ax.plot(t_plot, gt_td, color="tab:blue", linewidth=1.2, label="True signal")
ax.plot(t_plot, recon_td, color="tab:red", linewidth=1.0, linestyle="--",
        label="Recovered (MAP)")
ax.set_xlabel("Time relative to merger (s)")
ax.set_ylabel("Strain")
ax.set_title(f"Waveform Recovery (match={match:.4f})")
ax.legend(loc="upper left")
ax.set_xlim(-0.5, 0.05)

ax = axes[1, 0]
param_labels = {
    "chirp_mass": "$\\mathcal{M}$",
    "mass_ratio": "$q$",
    "luminosity_distance": "$d_L$",
    "mass_1": "$m_1$",
    "mass_2": "$m_2$",
}
x_pos = np.arange(len(all_report_params))
rel_errs = [relative_errors[p] for p in all_report_params]
colors = ["tab:green" if e < 5 else "tab:orange" if e < 15 else "tab:red" for e in rel_errs]
ax.bar(x_pos, rel_errs, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([param_labels.get(p, p) for p in all_report_params], fontsize=11)
ax.set_ylabel("Relative Error (%)")
ax.set_title("Parameter Recovery Accuracy")
ax.axhline(y=5, color="green", linestyle=":", alpha=0.5, label="5%")
ax.axhline(y=15, color="orange", linestyle=":", alpha=0.5, label="15%")
ax.legend()

ax = axes[1, 1]
freq_plot = freq_array[mask]
ax.loglog(freq_plot, np.abs(gt_signal), color="tab:blue", linewidth=1.0,
          label="True signal")
ax.loglog(freq_plot, np.abs(recon_signal), color="tab:red", linewidth=0.8,
          linestyle="--", label="Recovered (MAP)")
ax.loglog(freq_plot, np.sqrt(psd_safe), color="lightgray", linewidth=0.5,
          alpha=0.8, label="$\\sqrt{S_n(f)}$")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|h(f)|")
ax.set_title(f"Frequency-Domain Comparison (PSNR={psnr:.1f} dB)")
ax.legend(loc="upper right")
ax.set_xlim(MINIMUM_FREQ, SAMPLING_FREQ / 2)

plt.suptitle(
    f"Bilby GW Parameter Estimation: CBC Waveform Recovery\n"
    f"Match={match:.4f} | PSNR={psnr:.1f} dB | Corr={corr_complex:.4f} | "
    f"Mean RelErr={mean_rel_error:.1f}% | SNR={network_snr:.1f} | "
    f"Runtime={t_elapsed:.0f}s",
    fontsize=12, y=1.02,
)
plt.tight_layout()
fig_path = os.path.join(OUTDIR, "reconstruction_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved to {fig_path}")

try:
    result.plot_corner(
        parameters=["chirp_mass", "mass_ratio", "luminosity_distance"],
        filename=os.path.join(OUTDIR, "corner_plot.png"),
        save=True,
    )
    print("Corner plot saved.")
except Exception as e:
    print(f"Corner plot skipped: {e}")

print("\n=== DONE ===")
print(f"All outputs saved to {OUTDIR}/")
