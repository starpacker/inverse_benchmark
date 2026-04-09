"""
Operational Modal Analysis (OMA): Identify modal parameters from ambient vibration data.

Inverse Problem: Given multi-channel acceleration time histories from ambient vibration,
infer structural modal parameters (natural frequencies, damping ratios, mode shapes).

Forward Model: M*a + C*v + K*x = F(t) for a multi-DOF spring-mass-damper system.
Inverse Solver: Frequency Domain Decomposition (FDD) with Enhanced FDD (EFDD) for
damping via free-decay fitting of autocorrelation of SDOF bell functions.

Reference: pyOMA2 (https://github.com/dagghe/pyOMA2) - Operational Modal Analysis library.
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import numpy as np
from scipy import linalg, signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "repo")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEFINE STRUCTURAL SYSTEM  (5-DOF spring-mass-damper)
# ═══════════════════════════════════════════════════════════════════════════════
n_dof = 5
masses = np.array([1.0, 1.0, 1.0, 1.0, 1.0])                    # kg
stiffnesses = np.array([200.0, 180.0, 150.0, 120.0, 100.0])      # N/m
damping_ratios_true = np.array([0.02, 0.025, 0.02, 0.03, 0.025]) # target zeta

# Mass matrix
M = np.diag(masses)

# Stiffness matrix (tridiagonal for chain system)
K = np.zeros((n_dof, n_dof))
for i in range(n_dof):
    K[i, i] += stiffnesses[i]
    if i + 1 < n_dof:
        K[i, i] += stiffnesses[i + 1]
        K[i, i + 1] -= stiffnesses[i + 1]
        K[i + 1, i] -= stiffnesses[i + 1]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANALYTICAL EIGENVALUE PROBLEM  ->  ground truth
# ═══════════════════════════════════════════════════════════════════════════════
eigenvalues, eigenvectors = linalg.eigh(K, M)
omega_n = np.sqrt(eigenvalues)                       # rad/s
freq_true = omega_n / (2 * np.pi)                    # Hz
# Normalise mode shapes (max = 1)
mode_shapes_true = eigenvectors.copy()
for i in range(n_dof):
    mode_shapes_true[:, i] /= np.max(np.abs(mode_shapes_true[:, i]))

print("=" * 60)
print("GROUND TRUTH modal parameters")
print("=" * 60)
for i in range(n_dof):
    print(f"  Mode {i+1}: f = {freq_true[i]:.4f} Hz, zeta = {damping_ratios_true[i]:.4f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRUCT RAYLEIGH DAMPING MATRIX  (proportional damping)
# ═══════════════════════════════════════════════════════════════════════════════
omega1, omega2 = omega_n[0], omega_n[1]
zeta1, zeta2 = damping_ratios_true[0], damping_ratios_true[1]

A_ray = np.array([[1 / (2 * omega1), omega1 / 2],
                  [1 / (2 * omega2), omega2 / 2]])
b_ray = np.array([zeta1, zeta2])
alpha_ray, beta_ray = np.linalg.solve(A_ray, b_ray)
C = alpha_ray * M + beta_ray * K

# Effective damping ratios under Rayleigh model
damping_ratios_effective = alpha_ray / (2 * omega_n) + beta_ray * omega_n / 2
print("Effective Rayleigh damping ratios:")
for i in range(n_dof):
    print(f"  Mode {i+1}: zeta_eff = {damping_ratios_effective[i]:.4f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATE AMBIENT VIBRATION RESPONSE  (state-space integration)
# ═══════════════════════════════════════════════════════════════════════════════
fs = 100.0    # Hz
T = 300.0     # seconds - long record for good freq resolution & averaging
dt = 1.0 / fs
n_samples = int(T * fs)
t = np.arange(n_samples) * dt

# White noise force at all DOFs (ambient excitation)
F = np.random.randn(n_samples, n_dof) * 5.0

# State-space formulation
M_inv = np.linalg.inv(M)
A_ss = np.zeros((2 * n_dof, 2 * n_dof))
A_ss[:n_dof, n_dof:] = np.eye(n_dof)
A_ss[n_dof:, :n_dof] = -M_inv @ K
A_ss[n_dof:, n_dof:] = -M_inv @ C

B_ss = np.zeros((2 * n_dof, n_dof))
B_ss[n_dof:, :] = M_inv

# Discrete-time state-space (ZOH)
from scipy.linalg import expm
Ad = expm(A_ss * dt)
Bd = np.linalg.solve(A_ss, (Ad - np.eye(2 * n_dof))) @ B_ss

# Simulate
state = np.zeros(2 * n_dof)
accelerations = np.zeros((n_samples, n_dof))

for i in range(n_samples):
    accelerations[i] = M_inv @ (F[i] - C @ state[n_dof:] - K @ state[:n_dof])
    state = Ad @ state + Bd @ F[i]

# Add measurement noise (SNR ~ 30 dB)
for ch in range(n_dof):
    sig_power = np.var(accelerations[:, ch])
    noise_power = sig_power / (10 ** (30.0 / 10.0))
    accelerations[:, ch] += np.random.randn(n_samples) * np.sqrt(noise_power)

print(f"Simulated {n_samples} samples at {fs} Hz ({T} s)")
print(f"Acceleration RMS per channel: "
      f"{[f'{np.std(accelerations[:,i]):.4f}' for i in range(n_dof)]}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. INVERSE SOLVER -- Frequency Domain Decomposition (FDD)
# ═══════════════════════════════════════════════════════════════════════════════

nfft = 8192       # good frequency resolution: df ~ 0.012 Hz
noverlap = nfft * 3 // 4

# Compute cross-spectral density matrix
freqs_psd = None
Gxx = None

for i in range(n_dof):
    for j in range(n_dof):
        f, Pij = signal.csd(accelerations[:, i], accelerations[:, j],
                            fs=fs, nperseg=nfft, noverlap=noverlap,
                            window='hann')
        if Gxx is None:
            freqs_psd = f
            n_freq = len(f)
            Gxx = np.zeros((n_freq, n_dof, n_dof), dtype=complex)
        Gxx[:, i, j] = Pij

# SVD of spectral matrix at each frequency
sv1 = np.zeros(n_freq)
sv2 = np.zeros(n_freq)
U_all = np.zeros((n_freq, n_dof, n_dof), dtype=complex)

for k in range(n_freq):
    U, s, Vh = np.linalg.svd(Gxx[k])
    sv1[k] = s[0]
    sv2[k] = s[1] if len(s) > 1 else 0
    U_all[k] = U

sv1_db = 10 * np.log10(sv1 + 1e-30)

# -- Intelligent peak picking --
f_max_search = 6.0  # well above highest expected mode
freq_mask = (freqs_psd >= 0.1) & (freqs_psd <= f_max_search)
freq_idx_start = np.argmax(freq_mask)
freq_idx_end = len(freq_mask) - np.argmax(freq_mask[::-1])

sv1_db_restricted = sv1_db[freq_idx_start:freq_idx_end]
freqs_restricted = freqs_psd[freq_idx_start:freq_idx_end]

df = freqs_psd[1] - freqs_psd[0]
min_distance = max(int(0.3 / df), 3)

peak_indices_rel, peak_props = signal.find_peaks(
    sv1_db_restricted,
    distance=min_distance,
    prominence=2.0,
    height=np.max(sv1_db_restricted) - 40,
)

# Map back to absolute indices
peak_indices = peak_indices_rel + freq_idx_start

# Sort by prominence and take top n_dof
if len(peak_indices) > n_dof:
    prominences = peak_props['prominences']
    top_idx = np.argsort(prominences)[-n_dof:]
    peak_indices = np.sort(peak_indices[top_idx])

freq_identified = freqs_psd[peak_indices]
print(f"Identified {len(freq_identified)} peaks at: "
      f"{[f'{f:.3f}' for f in freq_identified]} Hz")

# -- Mode shapes from first singular vector --
mode_shapes_identified = np.zeros((n_dof, len(peak_indices)))
for m, pk in enumerate(peak_indices):
    u1 = U_all[pk, :, 0]
    phi = np.real(u1)
    phi /= np.max(np.abs(phi))
    mode_shapes_identified[:, m] = phi

# -- Enhanced FDD damping estimation --
damping_identified = np.zeros(len(peak_indices))

for m, pk in enumerate(peak_indices):
    fn = freqs_psd[pk]
    phi_peak = U_all[pk, :, 0]

    # Define SDOF bell: region where MAC > 0.80 around the peak
    left_idx = pk
    right_idx = pk
    for idx in range(pk - 1, max(freq_idx_start, pk - 200), -1):
        phi_test = U_all[idx, :, 0]
        mac_test = np.abs(np.dot(np.conj(phi_peak), phi_test))**2 / \
                   (np.dot(np.conj(phi_peak), phi_peak).real *
                    np.dot(np.conj(phi_test), phi_test).real)
        if mac_test < 0.80:
            break
        left_idx = idx

    for idx in range(pk + 1, min(freq_idx_end, pk + 200)):
        phi_test = U_all[idx, :, 0]
        mac_test = np.abs(np.dot(np.conj(phi_peak), phi_test))**2 / \
                   (np.dot(np.conj(phi_peak), phi_peak).real *
                    np.dot(np.conj(phi_test), phi_test).real)
        if mac_test < 0.80:
            break
        right_idx = idx

    # Extract SDOF bell from the first singular value
    bell = np.zeros(n_freq)
    bell[left_idx:right_idx + 1] = sv1[left_idx:right_idx + 1]

    # IFFT -> free-decay (autocorrelation of SDOF response)
    bell_sym = np.concatenate([bell, bell[-2:0:-1]])
    free_decay = np.fft.ifft(bell_sym).real
    free_decay = free_decay[:len(free_decay) // 2]

    # Normalize
    if np.abs(free_decay[0]) > 1e-30:
        free_decay_norm = free_decay / free_decay[0]
    else:
        damping_identified[m] = 0.03
        continue

    # Find zero crossings to estimate damped frequency
    t_decay = np.arange(len(free_decay)) / fs
    crossings = []
    for ci in range(1, min(len(free_decay_norm), 500)):
        if free_decay_norm[ci - 1] * free_decay_norm[ci] < 0:
            t_cross = t_decay[ci - 1] + (0 - free_decay_norm[ci - 1]) / \
                      (free_decay_norm[ci] - free_decay_norm[ci - 1]) * dt
            crossings.append(t_cross)

    if len(crossings) >= 4:
        # Period from consecutive positive-going crossings
        periods = []
        for ci in range(0, len(crossings) - 2, 2):
            periods.append(crossings[ci + 2] - crossings[ci])
        T_d = np.median(periods)
        f_d = 1.0 / T_d if T_d > 0 else fn

        # Logarithmic decrement from envelope peaks
        env_peaks_idx, _ = signal.find_peaks(np.abs(free_decay_norm[:500]))
        if len(env_peaks_idx) >= 2:
            env_vals = np.abs(free_decay_norm[env_peaks_idx])
            log_decs = []
            for ci in range(len(env_vals) - 1):
                if env_vals[ci + 1] > 1e-6 and env_vals[ci] > env_vals[ci + 1]:
                    log_decs.append(np.log(env_vals[ci] / env_vals[ci + 1]))
            if log_decs:
                delta = np.median(log_decs)
                zeta_est = delta / np.sqrt(4 * np.pi**2 + delta**2)
                damping_identified[m] = max(0.001, min(zeta_est, 0.2))
            else:
                damping_identified[m] = 0.03
        else:
            damping_identified[m] = 0.03
    else:
        # Fallback: half-power bandwidth
        peak_val = sv1_db[pk]
        half_power = peak_val - 3.0
        left_hp = pk
        while left_hp > 0 and sv1_db[left_hp] > half_power:
            left_hp -= 1
        right_hp = pk
        while right_hp < n_freq - 1 and sv1_db[right_hp] > half_power:
            right_hp += 1
        f_left = freqs_psd[left_hp]
        f_right = freqs_psd[right_hp]
        bandwidth = f_right - f_left
        damping_identified[m] = bandwidth / (2.0 * fn) if fn > 0 else 0.03

print(f"Identified damping ratios: {[f'{d:.4f}' for d in damping_identified]}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MATCH MODES AND EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
def mac_value(phi_a, phi_b):
    """Modal Assurance Criterion between two mode shape vectors."""
    num = np.abs(np.dot(phi_a, phi_b)) ** 2
    den = np.dot(phi_a, phi_a) * np.dot(phi_b, phi_b)
    return num / den if den > 0 else 0.0

n_identified = len(freq_identified)
matched = []
used_true = set()

for m in range(n_identified):
    best_idx = -1
    best_diff = np.inf
    for t_idx in range(n_dof):
        if t_idx in used_true:
            continue
        diff = abs(freq_identified[m] - freq_true[t_idx])
        if diff < best_diff:
            best_diff = diff
            best_idx = t_idx
    if best_idx >= 0 and best_diff < 1.0:
        matched.append((m, best_idx))
        used_true.add(best_idx)

freq_re_list = []
damping_re_list = []
mac_list = []

print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

for idx, (m_id, m_true) in enumerate(matched):
    f_re = abs(freq_identified[m_id] - freq_true[m_true]) / freq_true[m_true]
    freq_re_list.append(f_re)

    d_true = damping_ratios_effective[m_true]
    d_id = damping_identified[m_id]
    d_re = abs(d_id - d_true) / d_true if d_true > 0 else 0.0
    damping_re_list.append(d_re)

    phi_true = mode_shapes_true[:, m_true]
    phi_id = mode_shapes_identified[:, m_id]
    if np.dot(phi_true, phi_id) < 0:
        phi_id = -phi_id
        mode_shapes_identified[:, m_id] = phi_id

    mac_val = mac_value(phi_true, phi_id)
    mac_list.append(mac_val)

    print(f"  Mode {m_true+1}: f_true={freq_true[m_true]:.4f} Hz, "
          f"f_id={freq_identified[m_id]:.4f} Hz, RE={f_re*100:.2f}%")
    print(f"           zeta_true={d_true:.4f}, zeta_id={d_id:.4f}, RE={d_re*100:.1f}%")
    print(f"           MAC = {mac_val:.4f}")

# Full MAC matrix
mac_matrix_full = np.zeros((n_identified, n_dof))
for i in range(n_identified):
    for j in range(n_dof):
        mac_matrix_full[i, j] = mac_value(mode_shapes_identified[:, i],
                                          mode_shapes_true[:, j])

avg_freq_re = np.mean(freq_re_list) if freq_re_list else 1.0
avg_damping_re = np.mean(damping_re_list) if damping_re_list else 1.0
avg_mac = np.mean(mac_list) if mac_list else 0.0
min_mac = np.min(mac_list) if mac_list else 0.0

print()
print(f"Average frequency RE:  {avg_freq_re*100:.2f}%")
print(f"Average damping RE:    {avg_damping_re*100:.1f}%")
print(f"Average MAC:           {avg_mac:.4f}")
print(f"Min MAC:               {min_mac:.4f}")
print(f"Modes matched:         {len(matched)}/{n_dof}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
metrics = {
    "task": "pyoma2_modal",
    "inverse_problem": "Operational Modal Analysis - modal parameter identification",
    "method": "Frequency Domain Decomposition (FDD) + Enhanced FDD for damping",
    "n_dof": n_dof,
    "n_modes_identified": len(matched),
    "frequencies_true_Hz": freq_true.tolist(),
    "frequencies_identified_Hz": freq_identified.tolist(),
    "frequency_RE_per_mode": [round(x * 100, 2) for x in freq_re_list],
    "avg_frequency_RE_percent": round(avg_freq_re * 100, 2),
    "damping_true": damping_ratios_effective.tolist(),
    "damping_identified": damping_identified.tolist(),
    "damping_RE_per_mode": [round(x * 100, 1) for x in damping_re_list],
    "avg_damping_RE_percent": round(avg_damping_re * 100, 1),
    "MAC_per_mode": [round(x, 4) for x in mac_list],
    "avg_MAC": round(avg_mac, 4),
    "min_MAC": round(min_mac, 4),
}

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

gt_data = {
    "frequencies_Hz": freq_true,
    "damping_ratios": damping_ratios_effective,
    "mode_shapes": mode_shapes_true,
}
np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_data, allow_pickle=True)

recon_data = {
    "frequencies_Hz": freq_identified,
    "damping_ratios": damping_identified,
    "mode_shapes": mode_shapes_identified,
}
np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_data, allow_pickle=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATION  -- 4 subplots
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

# -- (a) Time history --
ax1 = fig.add_subplot(gs[0, 0])
t_plot_end = min(2000, n_samples)
ax1.plot(t[:t_plot_end], accelerations[:t_plot_end, 0],
         linewidth=0.4, color='steelblue')
ax1.set_xlabel('Time [s]', fontsize=11)
ax1.set_ylabel('Acceleration [m/s^2]', fontsize=11)
ax1.set_title('(a) Ambient Vibration - Channel 1', fontsize=12, fontweight='bold')
ax1.annotate(f'fs = {fs:.0f} Hz, T = {T:.0f} s\nSNR = 30 dB',
             xy=(0.98, 0.95), xycoords='axes fraction',
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))
ax1.grid(True, alpha=0.3)

# -- (b) FDD singular value spectrum --
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(freqs_psd, sv1_db, linewidth=0.8, color='navy', label='1st SV')
ax2.plot(freqs_psd, 10 * np.log10(sv2 + 1e-30), linewidth=0.6,
         color='gray', alpha=0.5, label='2nd SV')
for m, pk in enumerate(peak_indices):
    ax2.axvline(freqs_psd[pk], color='red', linestyle='--', alpha=0.6, linewidth=0.8)
    ax2.plot(freqs_psd[pk], sv1_db[pk], 'rv', markersize=8)
    ax2.annotate(f'{freqs_psd[pk]:.2f} Hz',
                 xy=(freqs_psd[pk], sv1_db[pk]),
                 xytext=(5, 10), textcoords='offset points',
                 fontsize=8, color='red', fontweight='bold')
ax2.set_xlabel('Frequency [Hz]', fontsize=11)
ax2.set_ylabel('Singular Value [dB]', fontsize=11)
ax2.set_title('(b) Frequency Domain Decomposition', fontsize=12, fontweight='bold')
ax2.set_xlim(0, f_max_search)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

# -- (c) Mode shape comparison --
ax3 = fig.add_subplot(gs[1, 0])
dof_positions = np.arange(1, n_dof + 1)
colors_plot = plt.cm.tab10(np.linspace(0, 0.5, min(5, len(matched))))

n_plot_modes = min(5, len(matched))
for idx in range(n_plot_modes):
    m_id, m_true = matched[idx]
    phi_true = mode_shapes_true[:, m_true]
    phi_id = mode_shapes_identified[:, m_id]
    color = colors_plot[idx]
    ax3.plot(dof_positions, phi_true, 'o-', color=color, linewidth=2,
             markersize=8, label=f'Mode {m_true+1} GT')
    ax3.plot(dof_positions, phi_id, 's--', color=color, linewidth=2,
             markersize=7, alpha=0.7, label=f'Mode {m_true+1} ID')

ax3.set_xlabel('DOF Number', fontsize=11)
ax3.set_ylabel('Normalised Amplitude', fontsize=11)
ax3.set_title('(c) Mode Shape Comparison (GT vs Identified)',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=7, ncol=2, loc='best')
ax3.set_xticks(dof_positions)
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='k', linewidth=0.5)

# -- (d) MAC matrix --
ax4 = fig.add_subplot(gs[1, 1])
im = ax4.imshow(mac_matrix_full, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
ax4.set_xlabel('True Mode', fontsize=11)
ax4.set_ylabel('Identified Mode', fontsize=11)
ax4.set_title('(d) MAC Matrix', fontsize=12, fontweight='bold')
ax4.set_xticks(range(n_dof))
ax4.set_xticklabels([f'{i+1}' for i in range(n_dof)])
ax4.set_yticks(range(n_identified))
ax4.set_yticklabels([f'{i+1}' for i in range(n_identified)])

for i in range(n_identified):
    for j in range(n_dof):
        val = mac_matrix_full[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=9, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cbar.set_label('MAC', fontsize=10)

fig.suptitle('Operational Modal Analysis - 5-DOF System\n'
             'Inverse: Ambient vibration -> Modal parameters (FDD method)',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(os.path.join(RESULTS_DIR, "reconstruction_result.png"),
            dpi=150, bbox_inches='tight')
plt.close()

print()
print(f"Results saved to {RESULTS_DIR}/")
print("  metrics.json, ground_truth.npy, reconstruction.npy, reconstruction_result.png")
print("DONE")
