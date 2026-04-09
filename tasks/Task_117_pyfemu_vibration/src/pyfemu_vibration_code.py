"""
pyfemu_vibration — Vibration-Based Damage Identification
==========================================================
From measured modal parameters (natural frequencies, mode shapes) of a beam,
identify damage locations and severities by updating a 1-D Euler-Bernoulli
beam FEM model.

Physics:
  Forward:
    - N-element Euler-Bernoulli beam  →  global K, M matrices
    - Damage in element i  →  K_i × (1 − d_i),  d_i ∈ [0, 1]
    - Eigenvalue problem:  (K − ω² M) φ = 0

  Inverse:
    - Minimise  Σ (ω_obs − ω_model)² + λ Σ (1 − MAC)
    - L-BFGS-B with bounds  d_i ∈ [0, 0.95]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import minimize
from scipy.linalg import eigh

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_117_pyfemu_vibration"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── beam properties ───────────────────────────────────────────────
N_ELEM     = 20        # number of elements
L_TOTAL    = 1.0       # total length (m)
E_BEAM     = 210e9     # Young's modulus (Pa) — steel
RHO_BEAM   = 7800.0    # density (kg/m³)
B_SECT     = 0.02      # width (m)
H_SECT     = 0.02      # height (m)
AREA       = B_SECT * H_SECT
I_SECT     = B_SECT * H_SECT ** 3 / 12.0   # second moment of area

N_MODES    = 5         # number of modes to use

# ── damage ground truth ───────────────────────────────────────────
# d_i = fractional stiffness reduction in element i
GT_DAMAGE  = {4: 0.30, 11: 0.50, 16: 0.20}   # elements 5, 12, 17 (0-indexed)

# ── noise / optimisation ──────────────────────────────────────────
FREQ_NOISE = 0.005     # 0.5 % noise on frequencies
MODE_NOISE = 0.02      # 2 % noise on mode shapes
SEED       = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1.  FEM ASSEMBLY
# ═══════════════════════════════════════════════════════════════════
def euler_bernoulli_element(L_e, EI, rhoA):
    """
    4×4 element stiffness and consistent mass matrices for an
    Euler-Bernoulli beam element of length L_e.
    """
    k_e = (EI / L_e ** 3) * np.array([
        [ 12,    6*L_e,  -12,    6*L_e],
        [  6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12,   -6*L_e,   12,   -6*L_e],
        [  6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ])

    m_e = (rhoA * L_e / 420.0) * np.array([
        [156,    22*L_e,   54,   -13*L_e],
        [ 22*L_e,  4*L_e**2,  13*L_e, -3*L_e**2],
        [ 54,    13*L_e,  156,   -22*L_e],
        [-13*L_e, -3*L_e**2, -22*L_e,  4*L_e**2]
    ])
    return k_e, m_e


def assemble(n_elem, L_total, EI, rhoA, damage_vec):
    """
    Assemble global K and M for a simply-supported beam.
    damage_vec[i] ∈ [0, 1] reduces stiffness of element i.
    """
    n_dof = 2 * (n_elem + 1)   # 2 DOF per node (w, θ)
    L_e   = L_total / n_elem
    K_global = np.zeros((n_dof, n_dof))
    M_global = np.zeros((n_dof, n_dof))

    for i in range(n_elem):
        EI_eff = EI * (1.0 - damage_vec[i])
        k_e, m_e = euler_bernoulli_element(L_e, EI_eff, rhoA)
        idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
        for a in range(4):
            for b in range(4):
                K_global[idx[a], idx[b]] += k_e[a, b]
                M_global[idx[a], idx[b]] += m_e[a, b]

    # Simply-supported BC: w=0 at node 0 and node N
    bc_dofs = [0, 2 * n_elem]   # translational DOFs at ends
    free = [d for d in range(n_dof) if d not in bc_dofs]
    K_free = K_global[np.ix_(free, free)]
    M_free = M_global[np.ix_(free, free)]
    return K_free, M_free, free


def solve_modal(n_elem, L_total, EI, rhoA, damage_vec, n_modes):
    """Solve eigenvalue problem → natural frequencies + mode shapes."""
    K, M, free_dofs = assemble(n_elem, L_total, EI, rhoA, damage_vec)
    eigvals, eigvecs = eigh(K, M)
    # Take first n_modes
    omega2 = eigvals[:n_modes]
    freqs  = np.sqrt(np.abs(omega2)) / (2 * np.pi)
    modes  = eigvecs[:, :n_modes]
    # Normalise mode shapes
    for j in range(n_modes):
        modes[:, j] /= np.max(np.abs(modes[:, j])) + 1e-30
    return freqs, modes


# ═══════════════════════════════════════════════════════════════════
# 2.  MAC (Modal Assurance Criterion)
# ═══════════════════════════════════════════════════════════════════
def mac_value(phi_a, phi_b):
    """MAC between two mode shape vectors."""
    num = np.dot(phi_a, phi_b) ** 2
    den = np.dot(phi_a, phi_a) * np.dot(phi_b, phi_b)
    return num / (den + 1e-30)


# ═══════════════════════════════════════════════════════════════════
# 3.  INVERSE OBJECTIVE
# ═══════════════════════════════════════════════════════════════════
def objective(d_vec, n_elem, L_total, EI, rhoA, n_modes, freqs_obs, modes_obs):
    """
    Misfit = frequency error + mode-shape MAC error.
    """
    freqs_mod, modes_mod = solve_modal(n_elem, L_total, EI, rhoA, d_vec, n_modes)

    # Frequency part
    freq_err = np.sum(((freqs_obs - freqs_mod) / (freqs_obs + 1e-30)) ** 2)

    # MAC part
    mac_err = 0.0
    for j in range(n_modes):
        # Ensure consistent sign
        if np.dot(modes_obs[:, j], modes_mod[:, j]) < 0:
            modes_mod[:, j] *= -1
        mac_err += (1.0 - mac_value(modes_obs[:, j], modes_mod[:, j]))

    return freq_err + 0.5 * mac_err


# ═══════════════════════════════════════════════════════════════════
# 4.  METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(d_gt, d_recon, freqs_gt, freqs_recon, modes_gt, modes_recon, n_modes):
    """Damage vector metrics + frequency + MAC metrics."""
    # PSNR on damage vector
    mse = np.mean((d_gt - d_recon) ** 2)
    data_range = max(np.max(d_gt) - np.min(d_gt), 0.01)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))

    cc = float(np.corrcoef(d_gt, d_recon)[0, 1]) if np.std(d_gt) > 1e-10 else 0.0
    rmse = float(np.sqrt(mse))

    # Frequency RMSE
    freq_rmse = float(np.sqrt(np.mean((freqs_gt - freqs_recon) ** 2)))

    # Average MAC
    mac_vals = []
    for j in range(n_modes):
        if np.dot(modes_gt[:, j], modes_recon[:, j]) < 0:
            modes_recon[:, j] *= -1
        mac_vals.append(mac_value(modes_gt[:, j], modes_recon[:, j]))
    avg_mac = float(np.mean(mac_vals))

    # Damage localisation accuracy
    gt_damaged   = set(np.where(d_gt > 0.05)[0])
    recon_damaged = set(np.where(d_recon > 0.05)[0])
    if len(gt_damaged) > 0:
        detection_rate = len(gt_damaged & recon_damaged) / len(gt_damaged) * 100
    else:
        detection_rate = 100.0

    return {
        "PSNR": float(psnr),
        "CC": cc,
        "RMSE": rmse,
        "freq_RMSE_Hz": freq_rmse,
        "avg_MAC": avg_mac,
        "damage_detection_pct": detection_rate
    }


# ═══════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════
def visualize(d_gt, d_recon, freqs_gt, freqs_obs, freqs_recon,
              modes_gt, modes_recon, n_modes, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    elem_centers = np.arange(N_ELEM) + 0.5

    # (a) Damage distribution
    ax = axes[0, 0]
    ax.bar(elem_centers - 0.2, d_gt, 0.4, label="True Damage", color="steelblue", alpha=0.8)
    ax.bar(elem_centers + 0.2, d_recon, 0.4, label="Identified Damage", color="salmon", alpha=0.8)
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Damage Parameter d")
    ax.set_title(f"Damage Identification  (PSNR={metrics['PSNR']:.1f} dB, CC={metrics['CC']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, N_ELEM)

    # (b) Frequency comparison
    ax = axes[0, 1]
    mode_idx = np.arange(1, n_modes + 1)
    ax.plot(mode_idx, freqs_gt, "bo-", label="GT Frequencies")
    ax.plot(mode_idx, freqs_obs, "g^--", label="Observed (noisy)", alpha=0.7)
    ax.plot(mode_idx, freqs_recon, "rs--", label="Identified Model")
    ax.set_xlabel("Mode Number")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Modal Frequencies  (freq RMSE={metrics['freq_RMSE_Hz']:.2f} Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Mode shapes (first 3)
    ax = axes[1, 0]
    n_dof = modes_gt.shape[0]
    x_nodes = np.linspace(0, L_TOTAL, n_dof)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for j in range(min(3, n_modes)):
        mg = modes_gt[:, j]
        mr = modes_recon[:, j]
        if np.dot(mg, mr) < 0:
            mr = -mr
        ax.plot(x_nodes, mg, "-", color=colors[j], lw=2,
                label=f"Mode {j+1} GT")
        ax.plot(x_nodes, mr, "--", color=colors[j], lw=2,
                label=f"Mode {j+1} Identified")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Mode Shape Amplitude")
    ax.set_title(f"Mode Shapes  (avg MAC={metrics['avg_MAC']:.4f})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (d) Residual / damage error
    ax = axes[1, 1]
    residual = d_gt - d_recon
    ax.bar(elem_centers, residual, 0.6, color="purple", alpha=0.6)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Damage Error (GT − Identified)")
    ax.set_title(f"Damage Residual  (RMSE={metrics['RMSE']:.4f}, Detection={metrics['damage_detection_pct']:.0f}%)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, N_ELEM)

    plt.suptitle("Vibration-Based Damage Identification — FE Model Updating", fontsize=14, y=1.01)
    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR,  "reconstruction_result.png"),
              os.path.join(ASSETS_DIR,  "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("pyfemu_vibration — Vibration-Based Damage Identification")
    print("=" * 60)

    EI   = E_BEAM * I_SECT
    rhoA = RHO_BEAM * AREA

    # 1. Ground-truth damage vector
    d_gt = np.zeros(N_ELEM)
    for elem, severity in GT_DAMAGE.items():
        d_gt[elem] = severity
    print(f"  GT damage: {GT_DAMAGE}")

    # 2. Undamaged modal data (for reference)
    d_zero = np.zeros(N_ELEM)
    freqs_undamaged, _ = solve_modal(N_ELEM, L_TOTAL, EI, rhoA, d_zero, N_MODES)
    print(f"  Undamaged frequencies: {np.round(freqs_undamaged, 2)} Hz")

    # 3. Forward: damaged modal data
    print("[1/4] Computing damaged modal data ...")
    freqs_gt, modes_gt = solve_modal(N_ELEM, L_TOTAL, EI, rhoA, d_gt, N_MODES)
    print(f"  Damaged frequencies:   {np.round(freqs_gt, 2)} Hz")

    # 4. Add noise to observations
    freq_noise_abs = FREQ_NOISE * freqs_gt * np.random.randn(N_MODES)
    freqs_obs = freqs_gt + freq_noise_abs

    mode_noise = MODE_NOISE * np.random.randn(*modes_gt.shape)
    modes_obs = modes_gt + mode_noise
    for j in range(N_MODES):
        modes_obs[:, j] /= np.max(np.abs(modes_obs[:, j])) + 1e-30

    # 5. Inverse: optimise damage parameters
    print("[2/4] Running L-BFGS-B damage identification ...")
    d0 = np.zeros(N_ELEM)  # start undamaged
    bounds = [(0.0, 0.95)] * N_ELEM
    result = minimize(
        objective, d0,
        args=(N_ELEM, L_TOTAL, EI, rhoA, N_MODES, freqs_obs, modes_obs),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10}
    )
    d_recon = result.x
    print(f"  Optimisation converged: {result.success}  (nit={result.nit})")
    print(f"  Identified damage >0.05: ", end="")
    for i in range(N_ELEM):
        if d_recon[i] > 0.05:
            print(f"elem {i}={d_recon[i]:.3f}  ", end="")
    print()

    # 6. Recompute modal data with identified damage
    freqs_recon, modes_recon = solve_modal(N_ELEM, L_TOTAL, EI, rhoA, d_recon, N_MODES)

    # 7. Metrics
    print("[3/4] Computing metrics ...")
    metrics = compute_metrics(d_gt, d_recon, freqs_gt, freqs_recon, modes_gt, modes_recon, N_MODES)
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.6f}")
    print(f"  Freq RMSE = {metrics['freq_RMSE_Hz']:.4f} Hz")
    print(f"  Avg MAC   = {metrics['avg_MAC']:.6f}")
    print(f"  Detection = {metrics['damage_detection_pct']:.0f}%")

    # 8. Save
    print("[4/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), d_gt)
        np.save(os.path.join(d, "recon_output.npy"), d_recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    visualize(d_gt, d_recon, freqs_gt, freqs_obs, freqs_recon,
              modes_gt, modes_recon, N_MODES, metrics)

    print("Done ✓")
    return metrics


if __name__ == "__main__":
    main()
