"""
Task 157: plasmapy_langmuir — Langmuir Probe I-V Curve Inversion

Inverse Problem: Extract plasma parameters (T_e, n_e, V_p, V_f) from a
Langmuir probe I-V characteristic curve.

Physics:
  The current collected by a Langmuir probe immersed in a plasma depends on
  the applied bias voltage V:
    - V << V_p  (ion saturation):  I ≈ I_ion_sat  (negative, constant)
    - V < V_p   (transition):      I = I_ion_sat + I_e_sat * exp(e(V-V_p)/(k_B T_e))
    - V >= V_p  (electron sat.):   I = I_ion_sat + I_e_sat

  where I_e_sat = n_e * e * A_probe * sqrt(k_B T_e / (2π m_e))

Forward model parameters:
  T_e      — electron temperature  [eV]
  n_e      — electron density      [m⁻³]
  V_p      — plasma potential      [V]
  I_ion_sat — ion saturation current [A]

Inversion: nonlinear least-squares curve_fit on the I-V data.
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Physical constants ──────────────────────────────────────────────
E_CHARGE = 1.602176634e-19   # C
M_ELECTRON = 9.1093837015e-31  # kg
K_BOLTZMANN = 1.380649e-23   # J/K
EV_TO_K = E_CHARGE / K_BOLTZMANN  # 1 eV ≈ 11604.5 K
A_PROBE = 1.0e-6             # probe collecting area [m²]

# ── Forward model ───────────────────────────────────────────────────

def electron_saturation_current(T_e_eV, n_e):
    """Electron saturation current [A] for given T_e (eV) and n_e (m⁻³)."""
    T_e_K = T_e_eV * EV_TO_K
    return n_e * E_CHARGE * A_PROBE * np.sqrt(K_BOLTZMANN * T_e_K / (2 * np.pi * M_ELECTRON))


def langmuir_iv(V, T_e, n_e, V_p, I_ion_sat):
    """
    Theoretical Langmuir probe I-V characteristic.

    Parameters
    ----------
    V : array_like   — bias voltage [V]
    T_e : float      — electron temperature [eV]
    n_e : float      — electron density [m⁻³]
    V_p : float      — plasma potential [V]
    I_ion_sat : float — ion saturation current [A] (negative)

    Returns
    -------
    I : ndarray — probe current [A]
    """
    T_e_K = T_e * EV_TO_K
    I_e_sat = electron_saturation_current(T_e, n_e)

    # Clamp the exponent to avoid overflow
    exponent = E_CHARGE * (V - V_p) / (K_BOLTZMANN * T_e_K)
    exponent = np.clip(exponent, -500, 500)

    I = np.where(
        V < V_p,
        I_ion_sat + I_e_sat * np.exp(exponent),
        I_ion_sat + I_e_sat,
    )
    return I


def floating_potential(T_e_eV, n_e, V_p, I_ion_sat):
    """Compute floating potential V_f where I(V_f) = 0."""
    I_e_sat = electron_saturation_current(T_e_eV, n_e)
    if I_e_sat <= 0 or -I_ion_sat <= 0:
        return V_p  # degenerate
    T_e_K = T_e_eV * EV_TO_K
    V_f = V_p + (K_BOLTZMANN * T_e_K / E_CHARGE) * np.log(-I_ion_sat / I_e_sat)
    return V_f


# ── Synthesis ───────────────────────────────────────────────────────

def synthesize_iv(T_e, n_e, V_p, I_ion_sat, V_range=(-30, 30), n_points=500,
                  noise_level=0.02, seed=42):
    """Generate noisy Langmuir probe I-V data from known parameters."""
    rng = np.random.default_rng(seed)
    V = np.linspace(V_range[0], V_range[1], n_points)
    I_clean = langmuir_iv(V, T_e, n_e, V_p, I_ion_sat)
    noise_amplitude = noise_level * (np.max(I_clean) - np.min(I_clean))
    noise = rng.normal(0, noise_amplitude, size=I_clean.shape)
    I_noisy = I_clean + noise
    return V, I_clean, I_noisy


# ── Inversion ───────────────────────────────────────────────────────

def invert_iv(V, I_noisy, p0=None):
    """
    Fit the Langmuir I-V model to noisy data.

    Returns
    -------
    params : dict with keys T_e, n_e, V_p, I_ion_sat, V_f
    """
    if p0 is None:
        # Heuristic initial guesses
        I_min = np.min(I_noisy)
        I_max = np.max(I_noisy)
        V_p_guess = V[np.argmax(np.gradient(I_noisy, V))]
        T_e_guess = 5.0
        n_e_guess = 1e17
        I_ion_sat_guess = I_min
        p0 = [T_e_guess, n_e_guess, V_p_guess, I_ion_sat_guess]

    bounds = (
        [0.1, 1e14, V.min(), -1.0],    # lower
        [100.0, 1e20, V.max(), 0.0],     # upper
    )

    popt, pcov = curve_fit(
        langmuir_iv, V, I_noisy,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
        method="trf",
    )

    T_e_fit, n_e_fit, V_p_fit, I_ion_sat_fit = popt
    V_f_fit = floating_potential(T_e_fit, n_e_fit, V_p_fit, I_ion_sat_fit)
    perr = np.sqrt(np.diag(pcov))

    return {
        "T_e": T_e_fit,
        "n_e": n_e_fit,
        "V_p": V_p_fit,
        "I_ion_sat": I_ion_sat_fit,
        "V_f": V_f_fit,
        "std_errors": {
            "T_e": perr[0],
            "n_e": perr[1],
            "V_p": perr[2],
            "I_ion_sat": perr[3],
        },
    }


# ── Evaluation ──────────────────────────────────────────────────────

def relative_error(true_val, est_val):
    return abs(est_val - true_val) / abs(true_val) if true_val != 0 else 0.0


# ── Test cases ──────────────────────────────────────────────────────

TEST_CASES = [
    {"T_e": 5.0,  "n_e": 1e17, "V_p": 10.0, "I_ion_sat": -0.005, "label": "Baseline"},
    {"T_e": 2.0,  "n_e": 5e16, "V_p": 5.0,  "I_ion_sat": -0.002, "label": "Low-T_e"},
    {"T_e": 10.0, "n_e": 5e17, "V_p": 15.0, "I_ion_sat": -0.01,  "label": "High-T_e"},
    {"T_e": 3.0,  "n_e": 1e18, "V_p": 8.0,  "I_ion_sat": -0.008, "label": "High-n_e"},
    {"T_e": 8.0,  "n_e": 2e16, "V_p": 12.0, "I_ion_sat": -0.003, "label": "Low-n_e"},
]


def run_all_cases():
    """Run forward + inverse on every test case, return aggregated results."""
    all_results = []
    for i, tc in enumerate(TEST_CASES):
        V, I_clean, I_noisy = synthesize_iv(
            tc["T_e"], tc["n_e"], tc["V_p"], tc["I_ion_sat"],
            noise_level=0.02, seed=42 + i,
        )
        fitted = invert_iv(V, I_noisy)

        V_f_true = floating_potential(tc["T_e"], tc["n_e"], tc["V_p"], tc["I_ion_sat"])
        I_fitted = langmuir_iv(V, fitted["T_e"], fitted["n_e"], fitted["V_p"], fitted["I_ion_sat"])

        re = {
            "T_e": relative_error(tc["T_e"], fitted["T_e"]),
            "n_e": relative_error(tc["n_e"], fitted["n_e"]),
            "V_p": relative_error(tc["V_p"], fitted["V_p"]),
            "I_ion_sat": relative_error(tc["I_ion_sat"], fitted["I_ion_sat"]),
            "V_f": relative_error(V_f_true, fitted["V_f"]),
        }

        all_results.append({
            "case": tc["label"],
            "true_params": tc,
            "fitted_params": fitted,
            "V_f_true": V_f_true,
            "relative_errors": re,
            "V": V,
            "I_clean": I_clean,
            "I_noisy": I_noisy,
            "I_fitted": I_fitted,
        })
    return all_results


# ── Visualization ───────────────────────────────────────────────────

def plot_results(results, save_path):
    """4-panel visualization for the baseline case + summary bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 157: Langmuir Probe I-V Curve Inversion", fontsize=14, fontweight="bold")

    # Use baseline case (index 0) for panels 1-3
    r = results[0]
    V = r["V"]
    I_noisy = r["I_noisy"]
    I_clean = r["I_clean"]
    I_fitted = r["I_fitted"]

    # Panel 1: Noisy I-V data
    ax = axes[0, 0]
    ax.scatter(V, I_noisy * 1e3, s=2, alpha=0.5, color="steelblue", label="Noisy data")
    ax.set_xlabel("Bias Voltage V [V]")
    ax.set_ylabel("Current I [mA]")
    ax.set_title("(a) Noisy I-V Data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: True vs Fitted curves
    ax = axes[0, 1]
    ax.plot(V, I_clean * 1e3, "k-", lw=2, label="Ground truth")
    ax.plot(V, I_fitted * 1e3, "r--", lw=2, label="Fitted")
    ax.scatter(V, I_noisy * 1e3, s=1, alpha=0.3, color="gray", label="Noisy data")
    ax.set_xlabel("Bias Voltage V [V]")
    ax.set_ylabel("Current I [mA]")
    ax.set_title("(b) True vs Fitted I-V Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Residuals
    ax = axes[1, 0]
    residuals = (I_fitted - I_clean) * 1e3
    ax.plot(V, residuals, "g-", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Bias Voltage V [V]")
    ax.set_ylabel("Residual [mA]")
    ax.set_title("(c) Fit Residuals (Fitted − True)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Parameter comparison bar chart (all cases)
    ax = axes[1, 1]
    param_names = ["T_e", "n_e", "V_p", "I_ion_sat"]
    mean_re = {}
    for pn in param_names:
        vals = [r2["relative_errors"][pn] * 100 for r2 in results]
        mean_re[pn] = np.mean(vals)
    x = np.arange(len(param_names))
    colors = ["#2196F3" if v < 5 else "#FF9800" if v < 10 else "#F44336" for v in mean_re.values()]
    bars = ax.bar(x, list(mean_re.values()), color=colors, edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["$T_e$", "$n_e$", "$V_p$", "$I_{ion,sat}$"])
    ax.set_ylabel("Mean Relative Error [%]")
    ax.set_title(f"(d) Mean RE Across {len(results)} Cases")
    ax.axhline(10, color="red", ls="--", lw=1, label="10% threshold")
    for bar, val in zip(bars, mean_re.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot → {save_path}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Task 157: Langmuir Probe I-V Curve Inversion")
    print("=" * 60)

    results = run_all_cases()

    # ── Metrics ──
    metrics = {"test_cases": []}
    all_re = {"T_e": [], "n_e": [], "V_p": [], "I_ion_sat": [], "V_f": []}

    for r in results:
        case_metric = {
            "case": r["case"],
            "true": {
                "T_e": r["true_params"]["T_e"],
                "n_e": r["true_params"]["n_e"],
                "V_p": r["true_params"]["V_p"],
                "I_ion_sat": r["true_params"]["I_ion_sat"],
                "V_f": r["V_f_true"],
            },
            "estimated": {
                "T_e": r["fitted_params"]["T_e"],
                "n_e": r["fitted_params"]["n_e"],
                "V_p": r["fitted_params"]["V_p"],
                "I_ion_sat": r["fitted_params"]["I_ion_sat"],
                "V_f": r["fitted_params"]["V_f"],
            },
            "relative_errors": {k: f"{v*100:.4f}%" for k, v in r["relative_errors"].items()},
        }
        metrics["test_cases"].append(case_metric)

        for k in all_re:
            all_re[k].append(r["relative_errors"][k])

        print(f"\n--- Case: {r['case']} ---")
        print(f"  T_e : true={r['true_params']['T_e']:.2f} eV,  est={r['fitted_params']['T_e']:.4f} eV,  RE={r['relative_errors']['T_e']*100:.4f}%")
        print(f"  n_e : true={r['true_params']['n_e']:.2e} m⁻³, est={r['fitted_params']['n_e']:.4e} m⁻³, RE={r['relative_errors']['n_e']*100:.4f}%")
        print(f"  V_p : true={r['true_params']['V_p']:.2f} V,   est={r['fitted_params']['V_p']:.4f} V,   RE={r['relative_errors']['V_p']*100:.4f}%")
        print(f"  V_f : true={r['V_f_true']:.4f} V,   est={r['fitted_params']['V_f']:.4f} V,   RE={r['relative_errors']['V_f']*100:.4f}%")

    mean_re = {k: np.mean(v) * 100 for k, v in all_re.items()}
    metrics["mean_relative_errors"] = {k: f"{v:.4f}%" for k, v in mean_re.items()}
    metrics["overall_mean_RE"] = f"{np.mean(list(mean_re.values())):.4f}%"

    print(f"\n{'='*60}")
    print("Mean Relative Errors across all cases:")
    for k, v in mean_re.items():
        status = "✓ PASS" if v < 10 else "✗ FAIL"
        print(f"  {k:12s}: {v:.4f}%  {status}")
    overall = np.mean(list(mean_re.values()))
    print(f"  {'Overall':12s}: {overall:.4f}%  {'✓ PASS' if overall < 10 else '✗ FAIL'}")

    # ── Save outputs ──
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[INFO] Saved metrics → {metrics_path}")

    # Ground truth: dict of all true parameters for baseline case
    gt = {
        "T_e": results[0]["true_params"]["T_e"],
        "n_e": results[0]["true_params"]["n_e"],
        "V_p": results[0]["true_params"]["V_p"],
        "I_ion_sat": results[0]["true_params"]["I_ion_sat"],
        "V_f": results[0]["V_f_true"],
        "I_V_data": {
            "V": results[0]["V"].tolist(),
            "I_clean": results[0]["I_clean"].tolist(),
            "I_noisy": results[0]["I_noisy"].tolist(),
        },
    }
    gt_path = os.path.join(out_dir, "ground_truth.npy")
    np.save(gt_path, gt, allow_pickle=True)
    print(f"[INFO] Saved ground truth → {gt_path}")

    # Reconstruction: dict of fitted parameters for baseline case
    recon = {
        "T_e": results[0]["fitted_params"]["T_e"],
        "n_e": results[0]["fitted_params"]["n_e"],
        "V_p": results[0]["fitted_params"]["V_p"],
        "I_ion_sat": results[0]["fitted_params"]["I_ion_sat"],
        "V_f": results[0]["fitted_params"]["V_f"],
        "I_V_data": {
            "V": results[0]["V"].tolist(),
            "I_fitted": results[0]["I_fitted"].tolist(),
        },
    }
    recon_path = os.path.join(out_dir, "reconstruction.npy")
    np.save(recon_path, recon, allow_pickle=True)
    print(f"[INFO] Saved reconstruction → {recon_path}")

    # Visualization
    plot_path = os.path.join(out_dir, "reconstruction_result.png")
    plot_results(results, plot_path)

    print(f"\n{'='*60}")
    print("Task 157 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
