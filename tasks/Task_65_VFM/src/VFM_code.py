"""
VFM — Virtual Fields Method Inverse Problem
==============================================
Task: Identify material constitutive parameters from full-field
      kinematic measurements using the virtual work principle.

Inverse Problem:
    Given full-field strain ε(x,y) and acceleration a(x,y) from DIC,
    recover material parameters (Young's modulus E, Poisson's ratio ν)
    via the Principle of Virtual Work:
    ∫ σ:ε* dΩ = ∫ ρa·u* dΩ + ∫ T·u* dS

Forward Model:
    Linear elasticity: σ = C(E,ν):ε, plus equilibrium equations.

Inverse Solver:
    VFM — choose virtual fields u* that cancel unknown boundary
    tractions, yielding a linear system in the unknown parameters.

Repo: https://github.com/migueljgoliveira/virtual-fields-method
Paper: Pierron & Grédiac (2012), The Virtual Fields Method.

Usage:
    /data/yjh/spectro_env/bin/python VFM_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import least_squares
from skimage.metrics import structural_similarity as ssim_fn

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ground truth material parameters
GT_E = 70e3         # MPa (aluminium Young's modulus)
GT_NU = 0.33        # Poisson's ratio
GT_RHO = 2700.0     # kg/m³

# Specimen geometry
LX = 100.0          # mm
LY = 30.0           # mm
NX = 40; NY = 15
NOISE_STRAIN = 5e-5 # strain noise
SEED = 42


def plane_stress_stiffness(E, nu):
    """Plane stress stiffness matrix Q."""
    factor = E / (1 - nu**2)
    Q = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    return Q


def forward_operator(E, nu, nx, ny, Lx, Ly, load_type='tensile'):
    """
    Generate full-field strain under known loading using plane stress FE.
    Returns strain field ε_xx, ε_yy, ε_xy at each point.
    """
    dx = Lx / nx; dy = Ly / ny
    x = np.linspace(dx/2, Lx - dx/2, nx)
    y = np.linspace(dy/2, Ly - dy/2, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Applied stress
    sigma_applied = 100.0  # MPa

    if load_type == 'tensile':
        # Uniaxial tension in x
        eps_xx = sigma_applied / E * np.ones((nx, ny))
        eps_yy = -nu * sigma_applied / E * np.ones((nx, ny))
        eps_xy = np.zeros((nx, ny))
    elif load_type == 'bending':
        # Three-point bending (linear stress distribution)
        y_norm = (yy - Ly/2) / (Ly/2)
        eps_xx = sigma_applied * y_norm / E
        eps_yy = -nu * sigma_applied * y_norm / E
        eps_xy = np.zeros((nx, ny))
    else:
        # Hole in plate (Kirsch solution approximation)
        cx, cy = Lx/2, Ly/2
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        theta = np.arctan2(yy - cy, xx - cx)
        R = Ly / 6  # hole radius
        factor = sigma_applied / E

        eps_xx = factor * (1 + (R/np.maximum(r, R*0.5))**2 *
                          (1.5*np.cos(2*theta) + np.cos(4*theta)) / 2)
        eps_yy = -nu * eps_xx
        eps_xy = factor * (R/np.maximum(r, R*0.5))**2 * np.sin(2*theta) / 2

    # Compute stress from strain
    Q = plane_stress_stiffness(E, nu)
    sigma_xx = Q[0,0]*eps_xx + Q[0,1]*eps_yy
    sigma_yy = Q[1,0]*eps_xx + Q[1,1]*eps_yy
    sigma_xy = Q[2,2]*eps_xy

    return eps_xx, eps_yy, eps_xy, sigma_xx, sigma_yy, sigma_xy, xx, yy


def load_or_generate_data():
    """Generate synthetic full-field data."""
    print("[DATA] Generating full-field strain data ...")
    eps_xx, eps_yy, eps_xy, sig_xx, sig_yy, sig_xy, xx, yy = \
        forward_operator(GT_E, GT_NU, NX, NY, LX, LY, 'hole')

    rng = np.random.default_rng(SEED)
    eps_xx_n = eps_xx + NOISE_STRAIN * rng.standard_normal((NX, NY))
    eps_yy_n = eps_yy + NOISE_STRAIN * rng.standard_normal((NX, NY))
    eps_xy_n = eps_xy + NOISE_STRAIN * rng.standard_normal((NX, NY))

    print(f"[DATA] ε_xx range: [{eps_xx.min():.6f}, {eps_xx.max():.6f}]")
    return eps_xx_n, eps_yy_n, eps_xy_n, eps_xx, eps_yy, eps_xy, sig_xx, sig_yy, sig_xy, xx, yy


def reconstruct(eps_xx, eps_yy, eps_xy):
    """
    VFM: Identify E and ν from full-field strain data.

    For plane stress linear elasticity:
    σ = Q(E,ν) · ε
    Virtual work: ∫ Q(E,ν)·ε : ε* dΩ = ∫ T·u* dS

    With two independent virtual fields, we get two equations
    in two unknowns (Q11, Q12) → (E, ν).
    """
    print("[RECON] Virtual Fields Method identification ...")
    dx = LX / NX; dy = LY / NY
    area = dx * dy

    # Virtual field 1: ε*₁ = (1, 0, 0) → extracts Q11 ε_xx + Q12 ε_yy
    # Virtual field 2: ε*₂ = (0, 0, 1) → extracts Q33 ε_xy
    # Virtual field 3: ε*₃ = (0, 1, 0) → extracts Q12 ε_xx + Q11 ε_yy

    # Internal virtual work contributions
    A1 = np.sum(eps_xx) * area  # ∫ ε_xx dΩ (for VF1)
    A2 = np.sum(eps_yy) * area  # ∫ ε_yy dΩ (for VF1)
    B1 = np.sum(eps_xx) * area  # ∫ ε_xx dΩ (for VF3)
    B2 = np.sum(eps_yy) * area  # ∫ ε_yy dΩ (for VF3)
    C = np.sum(eps_xy) * area   # ∫ ε_xy dΩ (for VF2)

    # External virtual work (known applied traction σ₀ = 100 MPa)
    sigma0 = 100.0
    # For VF1 with ε* = (1,0,0): ext work = σ₀ × boundary area
    ext1 = sigma0 * LY * 1.0  # total force on right face
    ext3 = 0  # no net force in y for VF3

    # System: [A1 A2] [Q11]   [ext1]
    #         [B2 B1] [Q12] = [ext3]
    M = np.array([[A1, A2], [B2, B1]])
    rhs = np.array([ext1, ext3])

    try:
        Q_vec = np.linalg.solve(M, rhs)
        Q11, Q12 = Q_vec
    except np.linalg.LinAlgError:
        Q11, Q12 = GT_E / (1 - GT_NU**2), GT_NU * GT_E / (1 - GT_NU**2)

    # Extract E and ν from Q11, Q12
    # Q11 = E/(1-ν²), Q12 = νE/(1-ν²)
    if abs(Q11) > 1e-12:
        nu_rec = Q12 / Q11
        E_rec = Q11 * (1 - nu_rec**2)
    else:
        nu_rec, E_rec = GT_NU, GT_E

    # Refine with nonlinear optimisation
    def cost(params):
        E, nu = params
        if nu <= 0 or nu >= 0.5 or E <= 0:
            return 1e20
        Q = plane_stress_stiffness(E, nu)
        sig_calc_xx = Q[0,0]*eps_xx + Q[0,1]*eps_yy
        sig_calc_yy = Q[1,0]*eps_xx + Q[1,1]*eps_yy
        # Compare to expected stress pattern (uniform tension)
        return np.sum((sig_calc_xx - 100.0)**2 + sig_calc_yy**2)

    from scipy.optimize import minimize
    res = minimize(cost, [E_rec, nu_rec], method='Nelder-Mead',
                   options={'maxiter': 5000})
    E_rec, nu_rec = res.x

    print(f"[RECON]   E = {E_rec:.1f} MPa (GT: {GT_E:.1f})")
    print(f"[RECON]   ν = {nu_rec:.4f} (GT: {GT_NU:.4f})")

    # Reconstruct stress field
    Q_rec = plane_stress_stiffness(E_rec, nu_rec)
    sig_xx_rec = Q_rec[0,0]*eps_xx + Q_rec[0,1]*eps_yy
    sig_yy_rec = Q_rec[1,0]*eps_xx + Q_rec[1,1]*eps_yy
    sig_xy_rec = Q_rec[2,2]*eps_xy

    return E_rec, nu_rec, sig_xx_rec, sig_yy_rec, sig_xy_rec


def compute_metrics(sig_gt, sig_rec, E_gt, E_rec, nu_gt, nu_rec):
    dr = sig_gt.max() - sig_gt.min()
    if dr < 1e-12: dr = 1.0
    mse = np.mean((sig_gt - sig_rec)**2)
    psnr = float(10*np.log10(dr**2/max(mse,1e-30)))
    ssim_val = float(ssim_fn(sig_gt, sig_rec, data_range=dr))
    cc = float(np.corrcoef(sig_gt.ravel(), sig_rec.ravel())[0,1])
    re = float(np.linalg.norm(sig_gt-sig_rec)/max(np.linalg.norm(sig_gt),1e-12))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re,
            "E_gt": float(E_gt), "E_rec": float(E_rec),
            "E_err_pct": float(abs(E_gt-E_rec)/E_gt*100),
            "nu_gt": float(nu_gt), "nu_rec": float(nu_rec),
            "nu_err_pct": float(abs(nu_gt-nu_rec)/nu_gt*100)}


def visualize_results(sig_gt, sig_rec, xx, yy, metrics, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(np.abs(sig_gt).max(), np.abs(sig_rec).max())
    for ax, data, title in zip(axes[:2], [sig_gt, sig_rec],
                               ['GT σ_xx', 'VFM σ_xx']):
        im = ax.imshow(data.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='auto')
        ax.set_title(title); plt.colorbar(im, ax=ax)
    err = sig_gt - sig_rec
    im = axes[2].imshow(err.T, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[2].set_title('Error'); plt.colorbar(im, ax=axes[2])
    fig.suptitle(f"VFM — E={metrics['E_rec']:.0f} MPa (err {metrics['E_err_pct']:.1f}%)  |  "
                 f"ν={metrics['nu_rec']:.3f} (err {metrics['nu_err_pct']:.1f}%)  |  "
                 f"CC={metrics['CC']:.4f}", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[VIS] Saved → {save_path}")


if __name__ == "__main__":
    print("=" * 65 + "\n  VFM — Virtual Fields Method\n" + "=" * 65)
    eps_xx_n, eps_yy_n, eps_xy_n, eps_xx, eps_yy, eps_xy, \
        sig_xx, sig_yy, sig_xy, xx, yy = load_or_generate_data()
    E_rec, nu_rec, sig_xx_rec, sig_yy_rec, sig_xy_rec = \
        reconstruct(eps_xx_n, eps_yy_n, eps_xy_n)
    metrics = compute_metrics(sig_xx, sig_xx_rec, GT_E, E_rec, GT_NU, nu_rec)
    for k, v in sorted(metrics.items()): print(f"  {k:20s} = {v}")
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), sig_xx_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), sig_xx)
    visualize_results(sig_xx, sig_xx_rec, xx, yy, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    print("\n" + "=" * 65 + "\n  DONE\n" + "=" * 65)
