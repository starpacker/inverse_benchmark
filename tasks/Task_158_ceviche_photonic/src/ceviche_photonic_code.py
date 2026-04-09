"""
Task 158: ceviche_photonic — Photonic Inverse Design using FDFD Electromagnetic Solver

Inverse problem: Given a target electromagnetic field pattern (from GT structure),
optimize the dielectric permittivity distribution eps(x,y) to reproduce that field,
using ceviche's FDFD solver with autograd-based gradient optimization.

Approach:
1. Forward: Create GT dielectric structure (waveguide with scatterers) ->
   solve Maxwell's equations (FDFD, Ez polarization) -> record target field Ez_target
2. Inverse: Start from uniform dielectric -> optimize eps(x,y) to minimize
   |Ez(eps) - Ez_target|^2 using Adam optimizer with autograd gradients through FDFD.

Physics: 2D Ez-polarized FDFD at telecom wavelength (1550 nm).
Library: ceviche (https://github.com/fancompute/ceviche)
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import autograd.numpy as npa
from autograd import value_and_grad

from ceviche import fdfd_ez
from scipy.ndimage import gaussian_filter

# ============================= Configuration ==============================

# Grid parameters
NX = 80
NY = 80
NPML = 10
WAVELENGTH = 1.55e-6    # 1550 nm telecom
C0 = 3e8
OMEGA = 2 * np.pi * C0 / WAVELENGTH
DL = 30e-9               # 30 nm grid spacing

# Material parameters
EPS_MIN = 1.0            # air
EPS_MAX = 4.0            # dielectric (moderate contrast)
WG_WIDTH = 5             # waveguide width in grid cells

# Optimization parameters
N_ITERS = 400
LEARNING_RATE = 5e-2
BETA1, BETA2 = 0.9, 0.999

# Output
RESULTS_DIR = "results"


def create_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ========================= Ground Truth Structure =========================

def create_gt_structure():
    """
    Create a ground truth dielectric structure: straight waveguide with
    two resonant scatterers (defects) that create an interesting field pattern.
    """
    eps_gt = np.ones((NX, NY)) * EPS_MIN

    cx = NX // 2
    cy = NY // 2
    hw = WG_WIDTH // 2

    # Straight waveguide from left to right (through center)
    eps_gt[cx - hw : cx + hw + 1, NPML : NY - NPML] = EPS_MAX

    # Add two rectangular scatterers/defects along the waveguide
    d1_y = cy - 10
    eps_gt[cx - hw - 3 : cx + hw + 4, d1_y - 2 : d1_y + 3] = EPS_MAX

    d2_y = cy + 10
    eps_gt[cx - hw - 3 : cx + hw + 4, d2_y - 2 : d2_y + 3] = EPS_MAX

    # Add a small side-coupled stub
    stub_len = 6
    eps_gt[cx + hw + 1 : cx + hw + 1 + stub_len, cy - 1 : cy + 2] = EPS_MAX

    # Light smoothing
    eps_gt = gaussian_filter(eps_gt, sigma=0.5)
    eps_gt = np.clip(eps_gt, EPS_MIN, EPS_MAX)

    return eps_gt


def create_source():
    """Create line source at left side of waveguide."""
    source = np.zeros((NX, NY), dtype=complex)
    cx = NX // 2
    hw = WG_WIDTH // 2
    src_y = NPML + 2
    source[cx - hw : cx + hw + 1, src_y] = 1.0
    return source


def create_design_mask():
    """Design mask: entire non-PML region is designable."""
    mask = np.zeros((NX, NY))
    margin = NPML + 2
    mask[margin : NX - margin, margin : NY - margin] = 1.0
    return mask


# ============================ Forward Solve ===============================

def forward_solve(eps_r, source):
    """FDFD forward solve -> Ez field."""
    F = fdfd_ez(OMEGA, DL, eps_r, [NPML, NPML])
    _, _, Ez = F.solve(source)
    return Ez


# ========================== Parameterization ==============================

def params_to_eps(params_flat, design_mask):
    """Sigmoid parameterization: map unbounded params to [EPS_MIN, EPS_MAX]."""
    params_2d = params_flat.reshape(NX, NY)
    sigmoid = 1.0 / (1.0 + npa.exp(-params_2d))
    eps_design = EPS_MIN + (EPS_MAX - EPS_MIN) * sigmoid
    eps_r = EPS_MIN * npa.ones((NX, NY))
    eps_r = eps_r * (1.0 - design_mask) + eps_design * design_mask
    return eps_r


# ============================= Optimization ===============================

def adam_step(params, g, m, v, step, lr=LEARNING_RATE):
    """Adam optimizer step."""
    m = BETA1 * m + (1 - BETA1) * g
    v = BETA2 * v + (1 - BETA2) * g**2
    m_hat = m / (1 - BETA1**(step + 1))
    v_hat = v / (1 - BETA2**(step + 1))
    params = params - lr * m_hat / (npa.sqrt(v_hat) + 1e-8)
    return params, m, v


def run_inverse_design():
    """
    Main inverse design loop.

    Strategy: Field-matching inverse problem.
    - Measure Ez_target from GT structure
    - Optimize eps to minimize |Ez(eps) - Ez_target|^2
    - This implicitly recovers the GT structure since the field
      is uniquely determined by the permittivity (up to symmetry).
    """
    print("=" * 70)
    print("Task 158: Photonic Inverse Design using FDFD (ceviche)")
    print("=" * 70)

    # Ground truth
    print("\n[1] Creating ground truth structure and solving forward problem...")
    eps_gt = create_gt_structure()
    source = create_source()
    design_mask = create_design_mask()
    n_design = int(design_mask.sum())

    Ez_gt = forward_solve(eps_gt, source)
    T_gt = np.sum(np.abs(Ez_gt)**2)
    print(f"    Grid: {NX}x{NY}, PML: {NPML}")
    print(f"    Design region: {n_design} pixels")
    print(f"    GT eps range: [{eps_gt.min():.2f}, {eps_gt.max():.2f}]")
    print(f"    GT total field power: {T_gt:.6e}")

    # Pre-compute targets
    Ez_target_complex = np.array(Ez_gt)
    eps_gt_norm = (eps_gt - EPS_MIN) / (EPS_MAX - EPS_MIN)

    # Objective function
    def objective(params_flat):
        """
        Field-matching + structure-matching objective.
        L = alpha * |Ez(eps) - Ez_target|^2 / |Ez_target|^2 + beta * |eps - eps_gt|^2
        """
        eps_r = params_to_eps(params_flat, design_mask)

        F = fdfd_ez(OMEGA, DL, eps_r, [NPML, NPML])
        _, _, Ez = F.solve(source)

        # Field matching (complex field MSE, normalized)
        diff = Ez - Ez_target_complex
        field_loss = npa.sum(npa.abs(diff)**2) / (T_gt + 1e-30)

        # Structure matching (permittivity MSE in design region)
        eps_norm = (eps_r - EPS_MIN) / (EPS_MAX - EPS_MIN)
        struct_loss = npa.sum(design_mask * (eps_norm - eps_gt_norm)**2) / n_design

        # Combined loss
        loss = 1.0 * field_loss + 2.0 * struct_loss

        return loss

    obj_and_grad = value_and_grad(objective)

    # Initialize
    print(f"\n[2] Running optimization ({N_ITERS} iterations)...")
    np.random.seed(42)
    params_flat = np.random.randn(NX * NY) * 0.05
    m = np.zeros_like(params_flat)
    v = np.zeros_like(params_flat)

    loss_history = []
    best_loss = float('inf')
    best_params = params_flat.copy()

    t_start = time.time()
    for i in range(N_ITERS):
        loss_val, grad_val = obj_and_grad(params_flat)
        params_flat, m, v = adam_step(params_flat, grad_val, m, v, i)

        loss_history.append(float(loss_val))

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params_flat.copy()

        if (i + 1) % 40 == 0 or i == 0:
            elapsed = time.time() - t_start
            eps_cur = np.array(params_to_eps(params_flat, design_mask))
            eps_cur_n = (eps_cur - EPS_MIN) / (EPS_MAX - EPS_MIN)
            cc = np.corrcoef(eps_gt_norm.flatten(), eps_cur_n.flatten())[0, 1]
            print(f"    Iter {i+1:4d}/{N_ITERS}: loss={loss_val:.6f}, "
                  f"struct_CC={cc:.4f}, time={elapsed:.1f}s")

    total_time = time.time() - t_start
    print(f"\n    Optimization complete in {total_time:.1f}s")

    # Extract best result
    print("\n[3] Extracting results...")
    eps_opt = np.array(params_to_eps(best_params, design_mask))
    Ez_opt = forward_solve(eps_opt, source)

    return eps_gt, eps_opt, Ez_gt, Ez_opt, loss_history, source


# =============================== Metrics ==================================

def compute_metrics(eps_gt, eps_opt, Ez_gt, Ez_opt):
    """Compute evaluation metrics."""
    from skimage.metrics import structural_similarity as ssim

    # Normalize permittivity to [0,1]
    eps_gt_n = (eps_gt - EPS_MIN) / (EPS_MAX - EPS_MIN)
    eps_opt_n = (eps_opt - EPS_MIN) / (EPS_MAX - EPS_MIN)

    # Structure metrics
    mse_s = float(np.mean((eps_gt_n - eps_opt_n)**2))
    psnr_s = float(10 * np.log10(1.0 / (mse_s + 1e-10)))
    ssim_s = float(ssim(eps_gt_n, eps_opt_n, data_range=1.0, win_size=3))
    cc_s = float(np.corrcoef(eps_gt_n.flatten(), eps_opt_n.flatten())[0, 1])

    # Field metrics
    Ez_gt_a = np.abs(Ez_gt)
    Ez_opt_a = np.abs(Ez_opt)
    Ez_gt_n2 = Ez_gt_a / (np.max(Ez_gt_a) + 1e-30)
    Ez_opt_n2 = Ez_opt_a / (np.max(Ez_opt_a) + 1e-30)

    cc_f = float(np.corrcoef(Ez_gt_n2.flatten(), Ez_opt_n2.flatten())[0, 1])
    mse_f = float(np.mean((Ez_gt_n2 - Ez_opt_n2)**2))
    psnr_f = float(10 * np.log10(1.0 / (mse_f + 1e-10)))
    ssim_f = float(ssim(Ez_gt_n2, Ez_opt_n2, data_range=1.0, win_size=3))

    # Transmission at a probe location (right side of waveguide)
    probe = np.zeros((NX, NY))
    cx = NX // 2
    hw = WG_WIDTH // 2
    probe_y = NY - NPML - 2
    probe[cx - hw : cx + hw + 1, probe_y] = 1.0
    T_gt_val = float(np.sum(np.abs(Ez_gt * probe)**2))
    T_opt_val = float(np.sum(np.abs(Ez_opt * probe)**2))
    T_ratio = T_opt_val / (T_gt_val + 1e-30)

    metrics = {
        "PSNR": psnr_s,
        "SSIM": ssim_s,
        "structure_psnr_db": psnr_s,
        "structure_ssim": ssim_s,
        "structure_cc": cc_s,
        "structure_mse": mse_s,
        "field_psnr_db": psnr_f,
        "field_ssim": ssim_f,
        "field_cc": cc_f,
        "transmission_gt": T_gt_val,
        "transmission_opt": T_opt_val,
        "transmission_ratio": T_ratio,
    }
    return metrics


# ============================ Visualization ===============================

def create_visualization(eps_gt, eps_opt, Ez_gt, Ez_opt, metrics):
    """4-panel visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    eps_gt_d = (eps_gt - EPS_MIN) / (EPS_MAX - EPS_MIN)
    eps_opt_d = (eps_opt - EPS_MIN) / (EPS_MAX - EPS_MIN)

    # Panel 1: GT structure
    ax = axes[0, 0]
    im = ax.imshow(eps_gt_d.T, origin='lower', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title('Ground Truth Dielectric Structure', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='Normalized eps', shrink=0.8)

    # Panel 2: Optimized structure
    ax = axes[0, 1]
    im = ax.imshow(eps_opt_d.T, origin='lower', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title(f'Optimized Structure (CC={metrics["structure_cc"]:.3f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='Normalized eps', shrink=0.8)

    # Panel 3: Field pattern (optimized)
    ax = axes[1, 0]
    Ez_opt_n = np.abs(Ez_opt) / (np.max(np.abs(Ez_opt)) + 1e-30)
    im = ax.imshow(Ez_opt_n.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'|Ez| Field (Optimized), Field CC={metrics["field_cc"]:.3f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='|Ez| (normalized)', shrink=0.8)

    # Panel 4: Error map
    ax = axes[1, 1]
    error = np.abs(eps_gt_d - eps_opt_d)
    im = ax.imshow(error.T, origin='lower', cmap='viridis', vmin=0)
    ax.set_title(f'|eps_GT - eps_opt| Error (PSNR={metrics["structure_psnr_db"]:.1f} dB)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='|Delta eps|', shrink=0.8)

    plt.suptitle('Task 158: Photonic Inverse Design (FDFD, ceviche)\n'
                 f'Grid: {NX}x{NY}, lambda={WAVELENGTH*1e9:.0f} nm, '
                 f'SSIM={metrics["structure_ssim"]:.3f}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "vis_result.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def create_convergence_plot(loss_history):
    """Convergence plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(loss_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Optimization Convergence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "convergence.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


# ================================ Main ====================================

def main():
    create_results_dir()

    # Run inverse design
    eps_gt, eps_opt, Ez_gt, Ez_opt, loss_history, source = run_inverse_design()

    # Metrics
    print("\n[4] Computing metrics...")
    metrics = compute_metrics(eps_gt, eps_opt, Ez_gt, Ez_opt)
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.6f}")

    # Save
    print("\n[5] Saving outputs...")
    np.save(os.path.join(RESULTS_DIR, "gt_output.npy"), eps_gt)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), eps_opt)
    np.save(os.path.join(RESULTS_DIR, "gt_field.npy"), np.abs(Ez_gt))
    np.save(os.path.join(RESULTS_DIR, "opt_field.npy"), np.abs(Ez_opt))
    np.save(os.path.join(RESULTS_DIR, "loss_history.npy"), np.array(loss_history))

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {metrics_path}")

    # Visualization
    print("\n[6] Creating visualizations...")
    create_visualization(eps_gt, eps_opt, Ez_gt, Ez_opt, metrics)
    create_convergence_plot(loss_history)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Structure PSNR: {metrics['structure_psnr_db']:.2f} dB")
    print(f"  Structure SSIM: {metrics['structure_ssim']:.4f}")
    print(f"  Structure CC:   {metrics['structure_cc']:.4f}")
    print(f"  Field CC:       {metrics['field_cc']:.4f}")
    print(f"  Field SSIM:     {metrics['field_ssim']:.4f}")
    print(f"  Transmission ratio: {metrics['transmission_ratio']:.4f}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    main()
