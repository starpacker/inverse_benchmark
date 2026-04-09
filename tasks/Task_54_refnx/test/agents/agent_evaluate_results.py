import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from refnx.reflect import SLD as SLDobj, ReflectModel, Structure

def build_structure(params, vary=False, dq_over_q=0.05):
    """
    Build a refnx Structure from a parameter dict.

    Parameters
    ----------
    params : dict
        Keys: polymer_thick, polymer_sld, polymer_rough,
              sio2_thick, sio2_sld, sio2_rough,
              si_sld, si_rough, bkg
    vary : bool
        If True, set bounds and mark parameters as variable.
    dq_over_q : float
        Resolution smearing parameter.

    Returns
    -------
    structure : refnx.reflect.Structure
    model : refnx.reflect.ReflectModel
    slabs : dict of slab objects
    """
    air = SLDobj(0.0, name='air')
    polymer = SLDobj(params["polymer_sld"], name='polymer')
    sio2 = SLDobj(params["sio2_sld"], name='sio2')
    si = SLDobj(params["si_sld"], name='si')

    polymer_slab = polymer(params["polymer_thick"], params["polymer_rough"])
    sio2_slab = sio2(params["sio2_thick"], params["sio2_rough"])
    si_slab = si(0, params["si_rough"])

    if vary:
        polymer_slab.thick.setp(bounds=(50, 500), vary=True)
        polymer_slab.sld.real.setp(bounds=(0.3, 6.0), vary=True)
        polymer_slab.rough.setp(bounds=(1, 25), vary=True)
        sio2_slab.thick.setp(bounds=(3, 50), vary=True)
        sio2_slab.rough.setp(bounds=(0.5, 15), vary=True)
        si_slab.rough.setp(bounds=(0.5, 15), vary=True)

    structure = air | polymer_slab | sio2_slab | si_slab
    model = ReflectModel(structure, bkg=params["bkg"], dq=dq_over_q * 100)

    if vary:
        model.bkg.setp(bounds=(1e-10, 1e-5), vary=True)

    return structure, model, {
        "polymer_slab": polymer_slab,
        "sio2_slab": sio2_slab,
        "si_slab": si_slab,
    }

def sld_profile_from_params(params, z_max=400, n_pts=500):
    """Compute SLD-vs-depth profile using refnx Structure."""
    structure, _, _ = build_structure(params, vary=False)
    z = np.linspace(0, z_max, n_pts)
    sld_profile = structure.sld_profile(z)
    return sld_profile[:, 0], sld_profile[:, 1]

def evaluate_results(result_dict, results_dir):
    """
    Compute reconstruction quality metrics and save results.

    Reflectivity-space:
      - RMSE(log₁₀R)
      - CC(log₁₀R): Pearson correlation in log₁₀ space
      - PSNR(log₁₀R)
      - SSIM(log₁₀R)

    Parameter-space:
      - Absolute errors for thickness, SLD, roughness of each layer
      - Relative errors

    Also saves metrics, reconstruction arrays, and visualization.

    Parameters
    ----------
    result_dict : dict
        Contains fitted_params, R_fit, q, R_meas, R_clean, gt_params.
    results_dir : str
        Directory to save results.

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    os.makedirs(results_dir, exist_ok=True)

    gt_params = result_dict["gt_params"]
    fit_params = result_dict["fitted_params"]
    R_clean = result_dict["R_clean"]
    R_fit = result_dict["R_fit"]
    R_meas = result_dict["R_meas"]
    q = result_dict["q"]

    logR_gt = np.log10(np.maximum(R_clean, 1e-12))
    logR_fit = np.log10(np.maximum(R_fit, 1e-12))

    # Reflectivity metrics
    rmse_logR = float(np.sqrt(np.mean((logR_gt - logR_fit) ** 2)))
    cc_logR = float(np.corrcoef(logR_gt, logR_fit)[0, 1])

    data_range = logR_gt.max() - logR_gt.min()
    mse = np.mean((logR_gt - logR_fit) ** 2)
    psnr_logR = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # 1-D SSIM: tile to make 2D (7×N) so win_size=7 works
    tile_rows = 7
    a2d = np.tile(logR_gt, (tile_rows, 1))
    b2d = np.tile(logR_fit, (tile_rows, 1))
    ssim_logR = float(ssim_fn(
        a2d, b2d,
        data_range=data_range, win_size=7
    ))

    # Parameter recovery
    param_keys = [
        ("polymer_thick", "d_polymer [Å]"),
        ("polymer_sld", "SLD_polymer [1e-6 Å⁻²]"),
        ("polymer_rough", "σ_polymer [Å]"),
        ("sio2_thick", "d_SiO₂ [Å]"),
        ("sio2_rough", "σ_SiO₂ [Å]"),
        ("si_rough", "σ_Si [Å]"),
    ]

    param_errors = {}
    for key, label in param_keys:
        gt = gt_params[key]
        fit = fit_params[key]
        err = abs(gt - fit)
        rel = err / max(abs(gt), 1e-12) * 100
        param_errors[f"gt_{key}"] = float(gt)
        param_errors[f"fit_{key}"] = float(fit)
        param_errors[f"abs_err_{key}"] = float(err)
        param_errors[f"rel_err_{key}"] = float(rel)

    metrics = {
        "PSNR_logR": psnr_logR,
        "SSIM_logR": ssim_logR,
        "CC_logR": cc_logR,
        "RMSE_logR": rmse_logR,
        **param_errors,
    }

    # Print metrics
    print("\n[EVAL] Computing metrics ...")
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")

    # Save metrics and arrays
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), R_fit)
    np.save(os.path.join(results_dir, "ground_truth.npy"), R_clean)
    np.save(os.path.join(results_dir, "measurements.npy"), R_meas)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Reflectivity
    ax = axes[0, 0]
    ax.semilogy(q, R_clean, 'b-', lw=2, label='Ground Truth', zorder=3)
    ax.semilogy(q, R_meas, 'k.', ms=1.5, alpha=0.4, label='Noisy Data')
    ax.semilogy(q, R_fit, 'r--', lw=1.5, label='refnx Fit', zorder=2)
    ax.set_xlabel('Q [Å⁻¹]')
    ax.set_ylabel('Reflectivity R(Q)')
    ax.set_title('(a) X-ray Reflectivity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Residuals in log space
    ax = axes[0, 1]
    ax.plot(q, logR_gt - logR_fit, 'g-', lw=1)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Q [Å⁻¹]')
    ax.set_ylabel('Δ log₁₀R')
    ax.set_title(f'(b) Residuals  RMSE={metrics["RMSE_logR"]:.4f}')
    ax.grid(True, alpha=0.3)

    # (c) SLD profile
    ax = axes[1, 0]
    try:
        z_gt, sld_gt = sld_profile_from_params(gt_params)
        z_fit, sld_fit = sld_profile_from_params(fit_params)
        ax.plot(z_gt, sld_gt, 'b-', lw=2, label='Ground Truth')
        ax.plot(z_fit, sld_fit, 'r--', lw=2, label='refnx Fit')
    except Exception:
        ax.text(0.5, 0.5, 'SLD profile unavailable',
                transform=ax.transAxes, ha='center')
    ax.set_xlabel('Depth [Å]')
    ax.set_ylabel('SLD [10⁻⁶ Å⁻²]')
    ax.set_title('(c) SLD Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Parameter bar chart
    ax = axes[1, 1]
    labels = ['d_poly', 'SLD_poly', 'σ_poly', 'd_SiO₂', 'σ_SiO₂', 'σ_Si']
    keys = ['polymer_thick', 'polymer_sld', 'polymer_rough',
            'sio2_thick', 'sio2_rough', 'si_rough']
    gt_vals = [gt_params[k] for k in keys]
    fit_vals = [fit_params[k] for k in keys]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, gt_vals, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_vals, w, label='Fit', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_title('(d) Parameter Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"refnx — X-ray Reflectometry Inversion\n"
        f"PSNR(logR)={metrics['PSNR_logR']:.1f} dB  |  "
        f"SSIM(logR)={metrics['SSIM_logR']:.4f}  |  "
        f"CC(logR)={metrics['CC_logR']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics
