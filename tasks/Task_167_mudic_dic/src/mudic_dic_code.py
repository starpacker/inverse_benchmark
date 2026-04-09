"""
Task 167: mudic_dic
Digital Image Correlation (DIC): Recover full-field displacement and strain
from synthetic speckle image pairs using the muDIC library.

Inverse Problem: Given a reference speckle image and a deformed image,
recover the full-field displacement (u, v) and strain fields (εxx, εyy, εxy).
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Repository path
# ---------------------------------------------------------------------------
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.insert(0, REPO_DIR)

import muDIC as dic
import muDIC.vlab as vlab

# Logging
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_synthetic_data():
    """
    Use muDIC's virtual lab to generate a reference + deformed speckle image
    pair with a known displacement field (harmonic bilateral).

    Returns
    -------
    image_stack : dic.ImageStack
        Stack of [reference, deformed] images.
    displacement_function : callable
        The function u(xs, ys) -> (u_x, u_y) used to deform the image.
    omega : float
        Angular frequency (in output-image pixel units).
    amp : float
        Amplitude of the deformation (in output-image pixel units).
    image_shape : tuple
        (H, W) of the output (down-sampled) images.
    downsample_factor : int
    """
    n_imgs = 2  # reference + 1 deformed
    image_shape = (256, 256)
    downsample_factor = 4
    super_image_shape = tuple(d * downsample_factor for d in image_shape)

    # Speckle pattern
    speckle_image = vlab.rosta_speckle(super_image_shape, dot_size=4,
                                       density=0.5, smoothness=2.0)

    # Deformation: harmonic bilateral (sinusoidal in both x and y)
    displacement_function = vlab.deformation_fields.harmonic_bilat
    omega = 2.0 * np.pi / (image_shape[0] * downsample_factor)  # in super-image coords
    amp = 2.0 * downsample_factor  # amplitude in super-image pixels

    image_deformer = vlab.imageDeformer_from_uFunc(
        displacement_function, omega=omega, amp=amp
    )

    # Down-sampler with realistic sensor model
    downsampler = vlab.Downsampler(
        image_shape=super_image_shape, factor=downsample_factor,
        fill=0.95, pixel_offset_stddev=0.05
    )

    # Small additive Gaussian noise (2 %)
    noise_inj = vlab.noise_injector("gaussian", sigma=0.02)

    # Build the synthetic image pipeline
    image_generator = vlab.SyntheticImageGenerator(
        speckle_image=speckle_image,
        image_deformer=image_deformer,
        downsampler=downsampler,
        noise_injector=noise_inj,
        n=n_imgs,
    )
    image_stack = dic.ImageStack(image_generator)

    # omega / amp in *output-image* coordinates (after downsampling)
    omega_out = omega * downsample_factor   # angular freq per output pixel
    amp_out = amp / downsample_factor        # amplitude in output pixels

    return (image_stack, displacement_function,
            omega_out, amp_out, omega, amp,
            image_shape, downsample_factor)


def run_dic_analysis(image_stack, image_shape):
    """
    Run muDIC DIC analysis on the image stack.

    Returns
    -------
    results : DICOutput
    fields  : dic.Fields
    mesh    : Mesh object
    """
    # Mesh with B-spline elements (degree 3)
    mesher = dic.Mesher(deg_n=3, deg_e=3, type="spline")

    # Mesh the ROI (leave border margin)
    margin = 30
    mesh = mesher.mesh(
        image_stack,
        Xc1=margin, Xc2=image_shape[1] - margin,
        Yc1=margin, Yc2=image_shape[0] - margin,
        n_elx=8, n_ely=8,
        GUI=False,
    )

    # DIC input
    dic_input = dic.DICInput(mesh, image_stack)
    dic_input.tol = 1e-6

    # Run
    dic_job = dic.DICAnalysis(dic_input)
    results = dic_job.run()

    # Post-process
    fields = dic.Fields(results, seed=101)

    return results, fields, mesh


def compute_ground_truth_on_dic_grid(fields, displacement_function,
                                      omega_super, amp_super,
                                      image_stack, downsample_factor):
    """
    Evaluate the analytical displacement field on the same material points
    that muDIC uses, so the comparison is fair.

    Returns
    -------
    gt_ux, gt_uy : ndarray  – ground-truth displacement at DIC nodes
    dic_ux, dic_uy : ndarray  – DIC-recovered displacement
    coords_e, coords_n : ndarray  – the element coordinates
    """
    # DIC displacement: shape [elm, component, e, n, frame]
    disp = fields.disp()  # displacement wrt frame-0
    coords = fields.coords()

    # Material-point image coordinates at frame 1
    e_coords = coords[0, 1, :, :, 1]  # x (column) in image pixels
    n_coords = coords[0, 0, :, :, 1]  # y (row)   in image pixels

    # Full-field analytical displacement on the whole image
    xs, ys = dic.utils.image_coordinates(image_stack[0])

    # omega and amp for the analytical function are in super-image units,
    # but image_coordinates returns output-image coordinates.
    # The displacement function was called with super-image omega/amp,
    # so we need to convert output coords to super-image coords:
    # super_coord = output_coord * downsample_factor
    # But the displacement values returned are also in super-image pixels,
    # so we divide by downsample_factor to get output-image pixels.
    omega_out = omega_super * downsample_factor
    amp_out = amp_super / downsample_factor

    u_x_full, u_y_full = displacement_function(
        xs, ys, omega=omega_out, amp=amp_out
    )

    # Extract GT at DIC material points
    gt_ux = dic.utils.extract_points_from_image(u_x_full, np.array([e_coords, n_coords]))
    gt_uy = dic.utils.extract_points_from_image(u_y_full, np.array([e_coords, n_coords]))

    # DIC displacement at frame 1
    dic_ux = disp[0, 0, :, :, 1]  # x-component
    dic_uy = disp[0, 1, :, :, 1]  # y-component

    return gt_ux, gt_uy, dic_ux, dic_uy, e_coords, n_coords


def compute_strain_fields(fields):
    """
    Compute engineering strain from muDIC Fields.

    Returns
    -------
    eps_xx, eps_yy, eps_xy : ndarray  (on element grid, frame=1)
    """
    strain = fields.eng_strain()
    # strain shape: [elm, i, j, e, n, frame]
    eps_xx = strain[0, 0, 0, :, :, 1]
    eps_yy = strain[0, 1, 1, :, :, 1]
    eps_xy = strain[0, 0, 1, :, :, 1]
    return eps_xx, eps_yy, eps_xy


def compute_gt_strain(gt_ux, gt_uy, e_coords, n_coords, omega_out, amp_out):
    """
    Compute ground-truth engineering strain analytically from the known
    harmonic bilateral displacement field.

    For u_x = A * sin(ω*x) * sin(ω*y), u_y = same:
        ∂u_x/∂x = A * ω * cos(ω*x) * sin(ω*y)
        ∂u_x/∂y = A * ω * sin(ω*x) * cos(ω*y)

    NOTE: muDIC's Fields grid uses element coordinates [e, n] where
    the first axis corresponds to y (rows) and the second to x (cols).
    The coords[0,0,...] = y and coords[0,1,...] = x in image space,
    but the *grid axes* (axis-0, axis-1 of the 2D arrays) map to
    (n → y direction, e → x direction).  The strain tensor from muDIC
    is computed in element-local coordinates: component (0,0) corresponds
    to the e-direction (y in image), and (1,1) to the n-direction (x in image).

    To match muDIC's convention, we compute:
        gt_exx = strain[0,0,0] → ∂u_y/∂y = A*ω*sin(ω*x)*cos(ω*y)  [e-dir strain]
        gt_eyy = strain[0,1,1] → ∂u_x/∂x = A*ω*cos(ω*x)*sin(ω*y)  [n-dir strain]
        gt_exy = 0.5*(∂u_y/∂x + ∂u_x/∂y)

    Returns
    -------
    gt_exx, gt_eyy, gt_exy : ndarray  (matching muDIC's [0,0], [1,1], [0,1] components)
    """
    # e_coords = image x, n_coords = image y
    x = e_coords
    y = n_coords
    w = omega_out
    A = amp_out

    # muDIC strain[0,0,0] = e-direction = y-derivative
    # muDIC strain[0,1,1] = n-direction = x-derivative
    # For harmonic_bilat: u_x = u_y = A*sin(ωx)*sin(ωy)
    gt_exx = A * w * np.sin(w * x) * np.cos(w * y)   # ∂u/∂y (matches strain[0,0,0])
    gt_eyy = A * w * np.cos(w * x) * np.sin(w * y)   # ∂u/∂x (matches strain[0,1,1])
    gt_exy = 0.5 * (A * w * np.cos(w * x) * np.sin(w * y) +
                     A * w * np.sin(w * x) * np.cos(w * y))  # ∂u_y/∂x + ∂u_x/∂y

    return gt_exx, gt_eyy, gt_exy


def compute_metrics(gt_ux, gt_uy, dic_ux, dic_uy,
                    gt_exx, gt_eyy, gt_exy,
                    eps_xx, eps_yy, eps_xy):
    """Compute RMSE, PSNR, and CC metrics."""
    # --- Displacement metrics ---
    rmse_ux = float(np.sqrt(np.mean((dic_ux - gt_ux) ** 2)))
    rmse_uy = float(np.sqrt(np.mean((dic_uy - gt_uy) ** 2)))
    disp_mag_gt = np.sqrt(gt_ux**2 + gt_uy**2)
    disp_mag_dic = np.sqrt(dic_ux**2 + dic_uy**2)
    rmse_disp = float(np.sqrt(np.mean((disp_mag_dic - disp_mag_gt)**2)))
    max_disp = float(np.max(np.abs(disp_mag_gt))) + 1e-12

    # Displacement PSNR
    mse_disp = float(np.mean((disp_mag_dic - disp_mag_gt)**2))
    psnr_disp = float(10.0 * np.log10(max_disp**2 / (mse_disp + 1e-20)))

    # Displacement CC
    a = (disp_mag_dic - disp_mag_dic.mean()).ravel()
    b = (disp_mag_gt - disp_mag_gt.mean()).ravel()
    cc_disp = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 else 0.0

    # --- Strain metrics ---
    # Trim borders where finite-diff gradient is unreliable
    s = 2  # border trim
    def _trim(arr):
        return arr[s:-s, s:-s]

    strain_gt_flat = np.stack([_trim(gt_exx), _trim(gt_eyy), _trim(gt_exy)]).ravel()
    strain_dic_flat = np.stack([_trim(eps_xx), _trim(eps_yy), _trim(eps_xy)]).ravel()

    rmse_strain = float(np.sqrt(np.mean((strain_dic_flat - strain_gt_flat)**2)))
    max_strain = float(np.max(np.abs(strain_gt_flat))) + 1e-12
    mse_strain = float(np.mean((strain_dic_flat - strain_gt_flat)**2))
    psnr_strain = float(10.0 * np.log10(max_strain**2 / (mse_strain + 1e-20)))

    cc_strain_a = (strain_dic_flat - strain_dic_flat.mean())
    cc_strain_b = (strain_gt_flat - strain_gt_flat.mean())
    cc_strain = float(np.corrcoef(cc_strain_a, cc_strain_b)[0, 1]) if np.std(cc_strain_a) > 1e-12 else 0.0

    # --- SSIM-like metric for displacement field (structural similarity) ---
    # Simple implementation
    mu_x = disp_mag_dic.mean()
    mu_y = disp_mag_gt.mean()
    sig_x = disp_mag_dic.std()
    sig_y = disp_mag_gt.std()
    sig_xy = np.mean((disp_mag_dic - mu_x) * (disp_mag_gt - mu_y))
    C1 = (0.01 * max_disp)**2
    C2 = (0.03 * max_disp)**2
    ssim_disp = float(((2*mu_x*mu_y + C1)*(2*sig_xy + C2)) /
                       ((mu_x**2 + mu_y**2 + C1)*(sig_x**2 + sig_y**2 + C2)))

    metrics = {
        "displacement_rmse_ux": round(rmse_ux, 6),
        "displacement_rmse_uy": round(rmse_uy, 6),
        "displacement_rmse_magnitude": round(rmse_disp, 6),
        "displacement_psnr_dB": round(psnr_disp, 2),
        "displacement_cc": round(cc_disp, 6),
        "displacement_ssim": round(ssim_disp, 6),
        "strain_rmse": round(rmse_strain, 6),
        "strain_psnr_dB": round(psnr_strain, 2),
        "strain_cc": round(cc_strain, 6),
        "max_gt_displacement_pixels": round(max_disp, 4),
    }
    return metrics


def visualize(image_stack, mesh, fields,
              gt_ux, gt_uy, dic_ux, dic_uy,
              eps_xx, gt_exx, metrics):
    """
    Create 4-subplot visualization:
    (a) Reference image with mesh overlay
    (b) Deformed image
    (c) Displacement magnitude map with GT overlay
    (d) Strain εxx: DIC vs GT comparison / error map
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Reference image with mesh overlay
    ax = axes[0, 0]
    ref_img = np.array(image_stack[0])
    ax.imshow(ref_img, cmap='gray', origin='upper')
    # Draw mesh lines
    mesh_obj = mesh
    for node_x in np.linspace(mesh_obj.Xc1, mesh_obj.Xc2, mesh_obj.n_elx + 1):
        ax.axvline(x=node_x, color='cyan', linewidth=0.5, alpha=0.6)
    for node_y in np.linspace(mesh_obj.Yc1, mesh_obj.Yc2, mesh_obj.n_ely + 1):
        ax.axhline(y=node_y, color='cyan', linewidth=0.5, alpha=0.6)
    ax.set_title("(a) Reference Image + Mesh", fontsize=12)
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")

    # (b) Deformed image
    ax = axes[0, 1]
    def_img = np.array(image_stack[1])
    ax.imshow(def_img, cmap='gray', origin='upper')
    ax.set_title("(b) Deformed Image", fontsize=12)
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")

    # (c) Displacement magnitude: DIC result
    ax = axes[1, 0]
    disp_mag_dic = np.sqrt(dic_ux**2 + dic_uy**2)
    disp_mag_gt = np.sqrt(gt_ux**2 + gt_uy**2)

    im = ax.imshow(disp_mag_dic, cmap='viridis', origin='upper', aspect='equal')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Displacement [px]")
    # Overlay GT contours
    ax.contour(disp_mag_gt, levels=5, colors='red', linewidths=0.8, linestyles='--')
    ax.set_title(f"(c) Disp. Magnitude (DIC)\nRMSE={metrics['displacement_rmse_magnitude']:.4f} px, CC={metrics['displacement_cc']:.4f}",
                 fontsize=11)
    ax.set_xlabel("Element e-coord")
    ax.set_ylabel("Element n-coord")

    # (d) Strain error map or comparison
    ax = axes[1, 1]
    s = 2
    strain_err = eps_xx[s:-s, s:-s] - gt_exx[s:-s, s:-s]
    vmax = max(abs(strain_err.min()), abs(strain_err.max())) or 1e-6
    im = ax.imshow(strain_err, cmap='RdBu_r', origin='upper', aspect='equal',
                   vmin=-vmax, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("εxx error")
    ax.set_title(f"(d) Strain εxx Error (DIC − GT)\nStrain RMSE={metrics['strain_rmse']:.6f}",
                 fontsize=11)
    ax.set_xlabel("Element e-coord")
    ax.set_ylabel("Element n-coord")

    plt.suptitle("Task 167: muDIC — Digital Image Correlation\n"
                 f"Displacement PSNR={metrics['displacement_psnr_dB']:.1f} dB, "
                 f"SSIM={metrics['displacement_ssim']:.4f}",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved reconstruction_result.png")


def main():
    print("=" * 60)
    print("Task 167: muDIC Digital Image Correlation")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n[1] Generating synthetic speckle images with known deformation...")
    (image_stack, displacement_function,
     omega_out, amp_out, omega_super, amp_super,
     image_shape, downsample_factor) = generate_synthetic_data()
    print(f"    Image shape: {image_shape}, downsample factor: {downsample_factor}")
    print(f"    Deformation: harmonic bilateral, amp={amp_out:.2f} px (output), omega={omega_out:.6f} rad/px")

    # 2. Run DIC analysis
    print("\n[2] Running muDIC DIC analysis (B-spline, 8×8 elements)...")
    results, fields, mesh = run_dic_analysis(image_stack, image_shape)
    print("    DIC analysis completed.")

    # 3. Compute GT displacement on DIC grid
    print("\n[3] Computing ground-truth displacement on DIC grid...")
    gt_ux, gt_uy, dic_ux, dic_uy, e_coords, n_coords = \
        compute_ground_truth_on_dic_grid(
            fields, displacement_function,
            omega_super, amp_super,
            image_stack, downsample_factor
        )

    # 4. Compute strain fields
    print("\n[4] Computing strain fields...")
    eps_xx, eps_yy, eps_xy = compute_strain_fields(fields)
    gt_exx, gt_eyy, gt_exy = compute_gt_strain(gt_ux, gt_uy, e_coords, n_coords,
                                                 omega_out, amp_out)

    # 5. Compute metrics
    print("\n[5] Computing metrics...")
    metrics = compute_metrics(
        gt_ux, gt_uy, dic_ux, dic_uy,
        gt_exx, gt_eyy, gt_exy,
        eps_xx, eps_yy, eps_xy
    )

    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # 6. Save outputs
    print("\n[6] Saving outputs...")
    # Ground truth: displacement fields
    gt_data = {
        'gt_ux': gt_ux,
        'gt_uy': gt_uy,
        'gt_exx': gt_exx,
        'gt_eyy': gt_eyy,
        'gt_exy': gt_exy,
    }
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_data)

    # Reconstruction: DIC displacement fields
    recon_data = {
        'dic_ux': dic_ux,
        'dic_uy': dic_uy,
        'eps_xx': eps_xx,
        'eps_yy': eps_yy,
        'eps_xy': eps_xy,
    }
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_data)

    # Metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved ground_truth.npy, reconstruction.npy, metrics.json")

    # 7. Visualization
    print("\n[7] Creating visualization...")
    visualize(image_stack, mesh, fields,
              gt_ux, gt_uy, dic_ux, dic_uy,
              eps_xx, gt_exx, metrics)

    print("\n" + "=" * 60)
    print("Task 167 COMPLETE")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    metrics = main()
