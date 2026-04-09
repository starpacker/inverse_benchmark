import sys
import os
import dill
import numpy as np
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================

def aperture_sum(image, row, col, radius):
    """Sum of pixel values inside a circular aperture."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius ** 2
    return image[mask].sum(), mask

def compute_snr_at_position(image, row, col, fwhm):
    """
    Detection SNR (aperture photometry, following Mawet et al. 2014).

    Signal = sum in planet aperture − mean of reference aperture sums.
    Noise  = std of reference aperture sums around the annulus.
    """
    ny, nx = image.shape
    cy, cx_img = ny // 2, nx // 2
    sep = np.sqrt((row - cy) ** 2 + (col - cx_img) ** 2)
    ap_r = fwhm / 2.0

    signal, _ = aperture_sum(image, row, col, ap_r)

    # Reference apertures at the same radial separation
    n_ref = max(int(2 * np.pi * sep / (2 * ap_r + 1)), 8)
    ref_sums = []
    for k in range(n_ref):
        theta = 2 * np.pi * k / n_ref
        rr = cy + sep * np.sin(theta)
        cc = cx_img + sep * np.cos(theta)
        if np.sqrt((rr - row) ** 2 + (cc - col) ** 2) < 3 * ap_r:
            continue
        if rr < ap_r or rr >= ny - ap_r or cc < ap_r or cc >= nx - ap_r:
            continue
        s, _ = aperture_sum(image, rr, cc, ap_r)
        ref_sums.append(s)

    if len(ref_sums) < 3:
        return np.abs(signal) / (np.abs(signal) * 0.1 + 1e-10)

    noise_std = np.std(ref_sums)
    mean_bg = np.mean(ref_sums)
    snr = (signal - mean_bg) / (noise_std + 1e-10)
    return snr

def compute_snr_map(image):
    """Pixel-wise SNR map using annular noise estimation."""
    ny, nx = image.shape
    cy, cx_img = ny // 2, nx // 2
    yy, xx = np.mgrid[:ny, :nx]
    r_map = np.sqrt((yy - cy) ** 2 + (xx - cx_img) ** 2)

    snr_map = np.zeros_like(image)
    max_r = int(r_map.max()) + 1
    for r in range(3, max_r):
        annulus = (r_map >= r - 1.5) & (r_map < r + 1.5)
        vals = image[annulus]
        if len(vals) > 10:
            std = np.std(vals)
            mean = np.mean(vals)
            if std > 1e-10:
                ring = (r_map >= r - 0.5) & (r_map < r + 0.5)
                snr_map[ring] = (image[ring] - mean) / std
    return snr_map

def find_peak_near(image, row, col, search_radius=10):
    """Find the peak pixel position near (row, col)."""
    ny, nx = image.shape
    r0 = max(0, int(row - search_radius))
    r1 = min(ny, int(row + search_radius + 1))
    c0 = max(0, int(col - search_radius))
    c1 = min(nx, int(col + search_radius + 1))
    sub = image[r0:r1, c0:c1]
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    return r0 + idx[0], c0 + idx[1]

def evaluate_results(
    final_image,
    ground_truth,
    cube,
    angles,
    params,
    save_dir=None,
    vis_path=None
):
    """
    Evaluate reconstruction quality and optionally save results.
    """
    gt_row, gt_col = ground_truth["planet_position"]
    clean_planet = ground_truth["clean_planet_image"]
    planet_fwhm = params.get("planet_fwhm", 5.0)
    image_size = params.get("image_size", 101)
    planet_sep = params.get("planet_sep", 30)

    # Detection SNR
    snr = compute_snr_at_position(final_image, gt_row, gt_col, planet_fwhm)

    # Position error
    peak_row, peak_col = find_peak_near(final_image, gt_row, gt_col)
    pos_error = np.sqrt((peak_row - gt_row) ** 2 + (peak_col - gt_col) ** 2)

    # Photometric accuracy
    ap_r = planet_fwhm / 2.0
    recovered, _ = aperture_sum(final_image, gt_row, gt_col, ap_r)
    injected, _ = aperture_sum(clean_planet, gt_row, gt_col, ap_r)
    photo_acc = (recovered / (injected + 1e-10)) * 100.0

    # PSNR
    signal_max = clean_planet.max()
    mse = np.mean((final_image - clean_planet) ** 2)
    psnr = 10.0 * np.log10(signal_max ** 2 / (mse + 1e-10)) if mse > 0 else float("inf")

    metrics = {
        "snr": snr,
        "pos_error": pos_error,
        "photometric_accuracy": photo_acc,
        "psnr": psnr,
    }

    # Save outputs if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "gt_output.npy"), clean_planet)
        np.save(os.path.join(save_dir, "recon_output.npy"), final_image)
        np.save(os.path.join(save_dir, "adi_cube.npy"), cube)
        np.save(os.path.join(save_dir, "angles.npy"), angles)

    # Create visualization if path provided
    if vis_path is not None:
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        cx = cy = image_size // 2

        # Panel 1: raw frame (log-stretch)
        ax = axes[0, 0]
        frame = cube[len(cube) // 2]
        log_frame = np.log10(np.clip(frame, 1, None))
        im0 = ax.imshow(log_frame, origin="lower", cmap="inferno")
        ax.set_title("Raw ADI Frame (log₁₀ stretch)", fontsize=13, fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        plt.colorbar(im0, ax=ax, label="log₁₀(Counts)", shrink=0.85)
        ax.plot(cx, cy, "w+", ms=12, mew=2, label="Star")
        mid = len(cube) // 2
        a_rad = np.radians(angles[mid])
        ax.plot(cx + planet_sep * np.cos(a_rad), cy + planet_sep * np.sin(a_rad),
                "co", ms=8, mfc="none", mew=1.5, label="Planet (this frame)")
        ax.legend(loc="upper left", fontsize=8)

        # Panel 2: PCA-ADI residual
        ax = axes[0, 1]
        vabs = np.percentile(np.abs(final_image), 99.5)
        if vabs < 1e-10:
            vabs = 1.0
        im1 = ax.imshow(final_image, origin="lower", cmap="RdBu_r",
                        vmin=-vabs, vmax=vabs)
        ax.set_title("PCA-ADI Residual (Mean Combined)", fontsize=13,
                     fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        plt.colorbar(im1, ax=ax, label="Residual flux", shrink=0.85)
        circ = plt.Circle((gt_col, gt_row), 5, fill=False, ec="lime",
                           lw=2, ls="--", label="True planet")
        ax.add_patch(circ)
        ax.legend(loc="upper left", fontsize=9)

        # Panel 3: ground truth
        ax = axes[1, 0]
        im2 = ax.imshow(clean_planet, origin="lower", cmap="hot")
        ax.set_title("Ground Truth Planet Map", fontsize=13, fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        ax.plot(gt_col, gt_row, "c+", ms=15, mew=2,
                label=f"Planet ({gt_row}, {gt_col})")
        ax.legend(loc="upper left", fontsize=9)
        plt.colorbar(im2, ax=ax, label="Flux", shrink=0.85)

        # Panel 4: SNR map
        ax = axes[1, 1]
        snr_map = compute_snr_map(final_image)
        vmax_snr = max(np.percentile(np.abs(snr_map), 99.5), 1)
        im3 = ax.imshow(snr_map, origin="lower", cmap="viridis",
                        vmin=-3, vmax=vmax_snr)
        ax.set_title("SNR Map", fontsize=13, fontweight="bold")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        circ2 = plt.Circle((gt_col, gt_row), 5, fill=False, ec="red",
                            lw=2, ls="--",
                            label=f"Planet SNR={metrics['snr']:.1f}")
        ax.add_patch(circ2)
        ax.legend(loc="upper left", fontsize=9)
        plt.colorbar(im3, ax=ax, label="SNR", shrink=0.85)

        fig.suptitle(
            f"Task 161 — PCA-ADI High-Contrast Imaging\n"
            f"SNR={metrics['snr']:.1f}  |  Pos Error={metrics['pos_error']:.1f} px  |  "
            f"Photo Acc={metrics['photometric_accuracy']:.1f}%  |  "
            f"PSNR={metrics['psnr']:.2f} dB",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        fig.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIS] Saved → {vis_path}")

    return metrics

# ============================================================
# Main test logic
# ============================================================

def main():
    data_paths = ['/data/yjh/pynpoint_hci_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    # Also scan the directory for any inner files we might have missed
    if outer_path:
        data_dir = os.path.dirname(outer_path)
        if os.path.isdir(data_dir):
            for f in os.listdir(data_dir):
                full = os.path.join(data_dir, f)
                if full not in data_paths and 'parent_function' in f and 'run_inversion' in f:
                    inner_paths.append(full)
    
    if outer_path is None:
        print("[ERROR] No outer data file found.")
        sys.exit(1)
    
    print(f"[INFO] Outer data: {outer_path}")
    print(f"[INFO] Inner data files: {inner_paths}")
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data. Keys: {list(outer_data.keys()) if isinstance(outer_data, dict) else type(outer_data)}")
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # Run the agent's function
    try:
        agent_output = run_inversion(*args, **kwargs)
        print("[INFO] Agent run_inversion executed successfully.")
    except Exception as e:
        print(f"[ERROR] Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if len(inner_paths) > 0:
        # Chained execution pattern
        print("[INFO] Chained execution detected.")
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"[ERROR] Chained call failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        print("[INFO] Direct execution detected.")
        final_result = agent_output
        std_result = std_output
    
    # Extract final_image from results (they are dicts)
    if isinstance(final_result, dict) and 'final_image' in final_result:
        agent_final_image = final_result['final_image']
    else:
        agent_final_image = final_result
    
    if isinstance(std_result, dict) and 'final_image' in std_result:
        std_final_image = std_result['final_image']
    else:
        std_final_image = std_result
    
    # We need ground_truth and params for evaluate_results
    # These are typically embedded in the data or we need to reconstruct them
    # The cube and angles are in args
    cube = args[0] if len(args) > 0 else kwargs.get('cube', None)
    angles = args[1] if len(args) > 1 else kwargs.get('angles', None)
    
    if cube is None or angles is None:
        print("[ERROR] Could not extract cube and angles from input data.")
        sys.exit(1)
    
    # Try to find ground_truth and params from the pkl data or reconstruct
    # Check if outer_data has ground_truth/params stored
    ground_truth = outer_data.get('ground_truth', None)
    params = outer_data.get('params', None)
    
    # If not in outer_data, check if there's a separate ground truth file
    if ground_truth is None:
        data_dir = os.path.dirname(outer_path)
        gt_candidates = ['ground_truth.pkl', 'gt_data.pkl', 'standard_data_ground_truth.pkl']
        for gc in gt_candidates:
            gc_path = os.path.join(data_dir, gc)
            if os.path.exists(gc_path):
                try:
                    with open(gc_path, 'rb') as f:
                        gt_data = dill.load(f)
                    if isinstance(gt_data, dict):
                        ground_truth = gt_data.get('ground_truth', gt_data)
                        params = gt_data.get('params', params)
                    break
                except:
                    pass
    
    # If we still don't have ground_truth, we fall back to direct comparison
    if ground_truth is None or params is None:
        print("[INFO] No ground_truth/params found. Using direct numerical comparison.")
        
        # Direct comparison: check that agent output matches std output closely
        try:
            # Compare final images
            if agent_final_image is not None and std_final_image is not None:
                diff = np.abs(agent_final_image - std_final_image)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                std_max = np.max(np.abs(std_final_image))
                
                # Relative error
                rel_error = max_diff / (std_max + 1e-10)
                
                print(f"[COMPARISON] Max absolute diff: {max_diff:.6e}")
                print(f"[COMPARISON] Mean absolute diff: {mean_diff:.6e}")
                print(f"[COMPARISON] Std max value: {std_max:.6e}")
                print(f"[COMPARISON] Relative error: {rel_error:.6e}")
                
                # Also compare other keys if both are dicts
                if isinstance(final_result, dict) and isinstance(std_result, dict):
                    for key in ['residuals', 'pca_model', 'derotated']:
                        if key in final_result and key in std_result:
                            d = np.max(np.abs(final_result[key] - std_result[key]))
                            s = np.max(np.abs(std_result[key]))
                            r = d / (s + 1e-10)
                            print(f"[COMPARISON] '{key}' - max diff: {d:.6e}, rel error: {r:.6e}")
                
                # Check correlation
                corr = np.corrcoef(agent_final_image.ravel(), std_final_image.ravel())[0, 1]
                print(f"[COMPARISON] Correlation: {corr:.6f}")
                
                # Success criteria: relative error < 10% and high correlation
                if rel_error < 0.1 and corr > 0.95:
                    print(f"[PASS] Agent output matches standard output. Rel error={rel_error:.6e}, Corr={corr:.6f}")
                    sys.exit(0)
                else:
                    print(f"[FAIL] Agent output differs significantly. Rel error={rel_error:.6e}, Corr={corr:.6f}")
                    sys.exit(1)
            else:
                print("[ERROR] Could not extract final images for comparison.")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # We have ground_truth and params - use evaluate_results
        print("[INFO] Using evaluate_results for comparison.")
        
        try:
            metrics_agent = evaluate_results(
                final_image=agent_final_image,
                ground_truth=ground_truth,
                cube=cube,
                angles=angles,
                params=params,
            )
            print(f"[AGENT METRICS] {metrics_agent}")
        except Exception as e:
            print(f"[ERROR] evaluate_results failed for agent: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            metrics_std = evaluate_results(
                final_image=std_final_image,
                ground_truth=ground_truth,
                cube=cube,
                angles=angles,
                params=params,
            )
            print(f"[STD METRICS] {metrics_std}")
        except Exception as e:
            print(f"[ERROR] evaluate_results failed for std: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare metrics
        # SNR: higher is better
        # pos_error: lower is better
        # photometric_accuracy: closer to 100% is better (but we just check it's not terrible)
        # PSNR: higher is better
        
        snr_agent = metrics_agent['snr']
        snr_std = metrics_std['snr']
        pos_agent = metrics_agent['pos_error']
        pos_std = metrics_std['pos_error']
        psnr_agent = metrics_agent['psnr']
        psnr_std = metrics_std['psnr']
        
        print(f"\nScores -> Agent SNR: {snr_agent:.2f}, Standard SNR: {snr_std:.2f}")
        print(f"Scores -> Agent pos_error: {pos_agent:.2f}, Standard pos_error: {pos_std:.2f}")
        print(f"Scores -> Agent PSNR: {psnr_agent:.2f}, Standard PSNR: {psnr_std:.2f}")
        
        # Check: agent SNR should be at least 90% of standard
        passed = True
        
        if snr_std > 0 and snr_agent < snr_std * 0.9:
            print(f"[FAIL] Agent SNR ({snr_agent:.2f}) is significantly lower than standard ({snr_std:.2f})")
            passed = False
        
        if pos_std >= 0 and pos_agent > pos_std * 1.5 + 2.0:  # Allow some margin
            print(f"[FAIL] Agent pos_error ({pos_agent:.2f}) is significantly worse than standard ({pos_std:.2f})")
            passed = False
        
        if psnr_std > 0 and psnr_agent < psnr_std * 0.9:
            print(f"[FAIL] Agent PSNR ({psnr_agent:.2f}) is significantly lower than standard ({psnr_std:.2f})")
            passed = False
        
        if passed:
            print("[PASS] Agent performance is acceptable.")
            sys.exit(0)
        else:
            print("[FAIL] Agent performance degraded significantly.")
            sys.exit(1)


if __name__ == "__main__":
    main()