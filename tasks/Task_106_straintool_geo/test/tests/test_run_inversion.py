import sys
import os
import dill
import numpy as np
import traceback
import json

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# --- Inject Referee (evaluate_results) verbatim from Reference B ---

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR = "/data/yjh/website_assets/Task_106_straintool_geo"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def evaluate_results(data, inversion_result):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR, SSIM, CC metrics for each strain component,
    generates visualizations, and saves outputs.
    
    Args:
        data: dict from load_and_preprocess_data
        inversion_result: dict from run_inversion
        
    Returns:
        dict containing metrics
    """
    gt_exx = data['gt_exx']
    gt_exy = data['gt_exy']
    gt_eyy = data['gt_eyy']
    grid_x = data['grid_x']
    grid_y = data['grid_y']
    stations = data['stations']
    vx = data['vx']
    vy = data['vy']
    
    rec_exx = inversion_result['rec_exx']
    rec_exy = inversion_result['rec_exy']
    rec_eyy = inversion_result['rec_eyy']
    
    def compute_field_metrics(gt, rec):
        """Compute PSNR, SSIM, CC for 2D field comparison."""
        gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-15)
        rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-15)
        
        # PSNR
        mse = np.mean((gt_n - rec_n)**2)
        psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0
        
        # SSIM
        data_range = max(gt_n.max() - gt_n.min(), rec_n.max() - rec_n.min())
        if data_range < 1e-15:
            data_range = 1.0
        ssim_val = ssim(gt_n, rec_n, data_range=data_range)
        
        # CC
        gt_z = gt_n - gt_n.mean()
        rec_z = rec_n - rec_n.mean()
        denom = np.sqrt(np.sum(gt_z**2) * np.sum(rec_z**2))
        cc = np.sum(gt_z * rec_z) / denom if denom > 1e-15 else 0.0
        
        return float(psnr), float(ssim_val), float(cc)
    
    # Compute metrics per component
    comp_metrics = {}
    all_psnr, all_ssim, all_cc = [], [], []
    
    for name, gt, rec in [("εxx", gt_exx, rec_exx),
                          ("εxy", gt_exy, rec_exy),
                          ("εyy", gt_eyy, rec_eyy)]:
        p, s, c = compute_field_metrics(gt, rec)
        comp_metrics[name] = {"PSNR": p, "SSIM": s, "CC": c}
        all_psnr.append(p)
        all_ssim.append(s)
        all_cc.append(c)
        print(f"    {name}: PSNR={p:.2f}, SSIM={s:.4f}, CC={c:.4f}")
    
    avg_psnr = float(np.mean(all_psnr))
    avg_ssim = float(np.mean(all_ssim))
    avg_cc = float(np.mean(all_cc))
    print(f"\n    Average: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, CC={avg_cc:.4f}")
    
    metrics = {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "CC": avg_cc,
        "components": comp_metrics,
    }
    
    # Save outputs
    gt_all = np.stack([gt_exx, gt_exy, gt_eyy])
    rec_all = np.stack([rec_exx, rec_exy, rec_eyy])
    
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_all)
        np.save(os.path.join(d, "recon_output.npy"), rec_all)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    
    components = [
        ("εxx", gt_exx, rec_exx),
        ("εxy", gt_exy, rec_exy),
        ("εyy", gt_eyy, rec_eyy),
    ]
    
    for row, (name, gt, rec) in enumerate(components):
        vmin = min(gt.min(), rec.min())
        vmax = max(gt.max(), rec.max())
        
        ax = axes[row, 0]
        im = ax.imshow(gt, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.scatter(stations[:, 0], stations[:, 1], c='k', s=10, marker='^')
        ax.set_title(f"GT {name} (nanostrain/yr)", fontsize=12)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.colorbar(im, ax=ax)
        
        ax = axes[row, 1]
        im = ax.imshow(rec, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.scatter(stations[:, 0], stations[:, 1], c='k', s=10, marker='^')
        m_comp = comp_metrics[f'{name}']
        ax.set_title(f"Reconstructed {name}\nPSNR={m_comp['PSNR']:.1f}dB, "
                     f"SSIM={m_comp['SSIM']:.3f}, CC={m_comp['CC']:.3f}", fontsize=11)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return metrics


# --- End of Referee injection ---


def main():
    data_paths = ['/data/yjh/straintool_geo_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # Classify files into outer and inner
    outer_files = []
    inner_files = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(p)
        else:
            outer_files.append(p)

    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")

    if not outer_files:
        print("ERROR: No outer (primary) data file found.")
        sys.exit(1)

    # Load outer data
    outer_path = outer_files[0]
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"Outer data keys: {list(outer_data.keys())}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    # Run agent function
    print("Running agent run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_files:
        # Pattern 2: Chained execution
        print("Detected chained execution pattern.")
        inner_path = inner_files[0]
        print(f"Loading inner data from: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)

        print("Running chained call (agent_output as operator)...")
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR in chained call: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct execution
        print("Detected direct execution pattern.")
        final_result = agent_output
        std_result = std_output

    # Now evaluate both agent and standard results
    # The evaluate_results function expects (data, inversion_result)
    # where data contains ground truth fields and inversion_result contains reconstructed fields.
    # The 'data' input is the first argument passed to run_inversion (args[0]).
    
    input_data = args[0] if args else kwargs.get('data', None)
    if input_data is None:
        print("ERROR: Could not extract input data for evaluation.")
        sys.exit(1)

    # Check that input_data has ground truth fields needed for evaluation
    required_keys = ['gt_exx', 'gt_exy', 'gt_eyy', 'grid_x', 'grid_y', 'stations', 'vx', 'vy']
    for key in required_keys:
        if key not in input_data:
            print(f"WARNING: Key '{key}' missing from input data. Evaluation may fail.")

    print("\n=== Evaluating AGENT result ===")
    try:
        metrics_agent = evaluate_results(input_data, final_result)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Evaluating STANDARD result ===")
    try:
        metrics_std = evaluate_results(input_data, std_result)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract primary scalar metrics (Higher is better for PSNR, SSIM, CC)
    score_agent_psnr = metrics_agent["PSNR"]
    score_std_psnr = metrics_std["PSNR"]
    score_agent_ssim = metrics_agent["SSIM"]
    score_std_ssim = metrics_std["SSIM"]
    score_agent_cc = metrics_agent["CC"]
    score_std_cc = metrics_std["CC"]

    print(f"\n{'='*60}")
    print(f"Scores -> Agent PSNR: {score_agent_psnr:.4f}, Standard PSNR: {score_std_psnr:.4f}")
    print(f"Scores -> Agent SSIM: {score_agent_ssim:.4f}, Standard SSIM: {score_std_ssim:.4f}")
    print(f"Scores -> Agent CC:   {score_agent_cc:.4f}, Standard CC:   {score_std_cc:.4f}")
    print(f"{'='*60}")

    # Verification: Higher is better, allow 10% margin
    margin = 0.90  # agent must be at least 90% of standard

    failed = False
    
    if score_std_psnr > 0 and score_agent_psnr < score_std_psnr * margin:
        print(f"FAIL: Agent PSNR ({score_agent_psnr:.4f}) is significantly worse than Standard ({score_std_psnr:.4f})")
        failed = True
    else:
        print(f"PASS: PSNR check OK")

    if score_std_ssim > 0 and score_agent_ssim < score_std_ssim * margin:
        print(f"FAIL: Agent SSIM ({score_agent_ssim:.4f}) is significantly worse than Standard ({score_std_ssim:.4f})")
        failed = True
    else:
        print(f"PASS: SSIM check OK")

    if score_std_cc > 0 and score_agent_cc < score_std_cc * margin:
        print(f"FAIL: Agent CC ({score_agent_cc:.4f}) is significantly worse than Standard ({score_std_cc:.4f})")
        failed = True
    else:
        print(f"PASS: CC check OK")

    if failed:
        print("\nOVERALL: FAIL - Performance degraded significantly.")
        sys.exit(1)
    else:
        print("\nOVERALL: PASS - Performance is acceptable.")
        sys.exit(0)


if __name__ == "__main__":
    main()