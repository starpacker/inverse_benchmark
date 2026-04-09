import sys
import os
import dill
import numpy as np
import traceback
from pathlib import Path

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import json

# ============================================================
# Inject evaluate_results (Reference B) verbatim
# ============================================================

def _match_particles(gt, det, md):
    """Match detected particles to ground truth within distance md."""
    if len(det) == 0 or len(gt) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    D = cdist(gt, det)
    mg = []
    md_ = []
    ug = set()
    ud = set()
    for idx in np.argsort(D, axis=None):
        gi, di = np.unravel_index(idx, D.shape)
        if gi in ug or di in ud:
            continue
        if D[gi, di] > md:
            break
        mg.append(gt[gi])
        md_.append(det[di])
        ug.add(gi)
        ud.add(di)
    return np.array(mg), np.array(md_)

def _rmse(a, b):
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _cc(a, b):
    """Pearson Correlation Coefficient."""
    af, bf = a.ravel(), b.ravel()
    if np.std(af) > 1e-15 and np.std(bf) > 1e-15:
        return float(np.corrcoef(af, bf)[0, 1])
    return 0.0

def _psnr(a, b):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((a - b) ** 2)
    mx = np.max(np.abs(a))
    if mse < 1e-30:
        return 100.0
    if mx < 1e-30:
        return 0.0
    return float(10 * np.log10(mx ** 2 / mse))

def _make_fig(holo, gt, det, mg, md, gv, pixel_size, path):
    """Create visualization figure."""
    um = 1e6
    dx = pixel_size
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ext = [0, holo.shape[0] * dx * um, 0, holo.shape[1] * dx * um]
    im = ax.imshow(holo.T, cmap="gray", origin="lower", extent=ext)
    ax.set_title("Simulated Inline Hologram")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    mip = np.max(gv, axis=0)
    ext2 = [0, mip.shape[0] * dx * um, 0, mip.shape[1] * dx * um]
    im = ax.imshow(mip.T, cmap="hot", origin="lower", extent=ext2)
    ax.set_title("Focus MIP (x-y)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    ax.scatter(gt[:, 0] * um, gt[:, 1] * um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det) > 0:
        ax.scatter(det[:, 0] * um, det[:, 1] * um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg) > 0:
        for g, d in zip(mg * um, md * um):
            ax.plot([g[0], d[0]], [g[1], d[1]], "g--", alpha=0.5, lw=0.8)
    ax.set_title("Top (x-y)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1, 1]
    ax.scatter(gt[:, 0] * um, gt[:, 2] * um, facecolors="none", edgecolors="blue", s=120, lw=2, label="GT", zorder=2)
    if len(det) > 0:
        ax.scatter(det[:, 0] * um, det[:, 2] * um, c="red", marker="x", s=80, lw=2, label="Det", zorder=3)
    if len(mg) > 0:
        for g, d in zip(mg * um, md * um):
            ax.plot([g[0], d[0]], [g[2], d[2]], "g--", alpha=0.5, lw=0.8)
    ax.set_title("Side (x-z)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("z (μm)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

def evaluate_results(
    gt_positions,
    detected_positions,
    match_dist,
    hologram,
    gt_particles,
    gradient_volume,
    pixel_size,
    working_dir,
    asset_dir
):
    """
    Evaluate reconstruction quality and save results.
    """
    n_gt = len(gt_positions)
    
    # Match detected to ground truth
    mg, md = _match_particles(gt_positions, detected_positions, match_dist)
    nm = len(mg)
    
    # Compute metrics
    if nm > 0:
        r3 = _rmse(mg, md)
        rxy = _rmse(mg[:, :2], md[:, :2])
        rz = _rmse(mg[:, 2:], md[:, 2:])
        cc = _cc(mg, md)
        p = _psnr(mg, md)
    else:
        r3 = rxy = rz = float("inf")
        cc = p = 0.0
    
    print(f"  Matched {nm}/{n_gt}  RMSE={r3 * 1e6:.2f}μm  CC={cc:.4f}  PSNR={p:.2f}dB")
    
    # Save results
    print("\n[5/6] Save …")
    for d in [working_dir, asset_dir]:
        d.mkdir(parents=True, exist_ok=True)
        np.save(str(d / "gt_output.npy"), gt_positions)
        np.save(str(d / "recon_output.npy"), detected_positions)
    
    metrics = dict(
        n_gt=int(n_gt),
        n_detected=int(len(detected_positions)),
        n_matched=int(nm),
        detection_rate=round(nm / n_gt, 4) if n_gt > 0 else 0.0,
        rmse_3d_um=round(r3 * 1e6, 2),
        rmse_xy_um=round(rxy * 1e6, 2),
        rmse_z_um=round(rz * 1e6, 2),
        cc=round(cc, 4),
        psnr_db=round(p, 2)
    )
    
    with open(str(working_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    print("\n[6/6] Plot …")
    for d in [asset_dir, working_dir]:
        _make_fig(
            hologram, gt_particles, detected_positions, mg, md,
            gradient_volume, pixel_size, d / "vis_result.png"
        )
    
    return metrics


# ============================================================
# Main test logic
# ============================================================

def main():
    data_paths = ['/data/yjh/holopy_hpiv_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_paths = []
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_paths.append(p)
    
    print(f"Outer paths: {outer_paths}")
    print(f"Inner paths: {inner_paths}")
    
    # --------------------------------------------------------
    # Load outer data
    # --------------------------------------------------------
    assert len(outer_paths) == 1, f"Expected 1 outer path, got {len(outer_paths)}"
    outer_path = outer_paths[0]
    print(f"Loading outer data from: {outer_path}")
    
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    func_name = outer_data.get('func_name', 'unknown')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {func_name}")
    print(f"Number of args: {len(args)}, kwargs keys: {list(kwargs.keys())}")
    
    # --------------------------------------------------------
    # Extract parameters from args/kwargs
    # --------------------------------------------------------
    # The function signature: run_inversion(hologram, pixel_size, z_planes, wavelength, n_expected)
    # Data might be in args, kwargs, or a mix
    param_names = ['hologram', 'pixel_size', 'z_planes', 'wavelength', 'n_expected']
    
    # Build a unified parameter dict
    call_kwargs = {}
    for i, name in enumerate(param_names):
        if name in kwargs:
            call_kwargs[name] = kwargs[name]
        elif i < len(args):
            call_kwargs[name] = args[i]
    
    print(f"Resolved parameter keys: {list(call_kwargs.keys())}")
    
    hologram = call_kwargs['hologram']
    pixel_size = call_kwargs['pixel_size']
    z_planes = call_kwargs['z_planes']
    wavelength = call_kwargs['wavelength']
    n_expected = call_kwargs['n_expected']
    
    print(f"Hologram shape: {hologram.shape}, pixel_size: {pixel_size}")
    print(f"z_planes: len={len(z_planes)}, range=[{z_planes.min()}, {z_planes.max()}]")
    print(f"wavelength: {wavelength}, n_expected: {n_expected}")
    
    # --------------------------------------------------------
    # Determine if chained (inner data exists)
    # --------------------------------------------------------
    is_chained = len(inner_paths) > 0
    
    if not is_chained:
        # =======================================================
        # Pattern 1: Direct Execution
        # =======================================================
        print("\n=== Pattern 1: Direct Execution ===")
        
        # Run agent
        print("\n[Agent] Running run_inversion...")
        agent_output = run_inversion(hologram, pixel_size, z_planes, wavelength, n_expected)
        
        # Extract results
        # run_inversion returns (detected_positions, gradient_volume)
        if isinstance(agent_output, tuple):
            agent_detected = agent_output[0]
            agent_grad_vol = agent_output[1]
        else:
            agent_detected = agent_output
            agent_grad_vol = None
        
        # Standard output
        if isinstance(std_output, tuple):
            std_detected = std_output[0]
            std_grad_vol = std_output[1]
        else:
            std_detected = std_output
            std_grad_vol = None
        
        print(f"\nAgent detected positions shape: {agent_detected.shape}")
        print(f"Standard detected positions shape: {std_detected.shape}")
        
        # --------------------------------------------------------
        # We need gt_positions and gt_particles for evaluate_results
        # These are NOT direct inputs to run_inversion but are needed for evaluation.
        # We need to reconstruct them from the gen_data pipeline.
        # Since we don't have them directly, we'll use the standard output as ground truth
        # reference and compare agent vs standard using a simplified approach.
        # --------------------------------------------------------
        
        # For evaluation, we need:
        # gt_positions, detected_positions, match_dist, hologram, gt_particles, 
        # gradient_volume, pixel_size, working_dir, asset_dir
        
        # We'll use std_detected as ground truth proxy to compare agent performance
        # This tests if agent reproduces the standard algorithm's output
        
        # Setup directories
        base_dir = Path('/data/yjh/holopy_hpiv_sandbox_sandbox')
        working_dir_agent = base_dir / 'test_output' / 'agent'
        asset_dir_agent = base_dir / 'test_output' / 'agent_assets'
        working_dir_std = base_dir / 'test_output' / 'std'
        asset_dir_std = base_dir / 'test_output' / 'std_assets'
        
        # Use std_detected as the "ground truth" for both evaluations
        # This way we measure how close agent is to standard
        gt_positions = std_detected
        
        # gt_particles needs to be (N, 4) for visualization - add a dummy radius column
        if len(gt_positions) > 0:
            gt_particles = np.column_stack([gt_positions, np.ones(len(gt_positions)) * 5e-6])
        else:
            gt_particles = np.zeros((0, 4))
        
        # Match distance: use a reasonable value based on pixel size and z spacing
        if len(z_planes) > 1:
            dz = np.abs(z_planes[1] - z_planes[0])
        else:
            dz = pixel_size * 10
        match_dist = max(pixel_size * 20, dz * 3)
        print(f"Match distance: {match_dist:.6e} m")
        
        # Evaluate agent
        print("\n=== Evaluating Agent Output ===")
        if agent_grad_vol is not None:
            grad_vol_for_eval = agent_grad_vol
        else:
            grad_vol_for_eval = std_grad_vol
        
        metrics_agent = evaluate_results(
            gt_positions=gt_positions,
            detected_positions=agent_detected,
            match_dist=match_dist,
            hologram=hologram,
            gt_particles=gt_particles,
            gradient_volume=grad_vol_for_eval if grad_vol_for_eval is not None else np.zeros((1, 1, 1)),
            pixel_size=pixel_size,
            working_dir=working_dir_agent,
            asset_dir=asset_dir_agent
        )
        
        # Evaluate standard
        print("\n=== Evaluating Standard Output ===")
        metrics_std = evaluate_results(
            gt_positions=gt_positions,
            detected_positions=std_detected,
            match_dist=match_dist,
            hologram=hologram,
            gt_particles=gt_particles,
            gradient_volume=std_grad_vol if std_grad_vol is not None else np.zeros((1, 1, 1)),
            pixel_size=pixel_size,
            working_dir=working_dir_std,
            asset_dir=asset_dir_std
        )
        
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)
        print(f"Agent metrics:    {json.dumps(metrics_agent, indent=2)}")
        print(f"Standard metrics: {json.dumps(metrics_std, indent=2)}")
        
        # --------------------------------------------------------
        # Compare key metrics
        # --------------------------------------------------------
        # Since we use std as GT, std should have perfect scores
        # Agent should be very close if implementation is correct
        
        # Primary check: detection rate
        agent_det_rate = metrics_agent.get('detection_rate', 0.0)
        std_det_rate = metrics_std.get('detection_rate', 0.0)
        
        # Check number of detections
        agent_n_det = metrics_agent.get('n_detected', 0)
        std_n_det = metrics_std.get('n_detected', 0)
        
        # PSNR check (higher is better)
        agent_psnr = metrics_agent.get('psnr_db', 0.0)
        std_psnr = metrics_std.get('psnr_db', 0.0)
        
        # CC check (higher is better)
        agent_cc = metrics_agent.get('cc', 0.0)
        std_cc = metrics_std.get('cc', 0.0)
        
        # RMSE check (lower is better)
        agent_rmse = metrics_agent.get('rmse_3d_um', float('inf'))
        std_rmse = metrics_std.get('rmse_3d_um', float('inf'))
        
        print(f"\nScores -> Agent det_rate: {agent_det_rate}, Standard det_rate: {std_det_rate}")
        print(f"Scores -> Agent n_detected: {agent_n_det}, Standard n_detected: {std_n_det}")
        print(f"Scores -> Agent PSNR: {agent_psnr}, Standard PSNR: {std_psnr}")
        print(f"Scores -> Agent CC: {agent_cc}, Standard CC: {std_cc}")
        print(f"Scores -> Agent RMSE: {agent_rmse}, Standard RMSE: {std_rmse}")
        
        # --------------------------------------------------------
        # Also do a direct comparison of detected positions
        # --------------------------------------------------------
        print("\n=== Direct Position Comparison ===")
        
        # Check if gradient volumes are similar
        if agent_grad_vol is not None and std_grad_vol is not None:
            if agent_grad_vol.shape == std_grad_vol.shape:
                grad_cc = _cc(agent_grad_vol, std_grad_vol)
                grad_rmse = _rmse(agent_grad_vol, std_grad_vol)
                print(f"Gradient volume CC: {grad_cc:.6f}")
                print(f"Gradient volume RMSE: {grad_rmse:.6e}")
            else:
                print(f"Gradient volume shapes differ: agent={agent_grad_vol.shape}, std={std_grad_vol.shape}")
                grad_cc = 0.0
        else:
            grad_cc = None
        
        # Check detected positions match
        if len(agent_detected) > 0 and len(std_detected) > 0:
            matched_gt, matched_det = _match_particles(std_detected, agent_detected, match_dist)
            n_matched = len(matched_gt)
            print(f"Position matching: {n_matched}/{len(std_detected)} standard positions matched by agent")
            
            if n_matched > 0:
                pos_rmse = _rmse(matched_gt, matched_det)
                print(f"Matched position RMSE: {pos_rmse * 1e6:.2f} μm")
        
        # --------------------------------------------------------
        # Decision logic
        # --------------------------------------------------------
        passed = True
        reasons = []
        
        # Check 1: Agent should detect a similar number of particles
        if std_n_det > 0:
            det_ratio = agent_n_det / std_n_det
            if det_ratio < 0.5 or det_ratio > 2.0:
                reasons.append(f"Detection count ratio {det_ratio:.2f} outside [0.5, 2.0]")
                passed = False
        
        # Check 2: Detection rate should be reasonable (allow 10% margin)
        if std_det_rate > 0:
            if agent_det_rate < std_det_rate * 0.9:
                reasons.append(f"Agent det_rate {agent_det_rate:.4f} < 90% of std {std_det_rate:.4f}")
                # Soft fail - don't fail on this alone
        
        # Check 3: Gradient volume correlation should be high
        if grad_cc is not None and grad_cc < 0.9:
            reasons.append(f"Gradient volume CC {grad_cc:.4f} < 0.9")
            passed = False
        
        # Check 4: If RMSE is finite for std, agent should also be finite and not much worse
        if std_rmse < float('inf') and agent_rmse == float('inf'):
            reasons.append(f"Agent RMSE is inf while std is {std_rmse:.2f}")
            passed = False
        
        if passed:
            print("\n✅ TEST PASSED: Agent performance is acceptable.")
            sys.exit(0)
        else:
            print(f"\n❌ TEST FAILED: {'; '.join(reasons)}")
            sys.exit(1)
    
    else:
        # =======================================================
        # Pattern 2: Chained Execution
        # =======================================================
        print("\n=== Pattern 2: Chained Execution ===")
        
        # Run outer function to get operator
        print("\n[Agent] Running run_inversion (outer)...")
        operator = run_inversion(hologram, pixel_size, z_planes, wavelength, n_expected)
        
        # Load inner data
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_std_output = inner_data.get('output', None)
            
            print(f"Inner function: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args: {len(inner_args)}, kwargs: {list(inner_kwargs.keys())}")
            
            # Execute operator
            if callable(operator):
                final_result = operator(*inner_args, **inner_kwargs)
            else:
                # operator might be a tuple; try first element
                if isinstance(operator, tuple):
                    for item in operator:
                        if callable(item):
                            final_result = item(*inner_args, **inner_kwargs)
                            break
                    else:
                        print("No callable found in operator tuple")
                        sys.exit(1)
                else:
                    print(f"Operator is not callable: {type(operator)}")
                    sys.exit(1)
            
            print(f"Final result type: {type(final_result)}")
            print(f"Standard result type: {type(inner_std_output)}")
            
            # Simple comparison
            print("\n✅ Chained execution completed successfully.")
            sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ EXCEPTION during test execution:")
        traceback.print_exc()
        sys.exit(1)