import sys
import os
import traceback
import numpy as np
import json

# Set matplotlib backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the working directory to path
sys.path.insert(0, '/data/yjh/pygimli_ert_sandbox_sandbox')
sys.path.insert(0, '/data/yjh/pygimli_ert_sandbox_sandbox/run_code')

from scipy.interpolate import griddata
from pygimli.physics import ert


# ============================================================
# INJECTED REFEREE: evaluate_results (verbatim from Reference B)
# ============================================================

def compute_psnr(gt, recon):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float('inf')
    data_range = np.max(gt) - np.min(gt)
    return 20.0 * np.log10(data_range / np.sqrt(mse))

def compute_ssim(gt, recon):
    """Compute SSIM between two 1D arrays (flattened cell-based)."""
    try:
        from skimage.metrics import structural_similarity
        n = len(gt)
        side = int(np.ceil(np.sqrt(n)))
        gt_mean = np.mean(gt)
        recon_mean = np.mean(recon)
        gt_pad = np.full(side * side, gt_mean)
        recon_pad = np.full(side * side, recon_mean)
        gt_pad[:n] = gt
        recon_pad[:n] = recon
        gt_2d = gt_pad.reshape(side, side)
        recon_2d = recon_pad.reshape(side, side)
        data_range = np.max(gt) - np.min(gt)
        win_size = min(7, side)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
        return structural_similarity(gt_2d, recon_2d, data_range=data_range,
                                     win_size=win_size)
    except Exception as e:
        print(f"SSIM computation warning: {e}")
        gt_norm = (gt - np.mean(gt)) / (np.std(gt) + 1e-10)
        recon_norm = (recon - np.mean(recon)) / (np.std(recon) + 1e-10)
        return float(np.mean(gt_norm * recon_norm))

def compute_rmse(gt, recon):
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((gt - recon) ** 2)))

def interpolate_to_grid(mesh, cell_values, x_range, y_range, nx=100, ny=50):
    """Interpolate cell-based values to a regular grid for visualization."""
    cell_centers = np.array([[mesh.cell(i).center().x(),
                              mesh.cell(i).center().y()]
                             for i in range(mesh.cellCount())])

    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x_coords, y_coords)

    grid_values = griddata(cell_centers, np.array(cell_values),
                           (xx, yy), method='linear', fill_value=np.nan)

    return grid_values, x_coords, y_coords

def evaluate_results(inversion_data):
    """
    Compute metrics and create visualization.
    """
    print("[5/6] Computing metrics...")

    mesh = inversion_data['mesh']
    scheme = inversion_data['scheme']
    data = inversion_data['data']
    gt_res_np = inversion_data['gt_res_np']
    inv_model_pd = inversion_data['inv_model_pd']
    pd = inversion_data['pd']
    chi2 = inversion_data['chi2']
    results_dir = inversion_data.get('results_dir', '/data/yjh/pygimli_ert_sandbox_sandbox/results')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Get para-domain cell centers
    pd_centers = np.array([[pd.cell(i).center().x(),
                            pd.cell(i).center().y()]
                           for i in range(pd.cellCount())])

    # Get forward mesh cell centers
    fwd_centers = np.array([[mesh.cell(i).center().x(),
                             mesh.cell(i).center().y()]
                            for i in range(mesh.cellCount())])

    # Interpolate GT onto para-domain cell centers
    gt_on_pd = griddata(fwd_centers, gt_res_np, pd_centers, method='nearest')
    inv_on_pd = np.array(inv_model_pd)

    # Work in log space
    gt_log_cells = np.log10(gt_on_pd)
    inv_log_cells = np.log10(inv_on_pd)

    psnr_val = compute_psnr(gt_log_cells, inv_log_cells)
    rmse_val = compute_rmse(gt_log_cells, inv_log_cells)

    psnr_lin = compute_psnr(gt_on_pd, inv_on_pd)
    rmse_lin = compute_rmse(gt_on_pd, inv_on_pd)

    # Grid interpolation for visualization
    xmin, xmax = pd.xmin(), pd.xmax()
    ymin, ymax = pd.ymin(), pd.ymax()
    nx, ny = 120, 60

    gt_grid, x_coords, y_coords = interpolate_to_grid(
        mesh, gt_res_np, (xmin, xmax), (ymin, ymax), nx, ny)

    inv_grid, _, _ = interpolate_to_grid(
        pd, np.array(inv_model_pd), (xmin, xmax), (ymin, ymax), nx, ny)

    # Compute SSIM on the 2D grid
    gt_grid_log = np.log10(np.where(np.isnan(gt_grid), 1.0, gt_grid))
    inv_grid_log = np.log10(np.where(np.isnan(inv_grid), 1.0, inv_grid))
    valid_mask = ~(np.isnan(gt_grid) | np.isnan(inv_grid))
    gt_grid_log_filled = gt_grid_log.copy()
    inv_grid_log_filled = inv_grid_log.copy()
    gt_grid_log_filled[~valid_mask] = np.mean(gt_grid_log[valid_mask])
    inv_grid_log_filled[~valid_mask] = np.mean(inv_grid_log[valid_mask])

    try:
        from skimage.metrics import structural_similarity
        data_range = np.max(gt_grid_log[valid_mask]) - np.min(gt_grid_log[valid_mask])
        ssim_val = structural_similarity(gt_grid_log_filled, inv_grid_log_filled,
                                         data_range=data_range, win_size=7)
    except Exception as e:
        print(f"   SSIM grid computation warning: {e}")
        ssim_val = compute_ssim(gt_log_cells, inv_log_cells)

    print(f"\n   === Metrics (log10 resistivity) ===")
    print(f"   PSNR  = {psnr_val:.2f} dB")
    print(f"   SSIM  = {ssim_val:.4f}")
    print(f"   RMSE  = {rmse_val:.4f}")
    print(f"\n   === Metrics (linear resistivity) ===")
    print(f"   PSNR  = {psnr_lin:.2f} dB")
    print(f"   RMSE  = {rmse_lin:.2f} Ohm·m")
    print(f"   Chi²  = {chi2:.3f}")

    # Save metrics
    metrics = {
        "PSNR_log_dB": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "RMSE_log": round(rmse_val, 4),
        "PSNR_linear_dB": round(psnr_lin, 2),
        "RMSE_linear_ohm_m": round(rmse_lin, 2),
        "chi2": round(chi2, 3),
        "num_electrodes": scheme.sensorCount(),
        "num_measurements": data.size(),
        "method": "Gauss-Newton with smoothness regularization",
        "scheme": "Dipole-dipole",
        "lambda": 1,
        "noise_level": 0.03,
    }

    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   Metrics saved to {metrics_path}")

    # Save numpy arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_grid)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), inv_grid)
    print(f"   Arrays saved to {results_dir}/")

    sandbox_dir = '/data/yjh/pygimli_ert_sandbox_sandbox'
    np.save(os.path.join(sandbox_dir, 'gt_output.npy'), gt_grid)
    np.save(os.path.join(sandbox_dir, 'recon_output.npy'), inv_grid)
    with open(os.path.join(sandbox_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Visualization
    print("[6/6] Creating visualization...")

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    vmin, vmax = 5, 500
    cmap = 'Spectral_r'

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(x_coords, y_coords, gt_grid,
                         norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap=cmap, shading='auto')
    ax1.set_title('(a) Ground Truth Resistivity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Resistivity (Ω·m)')

    ax2 = fig.add_subplot(gs[0, 1])
    try:
        ert.showERTData(data, ax=ax2, cMap=cmap)
        ax2.set_title('(b) Apparent Resistivity Pseudosection', fontsize=13,
                      fontweight='bold')
    except Exception as e:
        print(f"   Pseudosection plot warning: {e}")
        ax2.set_title('(b) Apparent Resistivity Data', fontsize=13, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.pcolormesh(x_coords, y_coords, inv_grid,
                         norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap=cmap, shading='auto')
    ax3.set_title('(c) Reconstructed Resistivity', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='Resistivity (Ω·m)')

    ax4 = fig.add_subplot(gs[1, 1])
    error_map = np.abs(np.log10(inv_grid) - np.log10(gt_grid))
    im4 = ax4.pcolormesh(x_coords, y_coords, error_map,
                         cmap='hot_r', shading='auto', vmin=0, vmax=1.5)
    ax4.set_title('(d) Log₁₀ Absolute Error', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='|log₁₀(recon) - log₁₀(GT)|')

    ax5 = fig.add_subplot(gs[2, 0])
    mid_y_idx = ny // 3
    depth_val = y_coords[mid_y_idx]
    ax5.semilogy(x_coords, gt_grid[mid_y_idx, :], 'b-', linewidth=2, label='Ground Truth')
    ax5.semilogy(x_coords, inv_grid[mid_y_idx, :], 'r--', linewidth=2, label='Reconstruction')
    ax5.set_title(f'(e) Horizontal Profile at depth={depth_val:.1f}m', fontsize=13,
                  fontweight='bold')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('Resistivity (Ω·m)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_text = (
        f"ERT Inversion Benchmark\n"
        f"{'─' * 35}\n"
        f"Method: Gauss-Newton + smoothness reg.\n"
        f"Scheme: Dipole-dipole, {scheme.sensorCount()} electrodes\n"
        f"Measurements: {data.size()}\n"
        f"Noise: 3% + 1µV\n"
        f"Lambda (regularization): 1\n"
        f"{'─' * 35}\n"
        f"PSNR (log₁₀):  {psnr_val:.2f} dB\n"
        f"SSIM (log₁₀):  {ssim_val:.4f}\n"
        f"RMSE (log₁₀):  {rmse_val:.4f}\n"
        f"PSNR (linear):  {psnr_lin:.2f} dB\n"
        f"RMSE (linear):  {rmse_lin:.2f} Ω·m\n"
        f"Chi²:  {chi2:.3f}\n"
    )
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('pyGIMLi ERT: Subsurface Resistivity Reconstruction',
                 fontsize=15, fontweight='bold', y=0.98)

    fig_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    fig_path2 = os.path.join(sandbox_dir, 'vis_result.png')
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Figure saved to {fig_path}")

    print("\n" + "=" * 60)
    print("DONE. All outputs saved to:", results_dir)
    print("=" * 60)

    return metrics


# ============================================================
# Helper: Try to load pkl with multiple strategies
# ============================================================

def try_load_pkl(filepath):
    """Try multiple strategies to load a pickle file."""
    import pickle
    
    filesize = os.path.getsize(filepath)
    print(f"  File size: {filesize} bytes")
    
    if filesize == 0:
        print(f"  WARNING: File is empty (0 bytes)")
        return None
    
    # Strategy 1: dill
    try:
        import dill
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print(f"  Loaded successfully with dill")
        return data
    except Exception as e:
        print(f"  dill failed: {e}")
    
    # Strategy 2: pickle
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded successfully with pickle")
        return data
    except Exception as e:
        print(f"  pickle failed: {e}")
    
    # Strategy 3: dill with different protocols
    try:
        import dill
        with open(filepath, 'rb') as f:
            unpickler = dill.Unpickler(f)
            data = unpickler.load()
        print(f"  Loaded successfully with dill.Unpickler")
        return data
    except Exception as e:
        print(f"  dill.Unpickler failed: {e}")
    
    # Strategy 4: Try reading partial data
    try:
        import dill
        with open(filepath, 'rb') as f:
            content = f.read()
        print(f"  Read {len(content)} bytes from file")
        # Try loading from bytes
        data = dill.loads(content)
        print(f"  Loaded successfully from bytes with dill")
        return data
    except Exception as e:
        print(f"  dill.loads from bytes failed: {e}")
    
    return None


# ============================================================
# Helper: Reconstruct forward_data from the pipeline scripts
# ============================================================

def reconstruct_forward_data():
    """
    Try to reconstruct the forward_data by running the pipeline from scratch,
    using the original pipeline scripts if available.
    """
    print("Attempting to reconstruct forward_data from pipeline...")
    
    # Look for pipeline scripts
    run_code_dir = '/data/yjh/pygimli_ert_sandbox_sandbox/run_code'
    sandbox_dir = '/data/yjh/pygimli_ert_sandbox_sandbox'
    
    # Try to find and run the full pipeline
    # Look for the original pipeline modules
    possible_locations = [
        run_code_dir,
        sandbox_dir,
        os.path.join(sandbox_dir, 'src'),
        '/data/yjh/pygimli_ert_sandbox_sandbox',
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            sys.path.insert(0, loc)
    
    # Try importing or finding the pipeline steps
    # Step 1: create_geometry/mesh
    # Step 2: create_model  
    # Step 3: forward simulation
    # These are typically defined in the pipeline
    
    # Look for any python files that might contain the pipeline
    pipeline_files = []
    for search_dir in [run_code_dir, sandbox_dir]:
        if os.path.exists(search_dir):
            for f in os.listdir(search_dir):
                if f.endswith('.py') and f not in ['test_run_inversion.py', 'agent_run_inversion.py']:
                    pipeline_files.append(os.path.join(search_dir, f))
    
    print(f"  Found pipeline files: {pipeline_files}")
    
    # Try to find pkl files from earlier pipeline steps
    std_data_dir = '/data/yjh/pygimli_ert_sandbox_sandbox/run_code/std_data'
    if os.path.exists(std_data_dir):
        all_pkl_files = [f for f in os.listdir(std_data_dir) if f.endswith('.pkl')]
        print(f"  All pkl files in std_data: {all_pkl_files}")
        
        # Look for forward_operator or earlier step output files
        for pkl_name in all_pkl_files:
            if 'run_inversion' not in pkl_name:
                pkl_path = os.path.join(std_data_dir, pkl_name)
                print(f"  Trying to load: {pkl_path}")
                data = try_load_pkl(pkl_path)
                if data is not None:
                    print(f"  Successfully loaded: {pkl_name}")
                    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                    # This might be the output of a prior step that is the input to run_inversion
                    if isinstance(data, dict) and 'output' in data:
                        output = data['output']
                        if isinstance(output, dict) and 'data' in output:
                            print(f"  Found forward_data candidate in {pkl_name}")
                            return output
    
    # If we can't find prior step data, try to run the full pipeline
    # This is the pyGIMLi ERT standard pipeline
    try:
        return _run_ert_pipeline()
    except Exception as e:
        print(f"  Pipeline reconstruction failed: {e}")
        traceback.print_exc()
    
    return None


def _run_ert_pipeline():
    """Run the standard pyGIMLi ERT pipeline to generate forward_data."""
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    
    print("  Running standard ERT pipeline...")
    
    # Step 1: Create geometry and mesh
    print("  [1/3] Creating geometry...")
    world = mt.createWorld(start=[-50, 0], end=[50, -50],
                           layers=[-1, -5], worldMarker=True)
    
    # Create scheme (dipole-dipole)
    scheme = ert.createData(elecs=np.linspace(start=-25, stop=25, num=51),
                           schemeName='dd')
    
    # Create mesh for forward modeling  
    for p in scheme.sensors():
        world.createNode(p)
        world.createNode(p + pg.RVector3(0, -0.1))
    
    mesh = mt.createMesh(world, quality=34, area=1.0)
    print(f"    Mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes")
    
    # Step 2: Create resistivity model
    print("  [2/3] Creating resistivity model...")
    rhomap = [
        [1, 100.0],    # background
        [2, 50.0],     # layer 1
        [3, 200.0],    # layer 2
    ]
    
    # Assign resistivities based on markers
    gt_res = pg.solver.parseArgToArray(rhomap, mesh.cellCount(), mesh)
    gt_res_np = np.array(gt_res)
    
    print(f"    Model: {len(gt_res_np)} cells")
    print(f"    Resistivity range: [{min(gt_res_np):.1f}, {max(gt_res_np):.1f}] Ohm·m")
    
    # Step 3: Forward simulation
    print("  [3/3] Running forward simulation...")
    mgr_fwd = ert.ERTManager()
    data = ert.simulate(mesh, scheme=scheme, res=rhomap,
                       noiseLevel=0.03, noiseAbs=1e-6, seed=42)
    
    # Ensure data has proper error
    if not data.haveData('err'):
        data['err'] = ert.estimateError(data, relativeError=0.03, absoluteUError=1e-6)
    
    results_dir = '/data/yjh/pygimli_ert_sandbox_sandbox/results'
    os.makedirs(results_dir, exist_ok=True)
    
    forward_data = {
        'mesh': mesh,
        'scheme': scheme,
        'data': data,
        'gt_res_np': gt_res_np,
        'rhomap': rhomap,
        'results_dir': results_dir,
    }
    
    print(f"    Forward data keys: {list(forward_data.keys())}")
    print(f"    Data size: {data.size()} measurements")
    
    return forward_data


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("=" * 60)
    print("QA Test for run_inversion")
    print("=" * 60)
    
    data_paths = ['/data/yjh/pygimli_ert_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data paths
    outer_paths = []
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_paths.append(p)
    
    print(f"Outer data paths: {outer_paths}")
    print(f"Inner data paths: {inner_paths}")
    
    # Try to load outer data
    forward_data = None
    std_output = None
    
    for outer_path in outer_paths:
        print(f"\nAttempting to load: {outer_path}")
        if os.path.exists(outer_path):
            loaded = try_load_pkl(outer_path)
            if loaded is not None:
                print(f"  Loaded data type: {type(loaded)}")
                if isinstance(loaded, dict):
                    print(f"  Keys: {list(loaded.keys())}")
                    if 'args' in loaded and 'kwargs' in loaded:
                        # Standard capture format
                        args = loaded.get('args', ())
                        kwargs = loaded.get('kwargs', {})
                        std_output = loaded.get('output', None)
                        if len(args) > 0:
                            forward_data = args[0]
                        elif kwargs:
                            forward_data = list(kwargs.values())[0] if kwargs else None
                    elif 'data' in loaded:
                        # Might be the forward_data directly
                        forward_data = loaded
                    elif 'output' in loaded:
                        std_output = loaded['output']
                else:
                    print(f"  Unexpected data type: {type(loaded)}")
        else:
            print(f"  File does not exist: {outer_path}")
    
    # If loading failed, reconstruct the data
    if forward_data is None:
        print("\n" + "=" * 60)
        print("Could not load pkl data. Reconstructing forward_data from pipeline...")
        print("=" * 60)
        forward_data = reconstruct_forward_data()
    
    if forward_data is None:
        print("FATAL: Could not obtain forward_data by any means.")
        sys.exit(1)
    
    print(f"\nForward data obtained. Type: {type(forward_data)}")
    if isinstance(forward_data, dict):
        print(f"Keys: {list(forward_data.keys())}")
    
    # Ensure results_dir exists
    if isinstance(forward_data, dict) and 'results_dir' not in forward_data:
        forward_data['results_dir'] = '/data/yjh/pygimli_ert_sandbox_sandbox/results'
    
    results_dir = forward_data.get('results_dir', '/data/yjh/pygimli_ert_sandbox_sandbox/results')
    os.makedirs(results_dir, exist_ok=True)
    
    # ============================================================
    # Run agent's run_inversion
    # ============================================================
    print("\n" + "=" * 60)
    print("Running AGENT's run_inversion...")
    print("=" * 60)
    
    try:
        from agent_run_inversion import run_inversion as agent_run_inversion
        agent_output = agent_run_inversion(forward_data)
        print("Agent run_inversion completed successfully.")
    except Exception as e:
        print(f"FATAL: Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================
    # Run standard run_inversion (if we don't have std_output)
    # ============================================================
    if std_output is None:
        print("\n" + "=" * 60)
        print("Running STANDARD run_inversion for comparison...")
        print("=" * 60)
        
        try:
            # Re-run forward to get fresh data for standard (avoid shared state)
            std_forward_data = reconstruct_forward_data()
            if std_forward_data is None:
                # Use the same forward_data but make a copy
                std_forward_data = forward_data.copy()
            
            if 'results_dir' not in std_forward_data:
                std_forward_data['results_dir'] = results_dir
            
            # Import and run the standard implementation directly
            # (same code as reference)
            def standard_run_inversion(fwd_data):
                print("[4/6] Running ERT inversion (Gauss-Newton, lambda=1)...")
                data = fwd_data['data']
                mgr = ert.ERTManager(data)
                inv_model = mgr.invert(lam=1, zWeight=0.3, verbose=True)
                chi2 = mgr.inv.chi2()
                print(f"   Inversion chi² = {chi2:.3f}")
                print(f"   Inversion model size: {len(inv_model)}")
                pd = mgr.paraDomain
                inv_model_pd = mgr.paraModel(inv_model)
                print(f"   Para domain: {pd.cellCount()} cells")
                print(f"   Inverted resistivity range: [{min(inv_model_pd):.2f}, {max(inv_model_pd):.2f}] Ohm·m")
                result = fwd_data.copy()
                result['inv_model'] = inv_model
                result['inv_model_pd'] = inv_model_pd
                result['pd'] = pd
                result['mgr'] = mgr
                result['chi2'] = chi2
                return result
            
            std_output = standard_run_inversion(std_forward_data)
            print("Standard run_inversion completed successfully.")
        except Exception as e:
            print(f"WARNING: Standard run_inversion failed: {e}")
            traceback.print_exc()
            std_output = None
    
    # ============================================================
    # Evaluate both outputs
    # ============================================================
    print("\n" + "=" * 60)
    print("Evaluating AGENT output...")
    print("=" * 60)
    
    try:
        score_agent = evaluate_results(agent_output)
        print(f"\nAgent metrics: {json.dumps(score_agent, indent=2)}")
    except Exception as e:
        print(f"FATAL: evaluate_results failed on agent output: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if std_output is not None:
        print("\n" + "=" * 60)
        print("Evaluating STANDARD output...")
        print("=" * 60)
        
        try:
            score_std = evaluate_results(std_output)
            print(f"\nStandard metrics: {json.dumps(score_std, indent=2)}")
        except Exception as e:
            print(f"WARNING: evaluate_results failed on standard output: {e}")
            traceback.print_exc()
            score_std = None
    else:
        score_std = None
    
    # ============================================================
    # Compare and decide
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if score_std is not None:
        # Extract key metrics for comparison
        agent_psnr = score_agent.get('PSNR_log_dB', 0)
        std_psnr = score_std.get('PSNR_log_dB', 0)
        agent_ssim = score_agent.get('SSIM', 0)
        std_ssim = score_std.get('SSIM', 0)
        agent_rmse = score_agent.get('RMSE_log', float('inf'))
        std_rmse = score_std.get('RMSE_log', float('inf'))
        agent_chi2 = score_agent.get('chi2', float('inf'))
        std_chi2 = score_std.get('chi2', float('inf'))
        
        print(f"\n  {'Metric':<20} {'Agent':>12} {'Standard':>12} {'Status':>10}")
        print(f"  {'-'*54}")
        print(f"  {'PSNR (log dB)':<20} {agent_psnr:>12.2f} {std_psnr:>12.2f}", end="")
        psnr_ok = agent_psnr >= std_psnr * 0.9
        print(f" {'  OK' if psnr_ok else '  FAIL':>10}")
        
        print(f"  {'SSIM':<20} {agent_ssim:>12.4f} {std_ssim:>12.4f}", end="")
        ssim_ok = agent_ssim >= std_ssim * 0.9
        print(f" {'  OK' if ssim_ok else '  FAIL':>10}")
        
        print(f"  {'RMSE (log)':<20} {agent_rmse:>12.4f} {std_rmse:>12.4f}", end="")
        rmse_ok = agent_rmse <= std_rmse * 1.1
        print(f" {'  OK' if rmse_ok else '  FAIL':>10}")
        
        print(f"  {'Chi²':<20} {agent_chi2:>12.3f} {std_chi2:>12.3f}", end="")
        # Chi2 should be close to 1; allow reasonable range
        chi2_ok = abs(agent_chi2 - std_chi2) < max(std_chi2 * 0.5, 1.0)
        print(f" {'  OK' if chi2_ok else '  WARN':>10}")
        
        print(f"\nScores -> Agent PSNR: {agent_psnr:.2f}, Standard PSNR: {std_psnr:.2f}")
        print(f"Scores -> Agent SSIM: {agent_ssim:.4f}, Standard SSIM: {std_ssim:.4f}")
        
        # Overall pass: PSNR and SSIM should be within 10% of standard
        # RMSE should not be more than 10% worse
        passed = psnr_ok and ssim_ok and rmse_ok
        
        if passed:
            print("\n✓ PASSED: Agent performance is within acceptable range of standard.")
            sys.exit(0)
        else:
            print("\n✗ FAILED: Agent performance degraded significantly.")
            if not psnr_ok:
                print(f"  - PSNR too low: {agent_psnr:.2f} < {std_psnr * 0.9:.2f}")
            if not ssim_ok:
                print(f"  - SSIM too low: {agent_ssim:.4f} < {std_ssim * 0.9:.4f}")
            if not rmse_ok:
                print(f"  - RMSE too high: {agent_rmse:.4f} > {std_rmse * 1.1:.4f}")
            sys.exit(1)
    else:
        # No standard to compare against - check basic sanity
        print("\nNo standard output available for comparison.")
        print("Performing sanity checks on agent output...")
        
        agent_psnr = score_agent.get('PSNR_log_dB', 0)
        agent_ssim = score_agent.get('SSIM', 0)
        agent_rmse = score_agent.get('RMSE_log', float('inf'))
        agent_chi2 = score_agent.get('chi2', float('inf'))
        
        print(f"  PSNR (log): {agent_psnr:.2f} dB")
        print(f"  SSIM: {agent_ssim:.4f}")
        print(f"  RMSE (log): {agent_rmse:.4f}")
        print(f"  Chi²: {agent_chi2:.3f}")
        
        # Basic sanity: PSNR should be positive, SSIM > 0, RMSE finite
        sanity_ok = (agent_psnr > 5.0 and agent_ssim > 0.3 and 
                     agent_rmse < 2.0 and np.isfinite(agent_chi2))
        
        if sanity_ok:
            print("\n✓ PASSED: Agent output passes sanity checks.")
            sys.exit(0)
        else:
            print("\n✗ FAILED: Agent output failed sanity checks.")
            sys.exit(1)


if __name__ == '__main__':
    main()