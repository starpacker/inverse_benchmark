import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# IMPORT TARGET
# -------------------------------------------------------------------------
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # Attempt to add current dir to path if module not found directly
    sys.path.append(os.getcwd())
    try:
        from agent_run_inversion import run_inversion
    except ImportError:
        print("CRITICAL: Could not import 'run_inversion' from 'agent_run_inversion.py'.")
        sys.exit(1)

# -------------------------------------------------------------------------
# REFEREE (Evaluation Logic)
# -------------------------------------------------------------------------
def evaluate_results(recon_vec, mesh_truth, mesh_baseline, grid_info, out_name='reconstruction_result.png'):
    """
    Computes metrics and plots results.
    """
    grid_shape = grid_info['shape']
    xgrid = grid_info['xgrid']
    ygrid = grid_info['ygrid']
    
    # Reconstruct Image
    recon_img = recon_vec.reshape(grid_shape)
    
    # Ground Truth Image
    dmua_mesh = mesh_truth.mua - mesh_baseline.mua
    dmua_truth_grid = grid_info['mesh2grid'] @ dmua_mesh
    truth_img = dmua_truth_grid.reshape(grid_shape)
    
    # Metrics
    mse = np.mean((recon_vec - dmua_truth_grid)**2)
    max_val = np.max(np.abs(dmua_truth_grid))
    if max_val == 0: max_val = 1.0
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-16))
    
    print(f"Evaluation Metrics:")
    print(f"  MSE: {mse:.6e}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]
    
    vmax = np.max(np.abs(truth_img))
    if vmax == 0: vmax = 1.0
    
    im1 = ax1.imshow(truth_img, origin='lower', extent=extent, cmap='jet', vmin=-vmax, vmax=vmax)
    ax1.set_title('Ground Truth ($\Delta \mu_a$)')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(recon_img, origin='lower', extent=extent, cmap='jet', vmin=-vmax, vmax=vmax)
    ax2.set_title(f'Reconstruction (PSNR: {psnr:.2f} dB)')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle("NIRFASTer Reconstruction Demo")
    plt.tight_layout()
    plt.savefig(out_name)
    print(f"Plot saved to {out_name}")
    
    return mse, psnr

# -------------------------------------------------------------------------
# TEST EXECUTION LOGIC
# -------------------------------------------------------------------------

def load_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test_logic(data_paths):
    # 1. Parse Paths
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if 'standard_data_run_inversion.pkl' in p and 'parent_function' not in p:
            outer_path = p
        elif 'standard_data_parent_function_run_inversion_' in p:
            inner_path = p
            
    if not outer_path:
        print("Error: Primary data file (standard_data_run_inversion.pkl) not found in paths.")
        sys.exit(1)

    print(f"Target Data: {outer_path}")
    if inner_path:
        print(f"Inner Data: {inner_path}")

    # 2. Load and Execute Outer
    try:
        outer_data = load_pkl(outer_path)
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')

        print(f"Executing run_inversion with {len(args)} args and {len(kwargs)} kwargs...")
        agent_output = run_inversion(*args, **kwargs)

        # 3. Handle Chained Execution
        # If run_inversion returns a function (factory pattern), we execute it with inner data.
        # Otherwise, we use the result directly.
        
        final_result = agent_output
        final_std_result = std_output
        
        if inner_path and callable(agent_output):
            print("Detected Factory Pattern. Loading Inner Data...")
            inner_data = load_pkl(inner_path)
            i_args = inner_data.get('args', [])
            i_kwargs = inner_data.get('kwargs', {})
            final_std_result = inner_data.get('output')
            
            print("Executing Inner Function (Operator)...")
            final_result = agent_output(*i_args, **i_kwargs)
        elif inner_path and not callable(agent_output):
            print("Warning: Inner data provided but Outer result is not callable. Using Outer result directly.")

        # 4. Evaluation
        # Check if we have the necessary metadata for full evaluation (mesh_truth, etc.)
        # These would typically be in the kwargs of the function call if captured.
        
        meta_keys = ['mesh_truth', 'mesh_baseline', 'grid_info']
        # Check if keys exist in kwargs
        can_run_full_eval = all(k in kwargs for k in meta_keys)

        if can_run_full_eval:
            print("Context metadata found. Running full 'evaluate_results'...")
            mesh_truth = kwargs['mesh_truth']
            mesh_baseline = kwargs['mesh_baseline']
            grid_info = kwargs['grid_info']
            
            # Evaluate Agent
            print("\n--- Agent Evaluation ---")
            _, psnr_agent = evaluate_results(final_result, mesh_truth, mesh_baseline, grid_info, out_name='agent_recon.png')
            
            # Evaluate Standard (Control)
            print("\n--- Standard Evaluation ---")
            _, psnr_std = evaluate_results(final_std_result, mesh_truth, mesh_baseline, grid_info, out_name='std_recon.png')
            
            print(f"\nScore Comparison (PSNR): Agent={psnr_agent:.2f} dB, Standard={psnr_std:.2f} dB")
            
            # Success Criteria: Agent should not be significantly worse than Standard
            # Allow 5% margin or 1dB drop
            if psnr_agent >= (psnr_std - 1.0):
                print("SUCCESS: Quality matches or exceeds baseline.")
                sys.exit(0)
            else:
                print("FAILURE: Quality degraded significantly.")
                sys.exit(1)
        
        else:
            print("Metadata (mesh/grid) NOT found in captured arguments.")
            print("Falling back to consistency check: Agent Output vs Standard Output.")
            
            # Calculate MSE between Agent Result and Standard Result
            # Here, 'Standard Output' acts as the Ground Truth for the code's behavior
            
            diff = final_result - final_std_result
            mse = np.mean(diff**2)
            
            # Also compute a relative error
            norm_std = np.mean(final_std_result**2)
            if norm_std == 0: norm_std = 1.0
            rel_error = mse / norm_std
            
            print(f" Consistency MSE: {mse:.6e}")
            print(f" Relative Error: {rel_error:.6e}")
            
            # Thresholds
            if rel_error < 1e-5:
                print("SUCCESS: Agent output matches Standard output.")
                sys.exit(0)
            else:
                print("FAILURE: Agent output deviates from Standard.")
                sys.exit(1)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Default path from prompt
    default_path = '/data/yjh/nirfaster-FF-main_2_sandbox/run_code/std_data/standard_data_run_inversion.pkl'
    
    # Use arguments if provided, else default
    if len(sys.argv) > 1:
        data_paths = sys.argv[1:]
    else:
        data_paths = [default_path]
        
    run_test_logic(data_paths)