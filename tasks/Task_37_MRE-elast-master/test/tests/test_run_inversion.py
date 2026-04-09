import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Error: Could not import 'run_inversion' from 'agent_run_inversion.py'")
    sys.exit(1)

# -------------------------------------------------------------------------
# 1. INJECTED REFEREE / EVALUATION LOGIC
# -------------------------------------------------------------------------
def evaluate_results(E_true, E_recon, vertices):
    """
    Calculates CNR and RMSE, and plots the results.
    Note: In the context of this test, E_true is often the 'standard output' 
    recorded previously, or a synthetic ground truth if available in kwargs.
    """
    # Sanity Check for NaNs
    if np.any(np.isnan(E_recon)) or np.any(np.isinf(E_recon)):
        print("ERROR: Reconstructed E contains NaNs or Infs.")
        return -100.0, 1e9

    # Calculate Metrics
    thresh = 0.8 * np.max(E_recon)
    idxe = np.where(E_recon > thresh)
    idxb = np.where(E_recon < thresh)
    
    EE = E_recon[idxe[0]]
    BB = E_recon[idxb[0]]
    
    if len(EE) == 0 or len(BB) == 0:
        cnr = 0
    else:
        # Avoid divide by zero variance
        var_sum = np.var(EE) + np.var(BB)
        if var_sum < 1e-9:
            cnr = 0
        else:
            cnr = 10 * np.log10(2 * (np.mean(EE) - np.mean(BB))**2 / var_sum)
        
    # RMSE calculation (Relative error formulation)
    # Avoid divide by zero
    denom = E_true + E_recon + 1e-9
    rms = np.sqrt(np.mean(np.abs(2 * (E_recon - E_true) / denom)**2))
    
    print("\nEvaluation Results:")
    print(f"CNR: {cnr:.2f} dB")
    print(f"RMSE (Relative): {rms:.4f}")

    # Plotting
    try:
        x = np.array(vertices[:, 0])
        y = np.array(vertices[:, 1])
        x_new = np.linspace(x.min(), x.max(), 100)
        y_new = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x_new, y_new)
        
        E_true_im = griddata((x, y), E_true, (X, Y), method='linear')
        E_recon_im = griddata((x, y), E_recon, (X, Y), method='linear')
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(E_true_im, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        plt.colorbar()
        plt.title("Reference Stiffness (Std Output)")
        
        plt.subplot(1, 2, 2)
        plt.imshow(E_recon_im, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        plt.colorbar()
        plt.title(f"Reconstructed (CNR={cnr:.1f}dB)")
        
        plt.savefig('mre_refactored_result.png')
        print("Result saved to mre_refactored_result.png")
    except Exception as e:
        print(f"Plotting failed (non-critical): {e}")
    
    return cnr, rms

# -------------------------------------------------------------------------
# 2. TEST EXECUTION LOGIC
# -------------------------------------------------------------------------
def main():
    data_paths = ['/data/yjh/MRE-elast-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']

    # 1. Identify Outer and Inner Data
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if "parent_function" in path:
            inner_paths.append(path)
        else:
            outer_path = path

    if not outer_path:
        print("Error: No primary data file found (standard_data_run_inversion.pkl).")
        sys.exit(1)

    # 2. Load Outer Data
    print(f"Loading Primary Data: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        sys.exit(1)

    # 3. Execute Target Function
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Determine if we have a closure pattern or direct execution
        # Check if output is callable? No, run_inversion returns 'sol' (numpy array).
        # We assume Direct Execution based on provided function signature.

        print("\n--- Executing run_inversion ---")
        agent_result = run_inversion(*args, **kwargs)
        print("Execution successful.")
        
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Handle Results & Evaluation
    # In this specific case, run_inversion returns a numpy array (sol).
    # We compare agent_result against outer_data['output'].
    
    std_output = outer_data.get('output')
    
    # Extract vertices for plotting from args.
    # Signature: run_inversion(disp_measured, E_initial_guess, matTens, Tens, KT, triangles, vertices, fxy, params)
    # vertices is the 7th argument (index 6).
    vertices = None
    if len(args) > 6:
        vertices = args[6]
    elif 'vertices' in kwargs:
        vertices = kwargs['vertices']
    
    if vertices is None:
        print("Warning: Could not find 'vertices' in arguments. Plotting will be disabled.")
        # Create dummy vertices to prevent crash in evaluate_results
        N = agent_result.shape[0] if hasattr(agent_result, 'shape') else 100
        vertices = np.random.rand(N, 2)

    print("\n--- Evaluating Agent Performance ---")
    
    # Check for NaNs in Agent Result first
    if np.any(np.isnan(agent_result)):
        print("FAILURE: Agent result contains NaNs.")
        sys.exit(1)

    # Compare Agent vs Standard Output using the Referee
    # We treat std_output as "Ground Truth" for this regression test
    cnr_agent, rms_diff = evaluate_results(std_output, agent_result, vertices)

    # 5. Verification Logic
    # Since we are comparing against a recorded standard output of the SAME function,
    # the results should be very close.
    # However, optimization solvers can vary slightly across environments/dependency versions.
    
    print(f"\nComparing Agent Result against Standard Recorded Output:")
    print(f"  RMSE (deviation from standard): {rms_diff:.6f}")
    
    # Thresholds
    # RMSE should be very low (close to 0) if the logic is identical.
    # We allow a small tolerance for floating point differences.
    rmse_threshold = 0.05  # 5% relative difference allowed
    
    if rms_diff > rmse_threshold:
        print(f"FAILURE: Agent result deviated significantly from standard output (RMSE: {rms_diff:.6f} > {rmse_threshold}).")
        sys.exit(1)
    
    print("SUCCESS: Agent result matches standard output within tolerance.")
    sys.exit(0)

if __name__ == "__main__":
    main()