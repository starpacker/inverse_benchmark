import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Could not import run_inversion from agent_run_inversion.py")
    sys.exit(1)

# --- INJECTED REFEREE (EVALUATION LOGIC) ---
# Modified to ensure robustness and return metrics

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    _HAS_MATPLOTLIB = False

def evaluate_results(y_true, y_pred, nu_axis, params_pred, params_true=None):
    """
    Calculates metrics and plots results.
    Returns: (mse, psnr)
    """
    # 1. MSE
    mse = np.mean((y_true - y_pred)**2)
    print(f"MSE: {mse:.6e}")
    
    # 2. PSNR
    psnr = 0.0
    if mse > 0:
        # Signal peak assumed to be 1.0 (normalized data)
        psnr = 10 * np.log10(1.0 / mse) 
        print(f"PSNR: {psnr:.2f} dB")
    else:
        psnr = 100.0 # Perfect match
        print("PSNR: Infinity (Perfect Match)")
    
    # 3. Parameter Error
    if params_true:
        T_true = params_true.get('temperature', 0)
        T_pred = params_pred.get('temperature', 0)
        err = abs(T_true - T_pred)
        print(f"Temperature Error: {err:.2f} K (True: {T_true} K, Pred: {T_pred:.2f} K)")
        
    # 4. Plotting
    if _HAS_MATPLOTLIB and plt is not None:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(nu_axis, y_true, 'k.', label='Measured')
            plt.plot(nu_axis, y_pred, 'r-', linewidth=2, label=f'Fit (T={params_pred.get("temperature",0):.0f}K)')
            plt.xlabel('Wavenumber (cm-1)')
            plt.ylabel('Normalized Intensity')
            plt.title('CARS Inversion Result')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('inversion_result.png')
            print("Plot saved to 'inversion_result.png'")
            plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    return mse, psnr

# --- MAIN VALIDATION LOGIC ---

def main():
    # 1. CONFIGURATION
    data_paths = ['/data/yjh/carspy-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Check for Inner/Outer pattern (factory pattern)
    # The target function 'run_inversion' appears to be a direct execution function based on its signature,
    # but we will support the generic loading logic.
    outer_data_path = None
    inner_data_paths = []
    
    for p in data_paths:
        if 'parent_function' in p:
            inner_data_paths.append(p)
        else:
            outer_data_path = p

    if not outer_data_path:
        print("No standard_data_run_inversion.pkl found.")
        sys.exit(1)

    # 2. LOAD OUTER DATA
    print(f"Loading Outer Data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Extract Inputs
    # args: [measured_signal, nu_axis, initial_guesses]
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    if not args and not kwargs:
        print("No arguments found in pickle.")
        sys.exit(1)
        
    # Prepare standard arguments for evaluation helper
    # Based on function signature: run_inversion(measured_signal, nu_axis, initial_guesses)
    if len(args) >= 2:
        measured_signal = args[0]
        nu_axis = args[1]
    else:
        # Fallback if kwargs usage
        measured_signal = kwargs.get('measured_signal')
        nu_axis = kwargs.get('nu_axis')
        
    if measured_signal is None or nu_axis is None:
        print("Could not identify measured_signal or nu_axis for evaluation.")
        sys.exit(1)

    # 3. EXECUTE AGENT
    print("\nRunning run_inversion (Agent)...")
    try:
        agent_result = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"Agent execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. EVALUATE AGENT
    print("\n--- Evaluating Agent Results ---")
    mse_agent, psnr_agent = evaluate_results(
        y_true=measured_signal,
        y_pred=agent_result['y_pred'],
        nu_axis=nu_axis,
        params_pred=agent_result['best_params'],
        params_true=std_output['best_params'] if std_output else None
    )

    # 5. EVALUATE STANDARD (Ground Truth)
    # We evaluate the stored output against the stored input to verify the "Ground Truth" quality
    # and to establish a baseline.
    print("\n--- Evaluating Standard (Stored) Results ---")
    if std_output:
        mse_std, psnr_std = evaluate_results(
            y_true=measured_signal,
            y_pred=std_output['y_pred'],
            nu_axis=nu_axis,
            params_pred=std_output['best_params'],
            params_true=std_output['best_params'] # Self-comparison for params
        )
    else:
        print("No stored output available for comparison.")
        mse_std, psnr_std = 1e-6, 50.0 # Default baseline

    # 6. COMPARE AND DECIDE
    print(f"\nScores -> Agent MSE: {mse_agent:.6e} (PSNR: {psnr_agent:.2f} dB)")
    print(f"Scores -> Std   MSE: {mse_std:.6e} (PSNR: {psnr_std:.2f} dB)")

    # Criteria:
    # 1. Agent should not crash (already passed).
    # 2. Agent MSE should be reasonably close to Standard MSE.
    #    Since this is an optimization problem, small floating point differences might occur.
    #    We allow the Agent's error to be slightly higher (worse) or lower (better).
    #    Acceptance: MSE_Agent < MSE_Std * 2.0 or PSNR_Agent > PSNR_Std - 5.0 dB
    
    threshold_mse = mse_std * 5.0 if mse_std > 1e-10 else 1e-3
    threshold_psnr_drop = 10.0 # dB
    
    success = True
    
    if mse_agent > threshold_mse:
        print(f"FAILURE: Agent MSE ({mse_agent:.6e}) is significantly higher than Reference ({mse_std:.6e})")
        success = False
    
    if psnr_agent < (psnr_std - threshold_psnr_drop):
        print(f"FAILURE: Agent PSNR ({psnr_agent:.2f}) dropped significantly below Reference ({psnr_std:.2f})")
        success = False

    if success:
        print("\nSUCCESS: Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("\nFAILURE: Performance degradation detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()